from __future__ import annotations

import itertools
import json
import math
import numbers
import os
from collections.abc import Iterator, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .codecs import Codec, Zlib, get_codec
from .storage import normalize_store

_PARALLEL_CHUNK_COUNT = 8
_PARALLEL_CHUNK_BYTES = 4 << 20
_MAX_CHUNK_WORKERS = min(8, os.cpu_count() or 1)


def _join(path: str, name: str) -> str:
    return "/".join(part for part in (path.strip("/"), name) if part)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, allow_nan=True).encode()


def _auto_chunks(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    if not shape:
        return ()
    chunks = [max(1, n) for n in shape]
    target = 1 << 20
    while math.prod(chunks) * itemsize > target:
        axis = max(range(len(chunks)), key=chunks.__getitem__)
        chunks[axis] = max(1, (chunks[axis] + 1) // 2)
    return tuple(chunks)


def _normalize_shape(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, numbers.Integral):
        shape = (int(shape),)
    result = tuple(int(n) for n in shape)
    if any(n < 0 for n in result):
        raise ValueError("shape dimensions must be non-negative")
    return result


def _normalize_chunks(
    chunks: Any, shape: tuple[int, ...], itemsize: int
) -> tuple[int, ...]:
    if chunks in (None, "auto"):
        return _auto_chunks(shape, itemsize)
    if isinstance(chunks, numbers.Integral):
        chunks = (int(chunks),) * len(shape)
    result = tuple(int(n) for n in chunks)
    if len(result) != len(shape) or any(n <= 0 for n in result):
        raise ValueError("chunks must have one positive value per dimension")
    return result


def _codec_config(codec: Codec | None) -> dict[str, Any] | None:
    return None if codec is None else codec.get_config()


class Attributes(MutableMapping[str, Any]):
    def __init__(self, array: Array):
        self._array = array

    def _read(self) -> dict[str, Any]:
        try:
            return json.loads(self._array.store[self._array._attrs_key])
        except KeyError:
            return {}

    def _write(self, values: dict[str, Any]) -> None:
        self._array._check_writable()
        self._array.store[self._array._attrs_key] = _json_bytes(values)

    def __getitem__(self, key: str) -> Any:
        return self._read()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        values = self._read()
        values[key] = value
        self._write(values)

    def __delitem__(self, key: str) -> None:
        values = self._read()
        del values[key]
        self._write(values)

    def __iter__(self) -> Iterator[str]:
        return iter(self._read())

    def __len__(self) -> int:
        return len(self._read())

    def asdict(self) -> dict[str, Any]:
        return self._read()


class Array:
    def __init__(
        self,
        store: MutableMapping[str, bytes],
        path: str,
        metadata: dict[str, Any],
        *,
        read_only: bool = False,
    ):
        self.store = store
        self.path = path.strip("/")
        if metadata.get("zarr_format") != 2:
            raise NotImplementedError("the covered storage format is Zarr v2")
        self.shape = _normalize_shape(metadata["shape"])
        self.dtype = np.dtype(metadata["dtype"])
        if self.dtype.hasobject or self.dtype.kind not in "biuf":
            raise ValueError(
                "the covered dtype subset is boolean, integer, and floating point"
            )
        self.chunks = _normalize_chunks(
            metadata["chunks"], self.shape, self.dtype.itemsize
        )
        self.fill_value = metadata.get("fill_value", 0)
        if self.fill_value is None:
            self.fill_value = 0
        self.order = metadata.get("order", "C")
        if self.order != "C":
            raise NotImplementedError("only C-order arrays are supported")
        self.filters = tuple(
            get_codec(config) for config in (metadata.get("filters") or ())
        )
        self.compressor = get_codec(metadata.get("compressor"))
        self.dimension_separator = metadata.get("dimension_separator", ".")
        if self.dimension_separator != ".":
            raise NotImplementedError("only '.' chunk-key separators are supported")
        self.read_only = read_only or bool(getattr(store, "read_only", False))
        self.attrs = Attributes(self)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @property
    def nchunks(self) -> int:
        if any(n == 0 for n in self.shape):
            return 0
        return math.prod(
            (n + c - 1) // c for n, c in zip(self.shape, self.chunks)
        )

    @property
    def _metadata_key(self) -> str:
        return _join(self.path, ".zarray")

    @property
    def _attrs_key(self) -> str:
        return _join(self.path, ".zattrs")

    def _check_writable(self) -> None:
        if self.read_only:
            raise PermissionError("array is read-only")

    def _chunk_key(self, coord: tuple[int, ...]) -> str:
        leaf = (
            self.dimension_separator.join(map(str, coord)) if coord else "0"
        )
        return _join(self.path, leaf)

    def _empty_chunk(self) -> np.ndarray:
        return np.full(self.chunks, self.fill_value, dtype=self.dtype, order="C")

    def _encode_chunk(self, chunk: np.ndarray) -> bytes:
        encoded: Any = np.ascontiguousarray(chunk, dtype=self.dtype)
        for codec in self.filters:
            encoded = codec.encode(encoded)
        if self.compressor is not None:
            encoded = self.compressor.encode(encoded)
        if isinstance(encoded, bytes):
            return encoded
        return _u8_bytes(encoded)

    def _decode_chunk(self, data: bytes) -> np.ndarray:
        decoded: Any = data
        if self.compressor is not None:
            decoded = self.compressor.decode(decoded)
        final = None
        reverse_filters = tuple(reversed(self.filters))
        if reverse_filters:
            final = np.empty(self.chunks, dtype=self.dtype, order="C")
        for index, codec in enumerate(reverse_filters):
            decoded = codec.decode(
                decoded,
                out=final if index + 1 == len(reverse_filters) else None,
            )
        expected_nbytes = math.prod(self.chunks) * self.dtype.itemsize
        actual_nbytes = memoryview(decoded).nbytes
        if actual_nbytes != expected_nbytes:
            raise ValueError(
                f"decoded chunk has {actual_nbytes} bytes; expected {expected_nbytes}"
            )
        if final is not None:
            return final
        array = np.frombuffer(decoded, dtype=self.dtype, count=math.prod(self.chunks))
        return array.copy().reshape(self.chunks, order="C")

    def _load_chunk(self, coord: tuple[int, ...]) -> np.ndarray:
        try:
            return self._decode_chunk(self.store[self._chunk_key(coord)])
        except KeyError:
            return self._empty_chunk()

    def _store_chunk(self, coord: tuple[int, ...], chunk: np.ndarray) -> None:
        key = self._chunk_key(coord)
        fill = np.full((), self.fill_value, dtype=self.dtype)[()]
        if np.all(np.equal(chunk, fill)) or (
            np.issubdtype(self.dtype, np.inexact)
            and np.isnan(fill)
            and np.all(np.isnan(chunk))
        ):
            if key in self.store:
                del self.store[key]
            return
        self.store[key] = self._encode_chunk(chunk)

    def _normalize_index(self, selection: Any) -> tuple[int | slice, ...]:
        if not isinstance(selection, tuple):
            selection = (selection,)
        ellipses = sum(item is Ellipsis for item in selection)
        if ellipses > 1:
            raise IndexError("an index can only have a single ellipsis")
        if any(item is None for item in selection):
            raise IndexError("newaxis indexing is not in the covered subset")
        if ellipses:
            position = selection.index(Ellipsis)
            missing = self.ndim - (len(selection) - 1)
            selection = (
                selection[:position]
                + (slice(None),) * missing
                + selection[position + 1 :]
            )
        if len(selection) > self.ndim:
            raise IndexError("too many indices for array")
        selection += (slice(None),) * (self.ndim - len(selection))
        result: list[int | slice] = []
        for axis, item in enumerate(selection):
            if isinstance(item, numbers.Integral):
                index = int(item)
                if index < 0:
                    index += self.shape[axis]
                if not 0 <= index < self.shape[axis]:
                    raise IndexError("array index out of bounds")
                result.append(index)
            elif isinstance(item, slice):
                start, stop, step = item.indices(self.shape[axis])
                result.append(slice(start, stop, step))
            else:
                raise IndexError("only integers, slices, and ellipsis are supported")
        return tuple(result)

    def _bounds(
        self, selection: tuple[int | slice, ...]
    ) -> tuple[tuple[int | slice, ...], list[np.ndarray], tuple[int, ...]]:
        bounds: list[int | slice] = []
        positions: list[np.ndarray] = []
        result_shape: list[int] = []
        for item in selection:
            if isinstance(item, int):
                bounds.append(item)
                continue
            indices = np.arange(item.start, item.stop, item.step, dtype=np.intp)
            result_shape.append(indices.size)
            if indices.size == 0:
                bounds.append(slice(0, 0, 1))
                positions.append(indices)
            else:
                low, high = int(indices.min()), int(indices.max()) + 1
                bounds.append(slice(low, high, 1))
                positions.append(indices - low)
        return tuple(bounds), positions, tuple(result_shape)

    def _coords_for(
        self, bounds: tuple[int | slice, ...]
    ) -> Iterator[tuple[int, ...]]:
        axes = []
        for item, chunk in zip(bounds, self.chunks):
            if isinstance(item, int):
                axes.append(range(item // chunk, item // chunk + 1))
            elif item.start == item.stop:
                return iter(())
            else:
                axes.append(range(item.start // chunk, (item.stop - 1) // chunk + 1))
        return itertools.product(*axes)

    def _read_region(self, bounds: tuple[int | slice, ...]) -> np.ndarray:
        result_shape = tuple(
            item.stop - item.start for item in bounds if isinstance(item, slice)
        )
        result = np.full(result_shape, self.fill_value, dtype=self.dtype)
        coords = tuple(self._coords_for(bounds))

        def read_chunk(coord: tuple[int, ...]) -> None:
            chunk = self._load_chunk(coord)
            chunk_sel: list[int | slice] = []
            result_sel: list[slice] = []
            for item, c, width in zip(bounds, coord, self.chunks):
                origin = c * width
                if isinstance(item, int):
                    chunk_sel.append(item - origin)
                else:
                    low, high = max(item.start, origin), min(item.stop, origin + width)
                    chunk_sel.append(slice(low - origin, high - origin))
                    result_sel.append(slice(low - item.start, high - item.start))
            result[tuple(result_sel)] = chunk[tuple(chunk_sel)]

        if self._parallel_chunks(len(coords)):
            with ThreadPoolExecutor(
                max_workers=min(_MAX_CHUNK_WORKERS, len(coords))
            ) as executor:
                tuple(executor.map(read_chunk, coords))
        else:
            for coord in coords:
                read_chunk(coord)
        return result

    def _write_region(
        self, bounds: tuple[int | slice, ...], value: Any
    ) -> None:
        target_shape = tuple(
            item.stop - item.start for item in bounds if isinstance(item, slice)
        )
        source = np.broadcast_to(np.asarray(value, dtype=self.dtype), target_shape)
        coords = tuple(self._coords_for(bounds))

        def write_chunk(coord: tuple[int, ...]) -> None:
            chunk_sel: list[int | slice] = []
            source_sel: list[slice] = []
            for item, c, width in zip(bounds, coord, self.chunks):
                origin = c * width
                if isinstance(item, int):
                    chunk_sel.append(item - origin)
                else:
                    low, high = max(item.start, origin), min(item.stop, origin + width)
                    chunk_sel.append(slice(low - origin, high - origin))
                    source_sel.append(slice(low - item.start, high - item.start))
            whole_chunk = len(source_sel) == self.ndim and all(
                isinstance(item, slice)
                and item.start == 0
                and item.stop == width
                for item, width in zip(chunk_sel, self.chunks)
            )
            if whole_chunk:
                chunk = np.ascontiguousarray(
                    source[tuple(source_sel)], dtype=self.dtype
                )
            else:
                chunk = self._load_chunk(coord)
                chunk[tuple(chunk_sel)] = source[tuple(source_sel)]
            self._store_chunk(coord, chunk)

        if self._parallel_chunks(len(coords)):
            with ThreadPoolExecutor(
                max_workers=min(_MAX_CHUNK_WORKERS, len(coords))
            ) as executor:
                tuple(executor.map(write_chunk, coords))
        else:
            for coord in coords:
                write_chunk(coord)

    def _parallel_chunks(self, count: int) -> bool:
        chunk_bytes = math.prod(self.chunks) * self.dtype.itemsize
        return (
            _MAX_CHUNK_WORKERS > 1
            and count >= _PARALLEL_CHUNK_COUNT
            and count * chunk_bytes >= _PARALLEL_CHUNK_BYTES
        )

    def __getitem__(self, selection: Any) -> Any:
        normalized = self._normalize_index(selection)
        bounds, positions, result_shape = self._bounds(normalized)
        if any(n == 0 for n in result_shape):
            return np.empty(result_shape, dtype=self.dtype)
        result = self._read_region(bounds)
        for axis, indices in enumerate(positions):
            result = np.take(result, indices, axis=axis)
        return result[()] if result.ndim == 0 else result

    def __setitem__(self, selection: Any, value: Any) -> None:
        self._check_writable()
        normalized = self._normalize_index(selection)
        bounds, positions, result_shape = self._bounds(normalized)
        if any(n == 0 for n in result_shape):
            return
        if all(
            isinstance(item, int) or item.step == 1 for item in normalized
        ):
            self._write_region(bounds, value)
            return
        region = self._read_region(bounds)
        if positions:
            region[np.ix_(*positions)] = value
        else:
            region[()] = value
        self._write_region(bounds, region)

    def __array__(
        self, dtype: Any | None = None, copy: bool | None = None
    ) -> np.ndarray:
        result = self[...] if self.ndim == 0 else self[:]
        result = np.asarray(result)
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        if copy:
            result = result.copy()
        return result

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __repr__(self) -> str:
        return (
            f"<Array path={self.path!r} shape={self.shape} "
            f"chunks={self.chunks} dtype={self.dtype}>"
        )


def _u8_bytes(value: Any) -> bytes:
    array = np.ascontiguousarray(value)
    return array.view(np.uint8).reshape(-1).tobytes()


def _clear_array(store: MutableMapping[str, bytes], path: str) -> None:
    prefix = path.strip("/")
    prefix = prefix + "/" if prefix else ""
    for key in list(store):
        relative = key[len(prefix) :] if key.startswith(prefix) else None
        if relative is not None and "/" not in relative:
            del store[key]


def create_array(
    store: Any = None,
    *,
    name: str | None = None,
    shape: Any | None = None,
    dtype: Any | None = None,
    data: Any | None = None,
    chunks: Any = "auto",
    filters: Any = "auto",
    compressors: Any = "auto",
    compressor: Any = "auto",
    fill_value: Any = 0,
    order: str = "C",
    zarr_format: int | None = 2,
    attributes: dict[str, Any] | None = None,
    overwrite: bool = False,
    path: str | None = None,
    **kwargs: Any,
) -> Array:
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported create_array arguments: {unsupported}")
    if zarr_format not in (None, 2):
        raise NotImplementedError("the covered storage format is Zarr v2")
    if order != "C":
        raise NotImplementedError("only C-order arrays are supported")
    target = normalize_store(store)
    array_path = (path if path is not None else name) or ""
    metadata_key = _join(array_path, ".zarray")
    if metadata_key in target and not overwrite:
        raise FileExistsError(f"array already exists at {array_path!r}")
    if overwrite:
        _clear_array(target, array_path)
    source = None if data is None else np.asarray(data)
    if shape is None:
        if source is None:
            raise TypeError("shape is required when data is not provided")
        shape = source.shape
    shape = _normalize_shape(shape)
    if source is not None and tuple(source.shape) != shape:
        raise ValueError("data shape does not match shape")
    if dtype is None:
        dtype = source.dtype if source is not None else np.dtype("f8")
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError("object arrays are not supported")
    if dtype.kind not in "biuf":
        raise ValueError("the covered dtype subset is boolean, integer, and floating point")
    chunks = _normalize_chunks(chunks, shape, dtype.itemsize)
    if filters == "auto":
        filter_codecs: tuple[Codec, ...] = ()
    else:
        filter_codecs = tuple(get_codec(value) for value in (filters or ()))
    selected = compressor if compressor != "auto" else compressors
    if selected == "auto":
        compressor_codec = Zlib(level=1)
    elif isinstance(selected, (list, tuple)):
        if len(selected) > 1:
            raise ValueError("Zarr v2 supports one compressor")
        compressor_codec = get_codec(selected[0]) if selected else None
    else:
        compressor_codec = get_codec(selected)
    metadata = {
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": dtype.str,
        "fill_value": np.asarray(fill_value, dtype=dtype).item(),
        "order": "C",
        "filters": [_codec_config(codec) for codec in filter_codecs] or None,
        "dimension_separator": ".",
        "compressor": _codec_config(compressor_codec),
        "zarr_format": 2,
    }
    target[metadata_key] = _json_bytes(metadata)
    target[_join(array_path, ".zattrs")] = _json_bytes(attributes or {})
    result = Array(target, array_path, metadata)
    if source is not None:
        result[...] = source
    return result


def open_array(
    store: Any = None,
    *,
    zarr_format: int | None = None,
    path: str = "",
    mode: str = "a",
    shape: Any | None = None,
    chunks: Any = "auto",
    dtype: Any | None = None,
    **kwargs: Any,
) -> Array:
    target = normalize_store(store)
    metadata_key = _join(path, ".zarray")
    exists = metadata_key in target
    if mode not in ("r", "r+", "a", "w", "w-", "x"):
        raise ValueError(f"invalid mode: {mode!r}")
    if mode == "w":
        return create_array(
            target,
            path=path,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            zarr_format=zarr_format or 2,
            overwrite=True,
            **kwargs,
        )
    if mode in ("w-", "x"):
        if exists:
            raise FileExistsError(f"array already exists at {path!r}")
        return create_array(
            target,
            path=path,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            zarr_format=zarr_format or 2,
            **kwargs,
        )
    if not exists:
        if mode in ("r", "r+"):
            raise FileNotFoundError(f"array does not exist at {path!r}")
        return create_array(
            target,
            path=path,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            zarr_format=zarr_format or 2,
            **kwargs,
        )
    metadata = json.loads(target[metadata_key])
    if zarr_format not in (None, 2):
        raise NotImplementedError("the covered storage format is Zarr v2")
    return Array(target, path, metadata, read_only=mode == "r")


def array(data: Any, **kwargs: Any) -> Array:
    return create_array(data=np.asarray(data), **kwargs)


def zeros(shape: Any, **kwargs: Any) -> Array:
    return create_array(shape=shape, fill_value=0, **kwargs)


def ones(shape: Any, **kwargs: Any) -> Array:
    result = create_array(shape=shape, fill_value=1, **kwargs)
    return result


def full(shape: Any, fill_value: Any, **kwargs: Any) -> Array:
    return create_array(shape=shape, fill_value=fill_value, **kwargs)


def empty(shape: Any, **kwargs: Any) -> Array:
    return create_array(shape=shape, fill_value=0, **kwargs)


def open(store: Any = None, *, mode: str = "a", path: str = "", **kwargs: Any) -> Array:
    return open_array(store, mode=mode, path=path, **kwargs)


def save(store: Any, *args: Any, path: str | None = None, **kwargs: Any) -> None:
    if len(args) != 1:
        raise NotImplementedError("the covered save subset writes one array")
    create_array(
        store,
        path=path or "",
        data=np.asarray(args[0]),
        overwrite=True,
        **kwargs,
    )


def load(store: Any, path: str | None = None, **kwargs: Any) -> np.ndarray:
    return np.asarray(open_array(store, path=path or "", mode="r", **kwargs))
