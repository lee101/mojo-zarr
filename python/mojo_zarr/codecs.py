from __future__ import annotations

import sys
import zlib
from typing import Any

import numpy as np

from ._lib import address, ffi_length, lib


def _u8(buf: Any) -> np.ndarray:
    if isinstance(buf, np.ndarray):
        return np.ascontiguousarray(buf).view(np.uint8).reshape(-1)
    try:
        return np.frombuffer(buf, dtype=np.uint8)
    except TypeError:
        return np.ascontiguousarray(buf).view(np.uint8).reshape(-1)


def _copy_array(result: np.ndarray, out: Any | None) -> np.ndarray:
    if out is None:
        return result
    target = np.asarray(out)
    if not target.flags.c_contiguous or not target.flags.writeable:
        raise ValueError("destination buffer must be writable and C-contiguous")
    if target.nbytes != result.nbytes:
        raise ValueError("destination buffer has the wrong size")
    target.view(np.uint8).reshape(-1)[: result.nbytes] = result.view(np.uint8).reshape(-1)
    return target


def _output_u8(out: Any, size: int) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(out)
    if not target.flags.c_contiguous or not target.flags.writeable:
        raise ValueError("destination buffer must be writable and C-contiguous")
    if target.nbytes != size:
        raise ValueError("destination buffer has the wrong size")
    return target.view(np.uint8).reshape(-1)[:size], target


class Codec:
    codec_id = ""

    def get_config(self) -> dict[str, Any]:
        return {"id": self.codec_id}


class Delta(Codec):
    codec_id = "delta"

    def __init__(self, dtype: Any, astype: Any | None = None):
        self.dtype = np.dtype(dtype)
        self.astype = self.dtype if astype is None else np.dtype(astype)
        if self.dtype.hasobject or self.astype.hasobject:
            raise ValueError("object arrays are not supported")

    def _mojo_kind(self) -> int:
        dtype = self.dtype
        native = dtype.byteorder in ("=", "|") or (
            dtype.byteorder == "<" and sys.byteorder == "little"
        ) or (dtype.byteorder == ">" and sys.byteorder == "big")
        if self.astype != dtype or not native:
            return 0
        if dtype.kind in "iu":
            return dtype.itemsize
        if dtype.kind == "f" and dtype.itemsize in (4, 8):
            return 10 + dtype.itemsize
        return 0

    def encode(self, buf: Any) -> np.ndarray:
        base = np.frombuffer(buf, dtype=np.uint8) if isinstance(
            buf, (bytes, bytearray, memoryview)
        ) else np.asarray(buf)
        if base.nbytes % self.dtype.itemsize:
            raise ValueError("Delta input size is not a multiple of dtype itemsize")
        arr = base.view(self.dtype).reshape(-1, order="A")
        if arr.size == 0:
            raise IndexError("index 0 is out of bounds for axis 0 with size 0")
        kind = self._mojo_kind()
        if not kind:
            encoded = np.empty_like(arr, dtype=self.astype)
            encoded[0] = arr[0]
            encoded[1:] = np.diff(arr)
            return encoded
        src = np.ascontiguousarray(arr)
        encoded = np.empty(arr.size, dtype=self.astype)
        lib().mz_delta_encode(
            address(src), address(encoded), ffi_length(arr.size), kind
        )
        return encoded

    def decode(self, buf: Any, out: Any | None = None) -> np.ndarray:
        base = np.frombuffer(buf, dtype=np.uint8) if isinstance(
            buf, (bytes, bytearray, memoryview)
        ) else np.asarray(buf)
        if base.nbytes % self.astype.itemsize:
            raise ValueError("Delta input size is not a multiple of astype itemsize")
        encoded = base.view(self.astype).reshape(-1, order="A")
        kind = self._mojo_kind()
        target = None if out is None else np.asarray(out)
        if target is not None and (
            not target.flags.c_contiguous or not target.flags.writeable
        ):
            raise ValueError("destination buffer must be writable and C-contiguous")
        if (
            target is not None
            and target.nbytes != encoded.size * self.dtype.itemsize
        ):
            raise ValueError("destination buffer has the wrong size")
        direct = (
            target is not None
            and target.flags.c_contiguous
            and target.flags.writeable
        )
        if direct:
            decoded = (
                target.view(np.uint8)
                .reshape(-1)[: encoded.size * self.dtype.itemsize]
                .view(self.dtype)
            )
        elif encoded.size == 0:
            decoded = np.empty(0, dtype=self.dtype)
        else:
            decoded = np.empty(encoded.size, dtype=self.dtype)
        if encoded.size == 0:
            pass
        elif not kind:
            np.cumsum(encoded, out=decoded)
        else:
            src = np.ascontiguousarray(encoded)
            lib().mz_delta_decode(
                address(src), address(decoded), ffi_length(encoded.size), kind
            )
        return target if direct else _copy_array(decoded, out)

    def get_config(self) -> dict[str, Any]:
        return {
            "id": self.codec_id,
            "dtype": self.dtype.str,
            "astype": self.astype.str,
        }

    def __repr__(self) -> str:
        extra = "" if self.astype == self.dtype else f", astype={self.astype.str!r}"
        return f"Delta(dtype={self.dtype.str!r}{extra})"


class Shuffle(Codec):
    codec_id = "shuffle"

    def __init__(self, elementsize: int = 4):
        if elementsize < 1:
            raise ValueError("elementsize must be positive")
        self.elementsize = int(elementsize)

    def _prepare(
        self, buf: Any, out: Any | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        src = _u8(buf)
        if src.nbytes % self.elementsize:
            raise ValueError("Shuffle buffer is not an integer multiple of elementsize")
        if out is None:
            result = np.empty(src.nbytes, dtype=np.uint8)
            dst = result
        else:
            dst, result = _output_u8(out, src.nbytes)
        if np.shares_memory(src, dst):
            src = src.copy()
        return src, dst, result

    def encode(self, buf: Any, out: Any | None = None) -> np.ndarray:
        src, dst, result = self._prepare(buf, out)
        if src.size == 0:
            return result
        if self.elementsize == 1:
            dst[: src.size] = src
        else:
            lib().mz_shuffle(
                address(src), address(dst), ffi_length(src.nbytes), self.elementsize
            )
        return result

    def decode(self, buf: Any, out: Any | None = None) -> np.ndarray:
        src, dst, result = self._prepare(buf, out)
        if src.size == 0:
            return result
        if self.elementsize == 1:
            dst[: src.size] = src
        else:
            lib().mz_unshuffle(
                address(src), address(dst), ffi_length(src.nbytes), self.elementsize
            )
        return result

    def get_config(self) -> dict[str, Any]:
        return {"id": self.codec_id, "elementsize": self.elementsize}

    def __repr__(self) -> str:
        return f"Shuffle(elementsize={self.elementsize})"


class PackBits(Codec):
    codec_id = "packbits"

    def encode(self, buf: Any) -> np.ndarray:
        source = np.asarray(buf)
        if source.dtype != np.dtype(bool):
            raise TypeError("PackBits input must have boolean dtype")
        src = np.ascontiguousarray(source).reshape(-1).view(np.uint8)
        packed_bytes = (src.size + 7) // 8
        encoded = np.empty(packed_bytes + 1, dtype=np.uint8)
        encoded[0] = (8 - src.size % 8) % 8
        if src.size:
            lib().mz_packbits(
                address(src), address(encoded[1:]), ffi_length(src.size)
            )
        return encoded

    def decode(self, buf: Any, out: Any | None = None) -> np.ndarray:
        encoded = _u8(buf)
        if encoded.size == 0:
            raise ValueError("encoded PackBits buffer is empty")
        padding = int(encoded[0])
        if padding > 7:
            raise ValueError("invalid PackBits padding")
        n = (encoded.size - 1) * 8 - padding
        decoded = np.empty(n, dtype=bool)
        if n:
            lib().mz_unpackbits(
                address(encoded[1:]),
                address(decoded.view(np.uint8)),
                ffi_length(n),
            )
        return _copy_array(decoded, out)

    def __repr__(self) -> str:
        return "PackBits()"


class CRC32(Codec):
    codec_id = "crc32"

    def __init__(self, location: str | None = None):
        self.location = "start" if location is None else location
        if self.location not in ("start", "end"):
            raise ValueError(f"Invalid checksum location: {self.location}")

    @staticmethod
    def checksum(data: Any, value: int = 0) -> int:
        src = _u8(data)
        return zlib.crc32(src, value)

    def encode(self, buf: Any) -> np.ndarray:
        src = _u8(buf)
        encoded = np.empty(src.size + 4, dtype=np.uint8)
        check = self.checksum(src).to_bytes(4, "little")
        if self.location == "start":
            encoded[:4] = np.frombuffer(check, dtype=np.uint8)
            encoded[4:] = src
        else:
            encoded[:-4] = src
            encoded[-4:] = np.frombuffer(check, dtype=np.uint8)
        return encoded

    def decode(self, buf: Any, out: Any | None = None) -> np.ndarray:
        encoded = _u8(buf)
        if encoded.size < 4:
            raise ValueError("Input buffer is too short to contain a 32-bit checksum.")
        check_bytes = encoded[:4] if self.location == "start" else encoded[-4:]
        payload = encoded[4:] if self.location == "start" else encoded[:-4]
        expected = int.from_bytes(check_bytes.tobytes(), "little")
        actual = self.checksum(payload)
        if expected != actual:
            raise RuntimeError(
                "Stored and computed crc32 checksum do not match. "
                f"Stored: {expected}. Computed: {actual}."
            )
        return _copy_array(payload.copy(), out)

    def get_config(self) -> dict[str, Any]:
        config = {"id": self.codec_id}
        if self.location != "start":
            config["location"] = self.location
        return config

    def __repr__(self) -> str:
        suffix = "" if self.location == "start" else f"location={self.location!r}"
        return f"CRC32({suffix})"


class Zlib(Codec):
    codec_id = "zlib"

    def __init__(self, level: int = 1):
        if not -1 <= level <= 9:
            raise ValueError("level must be between -1 and 9")
        self.level = int(level)

    def encode(self, buf: Any) -> bytes:
        return zlib.compress(_u8(buf), self.level)

    def decode(self, buf: Any, out: Any | None = None) -> bytes | np.ndarray:
        decoded = zlib.decompress(buf)
        if out is None:
            return decoded
        result = np.frombuffer(decoded, dtype=np.uint8)
        return _copy_array(result, out)

    def get_config(self) -> dict[str, Any]:
        return {"id": self.codec_id, "level": self.level}

    def __repr__(self) -> str:
        return f"Zlib(level={self.level})"


_CODECS = {
    "delta": Delta,
    "shuffle": Shuffle,
    "packbits": PackBits,
    "crc32": CRC32,
    "zlib": Zlib,
}


def get_codec(config: Codec | dict[str, Any] | None) -> Codec | None:
    if config is None or isinstance(config, Codec):
        return config
    if all(hasattr(config, name) for name in ("encode", "decode", "get_config")):
        return config
    if not isinstance(config, dict) or "id" not in config:
        raise TypeError("codec must be a Codec or a configuration mapping")
    values = dict(config)
    codec_id = values.pop("id")
    try:
        return _CODECS[codec_id](**values)
    except KeyError as exc:
        raise ValueError(f"unsupported codec: {codec_id!r}") from exc
