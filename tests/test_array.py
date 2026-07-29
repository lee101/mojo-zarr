import json

import numcodecs
import numpy as np
import pytest
import zarr

import mojo_zarr as mz
import mojo_zarr.core as core


rng = np.random.default_rng(7)


def test_our_store_is_readable_by_upstream(tmp_path):
    path = tmp_path / "ours.zarr"
    data = rng.integers(-1000, 1000, size=(37, 29), dtype=np.int32)
    array = mz.create_array(
        path,
        shape=data.shape,
        chunks=(11, 7),
        dtype=data.dtype,
        filters=[mz.Delta("<i4"), mz.Shuffle(4)],
        compressor=mz.Zlib(3),
    )
    array[:] = data
    reference = zarr.open_array(path, mode="r", zarr_format=2)
    assert np.array_equal(reference[:], data)
    assert reference.chunks == (11, 7)
    assert reference.fill_value == 0


def test_upstream_store_is_readable_by_us(tmp_path):
    path = tmp_path / "upstream.zarr"
    data = rng.normal(size=(23, 17, 9)).astype("f4")
    reference = zarr.create_array(
        path,
        shape=data.shape,
        chunks=(8, 6, 4),
        dtype="<f4",
        zarr_format=2,
        filters=[numcodecs.Delta("<f4"), numcodecs.Shuffle(4)],
        compressors=[numcodecs.Zlib(1)],
    )
    reference[:] = data
    ours = mz.open_array(path, mode="r")
    assert np.allclose(ours[:], reference[:], rtol=1e-6, atol=1e-6)
    assert ours.shape == reference.shape
    assert ours.chunks == reference.chunks


def test_packbits_boolean_chunks_interoperate(tmp_path):
    path = tmp_path / "bool.zarr"
    data = rng.integers(0, 2, size=(35, 27), dtype=np.uint8).astype(bool)
    ours = mz.create_array(
        path,
        data=data,
        chunks=(9, 8),
        filters=[mz.PackBits()],
        compressor=mz.CRC32(),
    )
    assert np.array_equal(ours[:], data)
    assert np.array_equal(zarr.open_array(path, mode="r", zarr_format=2)[:], data)


def test_basic_and_stepped_indexing_matches_numpy(tmp_path):
    data = np.arange(18 * 15 * 11, dtype=np.int32).reshape(18, 15, 11)
    array = mz.array(data, store=tmp_path / "a.zarr", chunks=(5, 4, 3))
    selections = [
        np.s_[2:17, 3:13, 1:10],
        np.s_[4, :, -1],
        np.s_[::-2, 1:14:3, 8:2:-2],
        np.s_[..., 5],
        np.s_[0:0, :, :],
    ]
    for selection in selections:
        assert np.array_equal(array[selection], data[selection])


def test_partial_and_strided_assignment_matches_numpy(tmp_path):
    expected = np.zeros((19, 17), dtype=np.float64)
    array = mz.zeros(
        expected.shape, store=tmp_path / "assign.zarr", chunks=(6, 5), dtype="f8"
    )
    values = rng.normal(size=(5, 5))
    expected[2:17:3, 1:16:3] = values
    array[2:17:3, 1:16:3] = values
    expected[-1, :] = 4.5
    array[-1, :] = 4.5
    assert np.array_equal(array[:], expected)


def test_fill_chunks_are_sparse(tmp_path):
    path = tmp_path / "sparse.zarr"
    array = mz.full((100, 100), 7, store=path, chunks=(10, 10), dtype="i2")
    assert np.all(array[:] == 7)
    assert not list(path.glob("[0-9]*"))
    array[15, 15] = 8
    assert (path / "1.1").exists()
    array[15, 15] = 7
    assert not (path / "1.1").exists()


def test_memory_store_roundtrip_and_reopen():
    store = mz.MemoryStore()
    data = np.arange(120, dtype=np.uint16).reshape(12, 10)
    array = mz.array(data, store=store, chunks=(5, 4), compressor=None)
    reopened = mz.open_array(store, mode="r")
    assert np.array_equal(reopened[:], data)
    with pytest.raises(PermissionError):
        reopened[0] = 1


def test_scalar_array_roundtrip():
    array = mz.array(np.array(3.5), chunks=(), compressor=None)
    assert array.shape == ()
    assert array[()] == 3.5
    array[...] = -2.25
    assert np.asarray(array)[()] == -2.25


def test_accepts_upstream_codec_instances(tmp_path):
    data = np.arange(100, dtype=np.int16).reshape(10, 10)
    array = mz.array(
        data,
        store=tmp_path / "codecs.zarr",
        chunks=(4, 5),
        filters=[numcodecs.Delta("<i2")],
        compressor=numcodecs.Zlib(1),
    )
    assert np.array_equal(array[:], data)


def test_attributes_persist_and_match_zarr_metadata(tmp_path):
    path = tmp_path / "attrs.zarr"
    array = mz.zeros((4, 5), store=path)
    array.attrs["units"] = "kelvin"
    array.attrs["scale"] = 0.25
    assert mz.open_array(path).attrs.asdict() == {"units": "kelvin", "scale": 0.25}
    assert zarr.open_array(path, mode="r", zarr_format=2).attrs["units"] == "kelvin"
    assert json.loads((path / ".zattrs").read_text())["scale"] == 0.25


def test_save_load_and_constructors(tmp_path):
    data = rng.normal(size=(13, 7))
    mz.save(tmp_path / "saved.zarr", data, chunks=(5, 4))
    assert np.array_equal(mz.load(tmp_path / "saved.zarr"), data)
    assert np.all(mz.ones((3, 4))[:] == 1)
    assert np.all(mz.full((3, 4), -2)[:] == -2)
    assert mz.empty((0, 3))[:].shape == (0, 3)


def test_open_modes(tmp_path):
    path = tmp_path / "modes.zarr"
    with pytest.raises(FileNotFoundError):
        mz.open_array(path, mode="r")
    mz.open_array(path, mode="a", shape=(5,), dtype="i4")
    with pytest.raises(FileExistsError):
        mz.open_array(path, mode="x", shape=(5,), dtype="i4")
    replaced = mz.open_array(path, mode="w", shape=(3,), dtype="f4")
    assert replaced.shape == (3,)


def test_metadata_matches_zarr_v2_schema(tmp_path):
    path = tmp_path / "meta.zarr"
    mz.zeros(
        (8, 9),
        store=path,
        chunks=(3, 4),
        dtype="<i4",
        filters=[mz.Delta("<i4")],
        compressor=mz.Zlib(1),
    )
    metadata = json.loads((path / ".zarray").read_text())
    assert metadata == {
        "shape": [8, 9],
        "chunks": [3, 4],
        "dtype": "<i4",
        "fill_value": 0,
        "order": "C",
        "filters": [{"id": "delta", "dtype": "<i4", "astype": "<i4"}],
        "dimension_separator": ".",
        "compressor": {"id": "zlib", "level": 1},
        "zarr_format": 2,
    }


def test_chunk_parallel_threshold_and_roundtrip(monkeypatch):
    real_executor = core.ThreadPoolExecutor
    launches = []

    def recording_executor(*args, **kwargs):
        launches.append(kwargs["max_workers"])
        return real_executor(*args, **kwargs)

    monkeypatch.setattr(core, "ThreadPoolExecutor", recording_executor)
    monkeypatch.setattr(core, "_PARALLEL_CHUNK_COUNT", 2)
    monkeypatch.setattr(core, "_PARALLEL_CHUNK_BYTES", 1)

    small = mz.create_array(
        shape=(4, 4), chunks=(4, 4), dtype="i4", compressor=None
    )
    small[:] = np.arange(16, dtype=np.int32).reshape(4, 4)
    assert np.array_equal(small[:], np.arange(16).reshape(4, 4))
    assert launches == []

    data = np.arange(64, dtype=np.int32).reshape(8, 8)
    large = mz.create_array(
        shape=data.shape, chunks=(4, 4), dtype="i4", compressor=None
    )
    large[:] = data
    assert np.array_equal(large[:], data)
    assert launches == [4, 4]


def test_malformed_chunk_and_unsupported_metadata_fail_loudly():
    store = mz.MemoryStore()
    array = mz.create_array(store, shape=(4,), chunks=(4,), dtype="i4", compressor=None)
    store["0"] = b"short"
    with pytest.raises(ValueError, match="decoded chunk"):
        array[:]

    filtered = mz.create_array(
        shape=(4,),
        chunks=(4,),
        dtype="i4",
        filters=[mz.Delta("i4")],
        compressor=None,
    )
    filtered.store["0"] = np.arange(2, dtype=np.int32).tobytes()
    with pytest.raises(ValueError, match="wrong size"):
        filtered[:]

    metadata = json.loads(store[".zarray"])
    metadata["dimension_separator"] = "/"
    store[".zarray"] = json.dumps(metadata).encode()
    with pytest.raises(NotImplementedError, match="separator"):
        mz.open_array(store)


def test_open_alias_and_broadcast_write():
    store = mz.MemoryStore()
    array = mz.open(store, mode="a", shape=(3, 4), chunks=(2, 2), dtype="i2")
    array[:, :] = np.arange(4, dtype=np.int16)
    assert np.array_equal(array[:], np.tile(np.arange(4, dtype=np.int16), (3, 1)))
