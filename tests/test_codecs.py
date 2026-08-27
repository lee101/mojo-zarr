import zlib

import numcodecs
import numpy as np
import pytest

import mojo_zarr as mz


rng = np.random.default_rng(20260729)


@pytest.mark.parametrize("dtype", ["u1", "<i2", "<u4", "<i8", "<f4", "<f8"])
def test_delta_matches_numcodecs(dtype):
    if np.dtype(dtype).kind == "f":
        source = np.cumsum(rng.normal(size=10_003)).astype(dtype)
    else:
        source = rng.integers(0, 1000, size=10_003).astype(dtype)
    ours = mz.Delta(dtype)
    reference = numcodecs.Delta(dtype)
    encoded = ours.encode(source)
    assert np.array_equal(encoded, reference.encode(source), equal_nan=True)
    decoded = ours.decode(encoded)
    expected = reference.decode(encoded)
    if np.dtype(dtype).kind == "f":
        assert np.allclose(decoded, expected, rtol=1e-6, atol=1e-6)
    else:
        assert np.array_equal(decoded, expected)
    assert ours.get_config() == reference.get_config()


def test_delta_astype_fallback_matches_numcodecs():
    source = np.arange(10_000, dtype=np.int32)
    ours = mz.Delta("<i4", astype="<i2")
    reference = numcodecs.Delta("<i4", astype="<i2")
    assert np.array_equal(ours.encode(source), reference.encode(source))
    assert np.array_equal(ours.decode(ours.encode(source)), source)


@pytest.mark.parametrize("elementsize", [1, 2, 3, 4, 8, 16])
def test_shuffle_is_byte_identical_to_numcodecs(elementsize):
    source = rng.integers(0, 256, 12_000, dtype=np.uint8)
    ours = mz.Shuffle(elementsize)
    reference = numcodecs.Shuffle(elementsize)
    encoded = ours.encode(source)
    assert np.array_equal(encoded, reference.encode(source))
    assert np.array_equal(ours.decode(encoded), source)


@pytest.mark.parametrize("count", [(1 << 20) - 1, (1 << 20) + 3])
def test_shuffle_parallel_threshold_and_simd_tail_matches_numcodecs(count):
    elementsize = 8
    source = rng.integers(0, 256, count * elementsize, dtype=np.uint8)
    ours = mz.Shuffle(elementsize)
    encoded = ours.encode(source)
    assert np.array_equal(encoded, numcodecs.Shuffle(elementsize).encode(source))
    assert np.array_equal(ours.decode(encoded), source)


@pytest.mark.parametrize("elementsize,count", [(4, 4099), (8, 4099)])
def test_shuffle_simd_scalar_tail_matches_numcodecs(elementsize, count):
    source = rng.integers(0, 256, count * elementsize, dtype=np.uint8)
    ours = mz.Shuffle(elementsize)
    encoded = ours.encode(source)
    assert np.array_equal(encoded, numcodecs.Shuffle(elementsize).encode(source))
    assert np.array_equal(ours.decode(encoded), source)


@pytest.mark.parametrize("size", [0, 1, 7, 8, 9, 1023, 10_007])
def test_packbits_is_byte_identical_to_numcodecs(size):
    source = rng.integers(0, 2, size=size, dtype=np.uint8).astype(bool)
    ours = mz.PackBits()
    reference = numcodecs.PackBits()
    encoded = ours.encode(source)
    assert np.array_equal(encoded, reference.encode(source))
    assert np.array_equal(ours.decode(encoded), source)


def test_packbits_simd_and_partial_byte_tail():
    backing = rng.integers(0, 2, size=8 * 257 + 4, dtype=np.uint8).astype(bool)
    source = backing[1:]
    encoded = mz.PackBits().encode(source)
    assert np.array_equal(encoded, numcodecs.PackBits().encode(source))
    assert np.array_equal(mz.PackBits().decode(encoded), source)


@pytest.mark.parametrize("location", ["start", "end"])
def test_crc32_matches_numcodecs(location):
    source = rng.integers(0, 256, size=100_003, dtype=np.uint8)
    ours = mz.CRC32(location)
    reference = numcodecs.CRC32(location)
    encoded = ours.encode(source)
    assert np.array_equal(encoded, reference.encode(source))
    assert ours.checksum(source, 12345) == zlib.crc32(source, 12345)
    assert np.array_equal(ours.decode(encoded), source)


def test_crc32_detects_corruption():
    encoded = mz.CRC32().encode(np.arange(100, dtype=np.uint8))
    encoded[20] ^= 1
    with pytest.raises(RuntimeError, match="checksum do not match"):
        mz.CRC32().decode(encoded)


def test_zlib_matches_numcodecs_and_roundtrips():
    source = np.repeat(np.arange(1000, dtype=np.int32), 20)
    ours = mz.Zlib(level=6)
    reference = numcodecs.Zlib(level=6)
    assert ours.encode(source) == reference.encode(source)
    assert ours.decode(ours.encode(source)) == source.tobytes()


def test_codec_out_buffers():
    source = np.arange(1000, dtype=np.int32)
    shuffled = mz.Shuffle(4).encode(source)
    shuffle_out = np.empty(source.nbytes, dtype=np.uint8)
    assert mz.Shuffle(4).decode(shuffled, out=shuffle_out) is shuffle_out
    assert shuffle_out.tobytes() == source.tobytes()
    delta_out = np.empty_like(source)
    assert mz.Delta("i4").decode(mz.Delta("i4").encode(source), delta_out) is delta_out
    assert np.array_equal(delta_out, source)


@pytest.mark.parametrize("codec", [mz.Shuffle(4), mz.Delta("i4")])
def test_native_codecs_reject_noncontiguous_output(codec):
    source = np.arange(32, dtype=np.int32)
    encoded = codec.encode(source)
    out = np.empty((32, 2), dtype=np.int32)[:, 0]
    with pytest.raises(ValueError, match="C-contiguous"):
        codec.decode(encoded, out=out)


def test_shuffle_handles_overlapping_input_and_output():
    source = np.arange(1000, dtype=np.int32)
    expected = numcodecs.Shuffle(4).encode(source)
    inplace = source.view(np.uint8).copy()
    assert mz.Shuffle(4).encode(inplace, out=inplace) is inplace
    assert np.array_equal(inplace, expected)


def test_codec_input_validation():
    with pytest.raises(ValueError, match="multiple"):
        mz.Delta("i4").decode(b"abc")
    with pytest.raises(ValueError, match="multiple"):
        mz.Shuffle(4).encode(b"abc")
    with pytest.raises(TypeError, match="boolean"):
        mz.PackBits().encode(np.arange(8, dtype=np.uint8))
    with pytest.raises(ValueError, match="wrong size"):
        mz.Shuffle(4).decode(np.arange(16, dtype=np.uint8), out=np.empty(32, "u1"))
