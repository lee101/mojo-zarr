# mojo-zarr

`mojo-zarr` is a standalone, focused port of the compute-heavy parts of
[Zarr](https://zarr.dev/) to Mojo. It provides a Python API for chunked
N-dimensional arrays, writes standard Zarr v2 stores, and moves Delta,
Shuffle, and PackBits codec loops into one compiled Mojo shared library.
CRC32 uses Python's zero-copy native zlib binding.

This is useful today for local or in-memory numeric arrays that need portable
Zarr v2 storage. Stores written here can be opened by upstream Zarr, and this
package can open equivalent stores written by upstream Zarr and numcodecs.
The Python package is named `mojo_zarr`, so both implementations can be
imported in the same process for comparison.

## Covered subset

- C-order boolean, integer, and floating-point arrays with any dimensionality
- Local-directory and mutable-mapping memory stores
- Zarr v2 `.zarray`/`.zattrs` metadata, `.` chunk keys, edge-chunk padding,
  fill values, and sparse deletion of all-fill chunks
- Integer indexing, ellipsis, unit/stepped/reversed slices, broadcast writes,
  scalar arrays, and partial-chunk read/modify/write
- `Array`, `create_array`, `open_array`, `open`, `array`, `zeros`, `ones`,
  `full`, `empty`, `save`, and `load`
- `Delta(dtype, astype=None)`, `Shuffle(elementsize=4)`, `PackBits()`,
  `CRC32(location=None)`, and `Zlib(level=1)`, with numcodecs-compatible
  configuration and byte streams
- Atomic replacement of local metadata and chunks

Zlib and CRC32 use Python's standard-library zlib binding; Delta, Shuffle,
and PackBits call Mojo. `compressor="auto"` selects Zlib level 1 rather than
upstream Zarr's usual Blosc default.

Not covered are Zarr v3, groups, sharding, Blosc/Zstd, remote or asynchronous
stores, object/structured/complex dtypes, Fortran-order arrays, fancy or mask
indexing, `newaxis`, resizing/appending, consolidated metadata, and
multi-process write coordination. Unsupported format and indexing features
raise explicit errors.

## Install and build

The repository pins the tested Mojo nightly and all development dependencies:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-zarr.so`. Run the benchmark only through its
locked Pixi task:

```bash
pixi run bench
```

## Usage

This example creates a compressed, upstream-readable array and exercises a
partial read:

```bash
pixi run python - <<'PY'
import numpy as np
import mojo_zarr as zarr

data = np.arange(240, dtype=np.int32).reshape(20, 12)
a = zarr.create_array(
    "example.zarr",
    shape=data.shape,
    chunks=(8, 5),
    dtype=data.dtype,
    filters=[zarr.Delta("<i4"), zarr.Shuffle(4)],
    compressor=zarr.Zlib(level=1),
    overwrite=True,
)
a[:] = data
print(a[3:8, ::2])
PY
```

For an in-memory array, omit `store` or pass `MemoryStore()`:

```python
a = zarr.array(np.arange(100), chunks=(16,))
assert np.array_equal(a[::-1], np.arange(100)[::-1])
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, Python 3.13.14, Zarr 3.2.1, NumPy 2.5.1, and numcodecs 0.16.5.
Times are the best of three runs. The ratio is upstream time divided by
Mojo time, so values above 1 mean Mojo is faster.

| case | mojo-zarr | upstream | upstream / Mojo |
|---|---:|---:|---:|
| Delta.encode, 12M int32 | 279.78 ms | 403.58 ms | 1.44x (faster) |
| Shuffle.encode, 64 MB / 8-byte | 443.78 ms | 487.74 ms | 1.10x (faster) |
| PackBits.encode, 40M bool | 5.52 ms | 5.68 ms | 1.03x (faster) |
| CRC32, 16 MB | 13.12 ms | 12.92 ms | 0.98x (slower) |
| Chunk write, 16 MB + codecs | 62.59 ms | 203.63 ms | 3.25x (faster) |
| Chunk read, 16 MB + codecs | 104.28 ms | 104.53 ms | 1.00x (faster) |

Shuffle uses contiguous native-width SIMD transposes for 4- and 8-byte
elements, strided SIMD for other widths, scalar tails, and eight bounded
element ranges at or above 8 MiB. PackBits handles eight boolean bytes per
SIMD lane and uses scalar tails for unaligned remainders. Large independent
chunk sets use bounded CPU threads; full-chunk writes skip read/modify/decode,
and decode pipelines reuse their final NumPy destination.

No GPU path is provided. Shuffle and PackBits are byte-bandwidth-bound, CRC32
has a serial recurrence handled by native zlib, and chunk I/O is dominated by
compression and storage. None offers the arithmetic intensity needed to repay
device transfer and launch overhead.

## How it works

`src/zarr.mojo` is one compilation unit. Python allocates all input and output
buffers and passes their addresses as 64-bit integers through `ctypes`; each
export reconstructs an `UnsafePointer[..., AnyOrigin[mut=True]]`. No Mojo
allocation crosses the ABI, so ownership and lifetime stay with NumPy.

Arrays are C-contiguous within each chunk. The Python layer maps a selection
to intersecting chunk coordinates, decodes full chunks, copies only the
intersection, and reverses the codec pipeline on write. Edge chunks retain
their full configured shape and are padded with the array fill value, exactly
as Zarr v2 requires. Codec configuration is serialized with numcodecs IDs, so
upstream Zarr can decode the resulting chunk bytes without this package.
