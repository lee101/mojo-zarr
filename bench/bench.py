"""Benchmarks Mojo codec kernels and chunk I/O against upstream Zarr/numcodecs."""

from __future__ import annotations

import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numcodecs
import numpy as np
import zarr

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_zarr as mz  # noqa: E402


def timeit(function, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    rng = np.random.default_rng(29)
    cases: list[tuple[str, callable, callable]] = []

    delta_data = np.cumsum(
        rng.integers(-4, 5, size=12_000_000, dtype=np.int32), dtype=np.int32
    )
    cases.append(
        (
            "Delta.encode, 12M int32",
            lambda: mz.Delta("<i4").encode(delta_data),
            lambda: numcodecs.Delta("<i4").encode(delta_data),
        )
    )

    shuffle_data = rng.integers(0, 2**63, size=8_000_000, dtype=np.int64)
    cases.append(
        (
            "Shuffle.encode, 64 MB / 8-byte",
            lambda: mz.Shuffle(8).encode(shuffle_data),
            lambda: numcodecs.Shuffle(8).encode(shuffle_data),
        )
    )

    bool_data = rng.integers(0, 2, size=40_000_000, dtype=np.uint8).astype(bool)
    cases.append(
        (
            "PackBits.encode, 40M bool",
            lambda: mz.PackBits().encode(bool_data),
            lambda: numcodecs.PackBits().encode(bool_data),
        )
    )

    checksum_data = rng.integers(0, 256, size=16 << 20, dtype=np.uint8)
    cases.append(
        (
            "CRC32, 16 MB",
            lambda: mz.CRC32.checksum(checksum_data),
            lambda: numcodecs.CRC32.checksum(checksum_data),
        )
    )

    with tempfile.TemporaryDirectory() as temporary:
        data = rng.integers(-100, 101, size=(2048, 2048), dtype=np.int32)
        ours = mz.create_array(
            Path(temporary) / "ours.zarr",
            shape=data.shape,
            chunks=(256, 256),
            dtype="<i4",
            filters=[mz.Delta("<i4"), mz.Shuffle(4)],
            compressor=mz.Zlib(1),
        )
        reference = zarr.create_array(
            Path(temporary) / "upstream.zarr",
            shape=data.shape,
            chunks=(256, 256),
            dtype="<i4",
            zarr_format=2,
            filters=[numcodecs.Delta("<i4"), numcodecs.Shuffle(4)],
            compressors=[numcodecs.Zlib(1)],
        )
        ours[:] = data
        reference[:] = data
        assert np.array_equal(ours[:], reference[:])
        cases.extend(
            [
                (
                    "Chunk write, 16 MB + codecs",
                    lambda: ours.__setitem__(slice(None), data),
                    lambda: reference.__setitem__(slice(None), data),
                ),
                (
                    "Chunk read, 16 MB + codecs",
                    lambda: ours[:],
                    lambda: reference[:],
                ),
            ]
        )

        rows = []
        for name, mojo_function, upstream_function in cases:
            mojo_function()
            upstream_function()
            mojo_seconds = timeit(mojo_function)
            upstream_seconds = timeit(upstream_function)
            rows.append(
                (
                    name,
                    mojo_seconds * 1000,
                    upstream_seconds * 1000,
                    upstream_seconds / mojo_seconds,
                )
            )

    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print(
        f"Python {platform.python_version()}, Zarr {zarr.__version__}, "
        f"NumPy {np.__version__}, numcodecs {numcodecs.__version__}"
    )
    print()
    print("| case | mojo-zarr | upstream | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, mojo_ms, upstream_ms, ratio in rows:
        label = "faster" if ratio > 1 else "slower"
        print(
            f"| {name} | {mojo_ms:.2f} ms | {upstream_ms:.2f} ms | "
            f"{ratio:.2f}x ({label}) |"
        )


if __name__ == "__main__":
    main()
