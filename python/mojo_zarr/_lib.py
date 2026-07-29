from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_ZARR_LIB") or os.path.join(
    ROOT, "dist", "libmojo-zarr.so"
)
SRC = os.path.join(ROOT, "src", "zarr.mojo")

I = ctypes.c_int64
_SIGNATURES = {
    "mz_delta_encode": ([I, I, I, I], None),
    "mz_delta_decode": ([I, I, I, I], None),
    "mz_shuffle": ([I, I, I, I], None),
    "mz_unshuffle": ([I, I, I, I], None),
    "mz_packbits": ([I, I, I], None),
    "mz_unpackbits": ([I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_ZARR_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_ZARR_LIB does not exist: {LIB}")
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo was not found; run `pixi run build` first")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def address(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if array.size and not array.ctypes.data:
        raise ValueError("FFI buffer has a null data pointer")
    return int(array.ctypes.data)


def ffi_length(value: int, name: str = "length") -> int:
    value = int(value)
    if not 0 <= value <= np.iinfo(np.int64).max:
        raise OverflowError(f"{name} does not fit a non-negative C int64")
    return value
