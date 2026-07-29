"""Compute kernels for Zarr-compatible codecs.

All buffers are owned by Python. Addresses cross the ABI as Int values and
are rebuilt as non-null pointers only inside functions that use them.
"""

from std.algorithm import parallelize
from std.sys import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U16Ptr = UnsafePointer[UInt16, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def delta_encode(src: BPtr, dst: BPtr, n: Int, kind: Int):
    if n == 0:
        return
    if kind == 1:
        dst[0] = src[0]
        for i in range(1, n):
            dst[i] = src[i] - src[i - 1]
    elif kind == 2:
        var a = src.bitcast[UInt16]()
        var b = dst.bitcast[UInt16]()
        b[0] = a[0]
        for i in range(1, n):
            b[i] = a[i] - a[i - 1]
    elif kind == 4:
        var a = src.bitcast[UInt32]()
        var b = dst.bitcast[UInt32]()
        b[0] = a[0]
        for i in range(1, n):
            b[i] = a[i] - a[i - 1]
    elif kind == 8:
        var a = src.bitcast[UInt64]()
        var b = dst.bitcast[UInt64]()
        b[0] = a[0]
        for i in range(1, n):
            b[i] = a[i] - a[i - 1]
    elif kind == 14:
        var a = src.bitcast[Float32]()
        var b = dst.bitcast[Float32]()
        b[0] = a[0]
        for i in range(1, n):
            b[i] = a[i] - a[i - 1]
    elif kind == 18:
        var a = src.bitcast[Float64]()
        var b = dst.bitcast[Float64]()
        b[0] = a[0]
        for i in range(1, n):
            b[i] = a[i] - a[i - 1]


def delta_decode(src: BPtr, dst: BPtr, n: Int, kind: Int):
    if n == 0:
        return
    if kind == 1:
        dst[0] = src[0]
        for i in range(1, n):
            dst[i] = dst[i - 1] + src[i]
    elif kind == 2:
        var a = src.bitcast[UInt16]()
        var b = dst.bitcast[UInt16]()
        b[0] = a[0]
        for i in range(1, n):
            b[i] = b[i - 1] + a[i]
    elif kind == 4:
        var a = src.bitcast[UInt32]()
        var b = dst.bitcast[UInt32]()
        b[0] = a[0]
        for i in range(1, n):
            b[i] = b[i - 1] + a[i]
    elif kind == 8:
        var a = src.bitcast[UInt64]()
        var b = dst.bitcast[UInt64]()
        b[0] = a[0]
        for i in range(1, n):
            b[i] = b[i - 1] + a[i]
    elif kind == 14:
        var a = src.bitcast[Float32]()
        var b = dst.bitcast[Float32]()
        b[0] = a[0]
        for i in range(1, n):
            b[i] = b[i - 1] + a[i]
    elif kind == 18:
        var a = src.bitcast[Float64]()
        var b = dst.bitcast[Float64]()
        b[0] = a[0]
        for i in range(1, n):
            b[i] = b[i - 1] + a[i]


@export("mz_delta_encode")
def mz_delta_encode(
    src_addr: Int, dst_addr: Int, n: Int, kind: Int
) abi("C"):
    if n <= 0:
        return
    delta_encode(bp(src_addr), bp(dst_addr), n, kind)


@export("mz_delta_decode")
def mz_delta_decode(
    src_addr: Int, dst_addr: Int, n: Int, kind: Int
) abi("C"):
    if n <= 0:
        return
    delta_decode(bp(src_addr), bp(dst_addr), n, kind)


def shuffle_plane(
    src: BPtr, dst: BPtr, count: Int, element_size: Int, byte: Int
):
    comptime W = simd_width_of[DType.uint8]()
    var i = 0
    var vector_end = count - count % W
    while i < vector_end:
        var values = (src + byte + i * element_size).strided_load[width=W](
            element_size
        )
        dst.store(i + byte * count, values)
        i += W
    while i < count:
        dst[byte * count + i] = src[i * element_size + byte]
        i += 1


def unshuffle_plane(
    src: BPtr, dst: BPtr, count: Int, element_size: Int, byte: Int
):
    comptime W = simd_width_of[DType.uint8]()
    var i = 0
    var vector_end = count - count % W
    while i < vector_end:
        var values = src.load[width=W](i + byte * count)
        (dst + byte + i * element_size).strided_store[width=W](
            values, element_size
        )
        i += W
    while i < count:
        dst[i * element_size + byte] = src[byte * count + i]
        i += 1


@export("mz_shuffle")
def mz_shuffle(
    src_addr: Int, dst_addr: Int, nbytes: Int, element_size: Int
) abi("C"):
    if nbytes <= 0 or element_size <= 0 or nbytes % element_size != 0:
        return
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var count = nbytes // element_size

    @parameter
    @__copy_capture(src, dst, count, element_size)
    def work(byte: Int):
        shuffle_plane(src, dst, count, element_size, byte)

    if nbytes >= 8 * 1024 * 1024 and element_size > 1:
        parallelize[work](element_size, element_size)
    else:
        for byte in range(element_size):
            shuffle_plane(src, dst, count, element_size, byte)


@export("mz_unshuffle")
def mz_unshuffle(
    src_addr: Int, dst_addr: Int, nbytes: Int, element_size: Int
) abi("C"):
    if nbytes <= 0 or element_size <= 0 or nbytes % element_size != 0:
        return
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var count = nbytes // element_size

    @parameter
    @__copy_capture(src, dst, count, element_size)
    def work(byte: Int):
        unshuffle_plane(src, dst, count, element_size, byte)

    if nbytes >= 8 * 1024 * 1024 and element_size > 1:
        parallelize[work](element_size, element_size)
    else:
        for byte in range(element_size):
            unshuffle_plane(src, dst, count, element_size, byte)


@export("mz_packbits")
def mz_packbits(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    if n <= 0:
        return
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var groups = n // 8
    var words = src.bitcast[UInt64]()
    comptime W = simd_width_of[DType.uint64]()
    var i = 0
    var vector_end = groups - groups % W
    while i < vector_end:
        var values = words.load[width=W, alignment=1](i)
        var packed = (
            (values & UInt64(0x0101010101010101))
            * UInt64(0x8040201008040201)
        ) >> UInt64(56)
        dst.store(i, packed.cast[DType.uint8]())
        i += W
    while i < groups:
        var word = words.load[alignment=1](i)
        var value = (
            (word & UInt64(0x0101010101010101))
            * UInt64(0x8040201008040201)
        ) >> UInt64(56)
        dst[i] = UInt8(value)
        i += 1
    if n % 8:
        var value = UInt8(0)
        for bit in range(n % 8):
            var j = groups * 8 + bit
            if src[j] != 0:
                value |= UInt8(1 << (7 - bit))
        dst[groups] = value


@export("mz_unpackbits")
def mz_unpackbits(src_addr: Int, dst_addr: Int, n: Int) abi("C"):
    if n <= 0:
        return
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    for i in range(n):
        dst[i] = UInt8((src[i // 8] >> UInt8(7 - i % 8)) & 1)
