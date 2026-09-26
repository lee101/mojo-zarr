"""Compute kernels for Zarr-compatible codecs.

All buffers are owned by Python. Addresses cross the ABI as Int values and
are rebuilt as non-null pointers only inside functions that use them.
"""

from std.sys import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U16Ptr = UnsafePointer[UInt16, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime SHUFFLE_PARALLEL_THRESHOLD = 8 * 1024 * 1024
comptime SHUFFLE_WORKERS = 8


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


def shuffle_range(
    src: BPtr,
    dst: BPtr,
    count: Int,
    element_size: Int,
    start: Int,
    stop: Int,
):
    comptime W = simd_width_of[DType.float64]()
    comptime BYTE_W = W * 8
    var i = start
    if element_size != 4 and element_size != 8:
        for byte in range(element_size):
            i = start
            while i + BYTE_W <= stop:
                var values = (
                    src + byte + i * element_size
                ).strided_load[width=BYTE_W](element_size)
                dst.store[alignment=1](byte * count + i, values)
                i += BYTE_W
            while i < stop:
                dst[byte * count + i] = src[i * element_size + byte]
                i += 1
        return
    if element_size == 4:
        comptime PLANE_W = BYTE_W // 4
        while i + PLANE_W <= stop:
            var values = src.load[width=BYTE_W, alignment=1](i * 4)
            var even, odd = values.deinterleave()
            var byte0, byte2 = even.deinterleave()
            var byte1, byte3 = odd.deinterleave()
            dst.store[alignment=1](i, byte0)
            dst.store[alignment=1](count + i, byte1)
            dst.store[alignment=1](2 * count + i, byte2)
            dst.store[alignment=1](3 * count + i, byte3)
            i += PLANE_W
    elif element_size == 8:
        comptime PLANE_W = BYTE_W // 8
        while i + PLANE_W <= stop:
            var values = src.load[width=BYTE_W, alignment=1](i * 8)
            var even, odd = values.deinterleave()
            var byte04, byte26 = even.deinterleave()
            var byte15, byte37 = odd.deinterleave()
            var byte0, byte4 = byte04.deinterleave()
            var byte2, byte6 = byte26.deinterleave()
            var byte1, byte5 = byte15.deinterleave()
            var byte3, byte7 = byte37.deinterleave()
            dst.store[alignment=1](i, byte0)
            dst.store[alignment=1](count + i, byte1)
            dst.store[alignment=1](2 * count + i, byte2)
            dst.store[alignment=1](3 * count + i, byte3)
            dst.store[alignment=1](4 * count + i, byte4)
            dst.store[alignment=1](5 * count + i, byte5)
            dst.store[alignment=1](6 * count + i, byte6)
            dst.store[alignment=1](7 * count + i, byte7)
            i += PLANE_W
    while i < stop:
        for byte in range(element_size):
            dst[byte * count + i] = src[i * element_size + byte]
        i += 1


def unshuffle_range(
    src: BPtr,
    dst: BPtr,
    count: Int,
    element_size: Int,
    start: Int,
    stop: Int,
):
    comptime W = simd_width_of[DType.float64]()
    comptime BYTE_W = W * 8
    var i = start
    if element_size != 4 and element_size != 8:
        for byte in range(element_size):
            i = start
            while i + BYTE_W <= stop:
                var values = src.load[width=BYTE_W, alignment=1](
                    byte * count + i
                )
                (dst + byte + i * element_size).strided_store[width=BYTE_W](
                    values, element_size
                )
                i += BYTE_W
            while i < stop:
                dst[i * element_size + byte] = src[byte * count + i]
                i += 1
        return
    if element_size == 4:
        comptime PLANE_W = BYTE_W // 4
        while i + PLANE_W <= stop:
            var byte0 = src.load[width=PLANE_W, alignment=1](i)
            var byte1 = src.load[width=PLANE_W, alignment=1](count + i)
            var byte2 = src.load[width=PLANE_W, alignment=1](2 * count + i)
            var byte3 = src.load[width=PLANE_W, alignment=1](3 * count + i)
            var even = byte0.interleave(byte2)
            var odd = byte1.interleave(byte3)
            dst.store[alignment=1](i * 4, even.interleave(odd))
            i += PLANE_W
    elif element_size == 8:
        comptime PLANE_W = BYTE_W // 8
        while i + PLANE_W <= stop:
            var byte0 = src.load[width=PLANE_W, alignment=1](i)
            var byte1 = src.load[width=PLANE_W, alignment=1](count + i)
            var byte2 = src.load[width=PLANE_W, alignment=1](2 * count + i)
            var byte3 = src.load[width=PLANE_W, alignment=1](3 * count + i)
            var byte4 = src.load[width=PLANE_W, alignment=1](4 * count + i)
            var byte5 = src.load[width=PLANE_W, alignment=1](5 * count + i)
            var byte6 = src.load[width=PLANE_W, alignment=1](6 * count + i)
            var byte7 = src.load[width=PLANE_W, alignment=1](7 * count + i)
            var byte04 = byte0.interleave(byte4)
            var byte26 = byte2.interleave(byte6)
            var byte15 = byte1.interleave(byte5)
            var byte37 = byte3.interleave(byte7)
            var even = byte04.interleave(byte26)
            var odd = byte15.interleave(byte37)
            dst.store[alignment=1](i * 8, even.interleave(odd))
            i += PLANE_W
    while i < stop:
        for byte in range(element_size):
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

    if nbytes >= SHUFFLE_PARALLEL_THRESHOLD and count >= SHUFFLE_WORKERS:
        for worker in range(SHUFFLE_WORKERS):
            var start = count * worker // SHUFFLE_WORKERS
            var stop = count * (worker + 1) // SHUFFLE_WORKERS
            shuffle_range(src, dst, count, element_size, start, stop)
    else:
        shuffle_range(src, dst, count, element_size, 0, count)


@export("mz_unshuffle")
def mz_unshuffle(
    src_addr: Int, dst_addr: Int, nbytes: Int, element_size: Int
) abi("C"):
    if nbytes <= 0 or element_size <= 0 or nbytes % element_size != 0:
        return
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var count = nbytes // element_size

    if nbytes >= SHUFFLE_PARALLEL_THRESHOLD and count >= SHUFFLE_WORKERS:
        for worker in range(SHUFFLE_WORKERS):
            var start = count * worker // SHUFFLE_WORKERS
            var stop = count * (worker + 1) // SHUFFLE_WORKERS
            unshuffle_range(src, dst, count, element_size, start, stop)
    else:
        unshuffle_range(src, dst, count, element_size, 0, count)


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
