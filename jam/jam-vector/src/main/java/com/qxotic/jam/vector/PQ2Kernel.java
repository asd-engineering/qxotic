package com.qxotic.jam.vector;

import static com.qxotic.jam.vector.VectorSupport.readFloat16;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

import java.lang.foreign.MemorySegment;

/**
 * PQ2_0 gemm. PQ2_0 block: 128 elements / 34 bytes ({@code f16 d; 32 2-bit bytes}). Element value
 * is {@code (((b >>> (2*i)) & 3) - 1) * d}. A {@link BandGemm#MR}-row band is dequantized into an
 * F32 scratch, then {@link BandGemm}'s F32 band sweeps the columns.
 */
final class PQ2Kernel {

    private PQ2Kernel() {}

    static final int QK = 128, BYTES = 34;

    public static void gemm(
            MemorySegment w,
            MemorySegment a,
            long aBase,
            MemorySegment o,
            long oBase,
            int aStride,
            int oStride,
            int n,
            int m,
            int k,
            long wOff,
            Scratch scratch) {
        BandGemm.gemm(
                w,
                a,
                aBase,
                o,
                oBase,
                aStride,
                oStride,
                n,
                m,
                k,
                wOff,
                scratch,
                PQ2Kernel::dequantizeRow);
    }

    /**
     * Dequantize one PQ2_0 weight row (dim1 % 128 == 0) into {@code dst[dstOffset..]} in element
     * order.
     */
    private static void dequantizeRow(
            MemorySegment w, long rowElemOffset, int dim1, MemorySegment dst, long dstBase) {
        int kblocks = dim1 / QK;
        long firstBlock = rowElemOffset / QK;
        for (int blk = 0; blk < kblocks; blk++) {
            long bo = (firstBlock + blk) * BYTES;
            float d = readFloat16(w, bo);
            long base = dstBase + (long) blk * QK * 4;
            for (int e = 0; e < 32; e++) {
                int b = Byte.toUnsignedInt(w.get(JAVA_BYTE, bo + 2 + e));
                long dstOff = base + (long) e * 16;
                dst.set(JAVA_FLOAT_UNALIGNED, dstOff, ((b & 3) - 1) * d);
                dst.set(JAVA_FLOAT_UNALIGNED, dstOff + 4, (((b >>> 2) & 3) - 1) * d);
                dst.set(JAVA_FLOAT_UNALIGNED, dstOff + 8, (((b >>> 4) & 3) - 1) * d);
                dst.set(JAVA_FLOAT_UNALIGNED, dstOff + 12, (((b >>> 6) & 3) - 1) * d);
            }
        }
    }
}
