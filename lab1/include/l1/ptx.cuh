// Thin wrappers around the PTX instructions used in Part C2 (sm_80+).
// Provided: you call them, you do not need to edit them. Reference:
// PTX ISA, sections "cp.async", "ldmatrix" and "mma.m16n8k16".
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace l1::ptx {

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- asynchronous global -> shared copy (Ampere+) -------------------------
// Copies 16 bytes (8 halves) without going through registers. Both addresses
// must be 16-byte aligned. Copies are grouped: commit() closes the current
// group; wait<N>() blocks until at most N groups are still in flight.
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(smem)), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ---- ldmatrix: shared -> registers in Tensor Core fragment layout ---------
// Loads four 8x8 matrices of 16-bit elements. Lane i supplies the address of
// one ROW (8 contiguous halves = 16 bytes):
//   lanes 0-7  -> rows of matrix 0 -> r[0]
//   lanes 8-15 -> rows of matrix 1 -> r[1]
//   lanes 16-23-> rows of matrix 2 -> r[2]
//   lanes 24-31-> rows of matrix 3 -> r[3]
// Afterwards lane i holds, in r[m], elements (row i/4, cols 2*(i%4), 2*(i%4)+1)
// of matrix m. The .trans variant transposes each 8x8 matrix on the way.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(smem)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(smem)));
}

// ---- mma.sync m16n8k16: D(16x8,f32) += A(16x16,f16,row) * B(16x8,f16,col) --
// Fragment layout, with g = lane / 4 and t = lane % 4:
//   a[0] = A[g   ][2t..2t+1]   a[1] = A[g+8][2t..2t+1]
//   a[2] = A[g   ][2t+8..+9]   a[3] = A[g+8][2t+8..+9]
//   b[0] = B[2t..2t+1  ][g]    b[1] = B[2t+8..2t+9][g]
//   d[0..1] = D[g][2t..2t+1]   d[2..3] = D[g+8][2t..2t+1]
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4],
                                          const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

} // namespace l1::ptx
