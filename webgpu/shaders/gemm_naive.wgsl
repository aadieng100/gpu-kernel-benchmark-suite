// WGSL Compute Shader 1: Naive GEMM
// Each thread computes one element C[row, col] by fetching directly from Storage Buffers

struct MatrixDimensions {
    M : u32,
    N : u32,
    K : u32,
};

@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform> dims : MatrixDimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;

    if (row < dims.M && col < dims.N) {
        var sum : f32 = 0.0;
        for (var k : u32 = 0u; k < dims.K; k = k + 1u) {
            sum = sum + A[row * dims.K + k] * B[k * dims.N + col];
        }
        C[row * dims.N + col] = sum;
    }
}
