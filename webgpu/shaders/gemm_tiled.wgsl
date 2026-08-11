// WGSL Compute Shader 2: Workgroup Shared Memory Tiled GEMM
// Uses workgroup memory (SRAM) and workgroupBarrier() for tile caching

struct MatrixDimensions {
    M : u32,
    N : u32,
    K : u32,
};

@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform> dims : MatrixDimensions;

// Workgroup Shared Memory Allocation (Equivalent to __shared__ in CUDA)
const TILE_SIZE : u32 = 16u;
var<workgroup> As : array<array<f32, 16>, 16>;
var<workgroup> Bs : array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) workgroup_id : vec3<u32>,
    @builtin(local_invocation_id) local_id : vec3<u32>,
    @builtin(global_invocation_id) global_id : vec3<u32>
) {
    let bx = workgroup_id.x;
    let by = workgroup_id.y;
    let tx = local_id.x;
    let ty = local_id.y;

    let row = by * TILE_SIZE + ty;
    let col = bx * TILE_SIZE + tx;

    var acc : f32 = 0.0;
    let numTiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t : u32 = 0u; t < numTiles; t = t + 1u) {
        // Collaborative load into Workgroup Shared Memory
        if (row < dims.M && (t * TILE_SIZE + tx) < dims.K) {
            As[ty][tx] = A[row * dims.K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }

        if ((t * TILE_SIZE + ty) < dims.K && col < dims.N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * dims.N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }

        // Barrier synchronization (Equivalent to __syncthreads() in CUDA)
        workgroupBarrier();

        // Accumulate tile product in workgroup memory
        for (var k : u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            acc = acc + As[ty][k] * Bs[k][tx];
        }

        // Barrier synchronization before loading next tile
        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = acc;
    }
}
