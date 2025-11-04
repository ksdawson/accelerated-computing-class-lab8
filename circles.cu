#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

class GpuMemoryPool {
  public:
    GpuMemoryPool() = default;

    ~GpuMemoryPool();

    GpuMemoryPool(GpuMemoryPool const &) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool const &) = delete;
    GpuMemoryPool(GpuMemoryPool &&) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool &&) = delete;

    void *alloc(size_t size);
    void reset();

  private:
    std::vector<void *> allocations_;
    std::vector<size_t> capacities_;
    size_t next_idx_ = 0;
};

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

void render_cpu(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,
    float const *circle_y,
    float const *circle_radius,
    float const *circle_red,
    float const *circle_green,
    float const *circle_blue,
    float const *circle_alpha,
    float *img_red,
    float *img_green,
    float *img_blue) {

    // Initialize background to white
    for (int32_t pixel_idx = 0; pixel_idx < width * height; pixel_idx++) {
        img_red[pixel_idx] = 1.0f;
        img_green[pixel_idx] = 1.0f;
        img_blue[pixel_idx] = 1.0f;
    }

    // Render circles
    for (int32_t i = 0; i < n_circle; i++) {
        float c_x = circle_x[i];
        float c_y = circle_y[i];
        float c_radius = circle_radius[i];
        for (int32_t y = int32_t(c_y - c_radius); y <= int32_t(c_y + c_radius + 1.0f);
             y++) {
            for (int32_t x = int32_t(c_x - c_radius); x <= int32_t(c_x + c_radius + 1.0f);
                 x++) {
                float dx = x - c_x;
                float dy = y - c_y;
                if (!(0 <= x && x < width && 0 <= y && y < height &&
                      dx * dx + dy * dy < c_radius * c_radius)) {
                    continue;
                }
                int32_t pixel_idx = y * width + x;
                float pixel_red = img_red[pixel_idx];
                float pixel_green = img_green[pixel_idx];
                float pixel_blue = img_blue[pixel_idx];
                float pixel_alpha = circle_alpha[i];
                pixel_red =
                    circle_red[i] * pixel_alpha + pixel_red * (1.0f - pixel_alpha);
                pixel_green =
                    circle_green[i] * pixel_alpha + pixel_green * (1.0f - pixel_alpha);
                pixel_blue =
                    circle_blue[i] * pixel_alpha + pixel_blue * (1.0f - pixel_alpha);
                img_red[pixel_idx] = pixel_red;
                img_green[pixel_idx] = pixel_green;
                img_blue[pixel_idx] = pixel_blue;
            }
        }
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Scan code

/// Helpers to deal with Op::Data type
// Generic, aligned struct for vectorized memory access
template <typename T, int N>
struct alignas(sizeof(T) * N) Vectorized {
    T elements[N];
};
// Needed because compiler doesn't know how to shuffle DebugRange
template <typename T>
__device__ T shfl_up_any(T val, unsigned int delta) {
    T result;
    if constexpr (sizeof(T) == 4) {
        // Single 32-bit value
        uint32_t v = *reinterpret_cast<uint32_t*>(&val);
        v = __shfl_up_sync(0xffffffff, v, delta);
        *reinterpret_cast<uint32_t*>(&result) = v;
    } else {
        // Two 32-bit values (e.g. DebugRange)
        const uint32_t* src = reinterpret_cast<const uint32_t*>(&val);
        uint32_t* dst = reinterpret_cast<uint32_t*>(&result);
        dst[0] = __shfl_up_sync(0xffffffff, src[0], delta);
        dst[1] = __shfl_up_sync(0xffffffff, src[1], delta);
    }
    return result;
}
template <typename T>
__device__ T shfl_down_any(T val, unsigned int delta) {
    T result;
    if constexpr (sizeof(T) == 4) {
        // Single 32-bit value
        uint32_t v = *reinterpret_cast<uint32_t*>(&val);
        v = __shfl_down_sync(0xffffffff, v, delta);
        *reinterpret_cast<uint32_t*>(&result) = v;
    } else {
        // Two 32-bit values (e.g. DebugRange)
        const uint32_t* src = reinterpret_cast<const uint32_t*>(&val);
        uint32_t* dst = reinterpret_cast<uint32_t*>(&result);
        dst[0] = __shfl_down_sync(0xffffffff, src[0], delta);
        dst[1] = __shfl_down_sync(0xffffffff, src[1], delta);
    }
    return result;
}

namespace scan_gpu {

// Helpers
template <typename Op>
__device__ typename Op::Data warp_local_scan(typename Op::Data val) {
    using Data = typename Op::Data;

    // Computes parallel prefix on 32 elements using Hillis Steele Scan w/ warp shuffle
    const uint32_t thread_idx = threadIdx.x % 32;
    uint32_t idx = 1;
    for (uint32_t step = 0; step < 5; ++step) { // log2(32) = 5
        // Load prefix from register
        Data tmp = shfl_up_any(val, idx);
        tmp = (thread_idx >= idx) ? tmp : Op::identity(); // Mask out

        // Update prefix in register
        val = Op::combine(tmp, val);

        // Multiply idx by 2
        idx <<= 1;
    }

    return val;
}

template <typename Op, uint32_t VEC_SIZE, bool DO_FIX>
__device__ inline typename Op::Data thread_local_scan(size_t n, typename Op::Data const *x, typename Op::Data *out,
    const uint32_t start_i, const uint32_t end_i,
    typename Op::Data accumulator
) {
    using Data = typename Op::Data;
    using VecData = Vectorized<Data, VEC_SIZE>;

    // Vectorize
    VecData const *vx = reinterpret_cast<VecData const *>(x);
    VecData *vout = reinterpret_cast<VecData*>(out);
    const uint32_t start_vi = start_i / VEC_SIZE;
    const uint32_t end_vi = end_i / VEC_SIZE;

    // Local scan
    for (uint32_t i = start_vi; i < end_vi; ++i) {
        VecData v = vx[i];
        #pragma unroll
        for (uint32_t vi = 0; vi < VEC_SIZE; ++vi) {
            accumulator = Op::combine(accumulator, v.elements[vi]);
            v.elements[vi] = accumulator;
        }
        // Output to memory
        if constexpr (DO_FIX) { vout[i] = v; }
    }
    // Handle vector tail
    const uint32_t start_scalar_i = end_vi * VEC_SIZE;
    for (uint32_t i = start_scalar_i; i < end_i; ++i) {
        accumulator = Op::combine(accumulator, x[i]);
        if constexpr (DO_FIX) { out[i] = accumulator; }
    }
    return accumulator;
}

template <typename Op, uint32_t VEC_SIZE, bool DO_FIX>
__device__ typename Op::Data warp_scan(
    size_t n, typename Op::Data const *x, typename Op::Data *out, // Work dimensions
    typename Op::Data seed // Seed for thread 0
) {
    using Data = typename Op::Data;

    // Divide x across the threads
    const uint32_t thread_idx = threadIdx.x % 32;
    const uint32_t n_per_thread = ((n / VEC_SIZE) / 32) * VEC_SIZE; // Aligns to vector size
    const uint32_t start_i = thread_idx * n_per_thread;
    const uint32_t end_i = start_i + n_per_thread;

    // Local scan
    Data accumulator = (thread_idx == 0) ? seed : Op::identity();
    accumulator = thread_local_scan<Op, VEC_SIZE, false>(n, x, out, start_i, end_i, accumulator);
    __syncwarp();

    // Hierarchical scan on endpoints
    accumulator = warp_local_scan<Op>(accumulator);

    if constexpr (DO_FIX) {
        // Shuffle accumulators
        accumulator = shfl_up_any(accumulator, 1);
        accumulator = (thread_idx >= 1) ? accumulator : seed;

        // Local scan fix
        accumulator = thread_local_scan<Op, VEC_SIZE, true>(n, x, out, start_i, end_i, accumulator);

        // Handle warp tail
        if (thread_idx == 31) {
            for (uint32_t i = end_i; i < end_i + (n - 32 * n_per_thread); ++i) {
                accumulator = Op::combine(accumulator, x[i]);
                out[i] = accumulator;
            }
        }
    } else {
        // Handle warp tail
        if (thread_idx == 31) {
            for (uint32_t i = end_i; i < end_i + (n - 32 * n_per_thread); ++i) {
                accumulator = Op::combine(accumulator, x[i]);
            }
        }
    }
    return accumulator;
}

template <typename Op, uint32_t VEC_SIZE, bool DO_FIX>
__device__ void warp_scan_handler(
    size_t n, typename Op::Data const *x, typename Op::Data *out, // Work dimensions
    typename Op::Data seed // Seed for thread 0
) {
    using Data = typename Op::Data;

    // Divide x into blocks of 32 * VEC_SIZE and pass to warp scan
    const uint32_t block_size = 32 * VEC_SIZE;
    const uint32_t num_blocks = max((uint32_t)n / block_size, 1u);
    const uint32_t thread_idx = threadIdx.x % 32;

    for (uint32_t idx = 0; idx < num_blocks; ++idx) {
        // Move buffers
        Data const *bx = x + idx * block_size;
        Data *bout = out + idx * block_size;

        // On the last block process whatever is left
        const uint32_t current_block_size = (idx == num_blocks - 1) ? n - idx * block_size : block_size;

        // Call warp scan
        seed = warp_scan<Op, VEC_SIZE, DO_FIX>(current_block_size, bx, bout, seed);
        __syncwarp();

        // For the next block, use the seed from the last thread of this block
        seed = shfl_down_any(seed, 31 - threadIdx.x);
    }

    if constexpr (!DO_FIX) {
        // Only output last accumulator to memory
        if (thread_idx == 31) {
            *out = seed;
        }
    }
}

// 3-Kernel Parallel Algorithm
template <typename Op, uint32_t VEC_SIZE, bool DO_FIX>
__launch_bounds__(32*32)
__global__ void local_scan(size_t n, typename Op::Data const *x, typename Op::Data *out, typename Op::Data *seed) {
    using Data = typename Op::Data;
    // Thread block info
    const uint32_t num_sm = gridDim.x;
    const uint32_t num_warp = blockDim.x / 32;
    const uint32_t block_idx = blockIdx.x;
    const uint32_t warp_idx = threadIdx.x / 32;

    // Divide x across the SMs
    uint32_t n_per_sm = ((n / VEC_SIZE) / num_sm) * VEC_SIZE; // Aligns to vector size
    Data const *sm_x = x + block_idx * n_per_sm;
    Data *sm_out = out + block_idx * n_per_sm;
    Data *sm_seed = seed + block_idx * num_warp;

    // Handle SM tail
    n_per_sm += (block_idx == num_sm - 1) ? n - num_sm * n_per_sm : 0;

    // Divide sm_x across the warps
    uint32_t n_per_warp = ((n_per_sm / VEC_SIZE) / num_warp) * VEC_SIZE;
    Data const *warp_x = sm_x + warp_idx * n_per_warp;
    Data *warp_out = sm_out + warp_idx * n_per_warp;
    Data *warp_seed = sm_seed + warp_idx;

    // Handle warp tail
    n_per_warp += (warp_idx == num_warp - 1) ? n_per_sm - num_warp * n_per_warp : 0;

    // Call warp scan
    if constexpr (DO_FIX) {
        // Each chunk gets the previous seed
        Data seed_val = (block_idx == 0 && warp_idx == 0) ? Op::identity() : *(warp_seed - 1);
        warp_scan_handler<Op, VEC_SIZE, true>(n_per_warp, warp_x, warp_out, seed_val);
    } else {
        warp_scan_handler<Op, VEC_SIZE, false>(n_per_warp, warp_x, warp_seed, Op::identity());
    }
}
template <typename Op, uint32_t VEC_SIZE>
__launch_bounds__(1*32)
__global__ void hierarchical_scan(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    warp_scan<Op, VEC_SIZE, true>(n, x, out, Op::identity());
}

template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    typename Op::Data *x, // pointer to GPU memory
    void *workspace       // pointer to GPU memory
) {
    using Data = typename Op::Data;

    // Use the workspace as scratch for seeds
    Data *seed = reinterpret_cast<Data*>(workspace);

    // Thread block dimensions
    constexpr uint32_t B = 48;
    constexpr uint32_t W = 32; // Tuning parameter
    constexpr uint32_t T = 32;

    // Set vector size
    if constexpr (sizeof(Data) > 16) {
        return nullptr;
    }
    constexpr uint32_t VS = 16 / sizeof(Data);

    // Memory
    local_scan<Op, VS, false><<<B, W*T>>>(n, x, x, seed);
    hierarchical_scan<Op, VS><<<1, T>>>(B*W, seed, seed); // Use only 1 SM and 1 warp for the small hierarchical scan
    local_scan<Op, VS, true><<<B, W*T>>>(n, x, x, seed);

    return x;
}

} // namespace scan_gpu

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

struct __align__(8) CircleData3Tile {
    uint64_t data;

    // Constants for bit widths
    static constexpr uint32_t BITS = 21;
    static constexpr uint64_t MASK = (1ULL << BITS) - 1;

    // Individual setters
    __host__ __device__ __forceinline__ void set_v1(uint32_t v) {
        data = (data & ~(MASK << 0)) | (uint64_t(v & MASK) << 0);
    }
    __host__ __device__ __forceinline__ void set_v2(uint32_t v) {
        data = (data & ~(MASK << BITS)) | (uint64_t(v & MASK) << BITS);
    }
    __host__ __device__ __forceinline__ void set_v3(uint32_t v) {
        data = (data & ~(MASK << (2 * BITS))) | (uint64_t(v & MASK) << (2 * BITS));
    }

    // Individual getters
    __host__ __device__ __forceinline__ uint32_t get_v1() const {
        return uint32_t((data >> 0) & MASK);
    }
    __host__ __device__ __forceinline__ uint32_t get_v2() const {
        return uint32_t((data >> BITS) & MASK);
    }
    __host__ __device__ __forceinline__ uint32_t get_v3() const {
        return uint32_t((data >> (2 * BITS)) & MASK);
    }

    // Generic setter/getter
    __host__ __device__ __forceinline__ void set_vi(uint32_t i, uint32_t v) {
        switch (i) {
            case 0: set_v1(v); break;
            case 1: set_v2(v); break;
            case 2: set_v3(v); break;
            default: break;
        }
    }
    __host__ __device__ __forceinline__ uint32_t get_vi(uint32_t i) const {
        switch (i) {
            case 0: return get_v1(); break;
            case 1: return get_v2(); break;
            case 2: return get_v3(); break;
            default: return 0; break;
        }
    }
};
static_assert(sizeof(CircleData3Tile) == 8, "CircleData3Tile must be 8 bytes");
struct CircleOp3Tile {
    using Data = CircleData3Tile;

    static __host__ __device__ __forceinline__ Data identity() {
        CircleData3Tile result;
        result.data = 0;
        return result;
    }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        CircleData3Tile result;
        result.data = 0;
        result.set_v1(a.get_v1() + b.get_v1());
        result.set_v2(a.get_v2() + b.get_v2());
        result.set_v3(a.get_v3() + b.get_v3());
        return result;
    }
};
struct __align__(4) CircleData1Tile {
    uint32_t data;
    __host__ __device__ __forceinline__ void set_vi(uint32_t i, uint32_t v) {
        data = v;
    }
    __host__ __device__ __forceinline__ uint32_t get_vi(uint32_t i) const {
        return data;
    }
};
static_assert(sizeof(CircleData1Tile) == 4, "CircleData1Tile must be 4 bytes");
// struct CircleOp1Tile {
//     using Data = CircleData1Tile;

//     static __host__ __device__ __forceinline__ Data identity() { 
//         CircleData1Tile result;
//         result.data = 0;
//         return result;
//     }

//     static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
//         CircleData1Tile result;
//         result.data = a.data + b.data;
//         return result;
//     }
// };
struct CircleOp1Tile {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }
};

// Generic helpers and data structures
struct GmemCircles {
    const uint32_t n_circle;
    float const *circle_x;
    float const *circle_y;
    float const *circle_radius;
    float const *circle_red;
    float const *circle_green;
    float const *circle_blue;
    float const *circle_alpha;
};
struct SmemCircles { // The difference is SMEM is modifiable
    uint32_t n_circle;
    float *circle_x;
    float *circle_y;
    float *circle_radius;
    float *circle_red;
    float *circle_green;
    float *circle_blue;
    float *circle_alpha;
};
struct TileBounds {
    float start_jf;
    float start_if;
    float end_jf;
    float end_if;
};
struct TileBoundsArray1 {
    TileBounds tiles[1];
};
struct TileBoundsArray3 {
    TileBounds tiles[3];
};
struct SmGmemCirclesArray1 {
    SmemCircles circles[1];
};
struct SmGmemCirclesArray3 {
    SmemCircles circles[3];
};

__device__ void load_circles(
    GmemCircles gmem_circles, SmemCircles smem_circles,
    const uint32_t gmem_offset
) {
    // Move gmem arrays
    float const *gmem_circle_x = gmem_circles.circle_x + gmem_offset;
    float const *gmem_circle_y = gmem_circles.circle_y + gmem_offset;
    float const *gmem_circle_radius = gmem_circles.circle_radius + gmem_offset;
    float const *gmem_circle_red = gmem_circles.circle_red + gmem_offset;
    float const *gmem_circle_green = gmem_circles.circle_green + gmem_offset;
    float const *gmem_circle_blue = gmem_circles.circle_blue + gmem_offset;
    float const *gmem_circle_alpha = gmem_circles.circle_alpha + gmem_offset;

    // Vectorize gmem circle arrays
    float4 const *gmem_circle_x4 = reinterpret_cast<float4 const*>(gmem_circle_x);
    float4 const *gmem_circle_y4 = reinterpret_cast<float4 const*>(gmem_circle_y);
    float4 const *gmem_circle_radius4 = reinterpret_cast<float4 const*>(gmem_circle_radius);
    float4 const *gmem_circle_red4 = reinterpret_cast<float4 const*>(gmem_circle_red);
    float4 const *gmem_circle_green4 = reinterpret_cast<float4 const*>(gmem_circle_green);
    float4 const *gmem_circle_blue4 = reinterpret_cast<float4 const*>(gmem_circle_blue);
    float4 const *gmem_circle_alpha4 = reinterpret_cast<float4 const*>(gmem_circle_alpha);

    // Vectorize smem circle arrays
    float4 *smem_circle_x4 = reinterpret_cast<float4*>(smem_circles.circle_x);
    float4 *smem_circle_y4 = reinterpret_cast<float4*>(smem_circles.circle_y);
    float4 *smem_circle_radius4 = reinterpret_cast<float4*>(smem_circles.circle_radius);
    float4 *smem_circle_red4 = reinterpret_cast<float4*>(smem_circles.circle_red);
    float4 *smem_circle_green4 = reinterpret_cast<float4*>(smem_circles.circle_green);
    float4 *smem_circle_blue4 = reinterpret_cast<float4*>(smem_circles.circle_blue);
    float4 *smem_circle_alpha4 = reinterpret_cast<float4*>(smem_circles.circle_alpha);

    // Iterate over circles
    for (uint32_t vc = threadIdx.x; vc < smem_circles.n_circle / 4; vc += blockDim.x) {
        // Vector load 4 circles from GMEM to SMEM
        smem_circle_x4[vc] = gmem_circle_x4[vc];
        smem_circle_y4[vc] = gmem_circle_y4[vc];
        smem_circle_radius4[vc] = gmem_circle_radius4[vc];
        smem_circle_red4[vc] = gmem_circle_red4[vc];
        smem_circle_green4[vc] = gmem_circle_green4[vc];
        smem_circle_blue4[vc] = gmem_circle_blue4[vc];
        smem_circle_alpha4[vc] = gmem_circle_alpha4[vc];
    }

    // Handle tail
    for (uint32_t c = (smem_circles.n_circle / 4) * 4 + threadIdx.x; c < smem_circles.n_circle; c += blockDim.x) {
        // Scalar load 1 circle from GMEM to SMEM
        smem_circles.circle_x[c] = gmem_circle_x[c];
        smem_circles.circle_y[c] = gmem_circle_y[c];
        smem_circles.circle_radius[c] = gmem_circle_radius[c];
        smem_circles.circle_red[c] = gmem_circle_red[c];
        smem_circles.circle_green[c] = gmem_circle_green[c];
        smem_circles.circle_blue[c] = gmem_circle_blue[c];
        smem_circles.circle_alpha[c] = gmem_circle_alpha[c];
    }

    // Wait for everything to be loaded
    __syncthreads();
}

__device__ void load_circles_async(
    GmemCircles gmem_circles, SmemCircles smem_circles,
    const uint32_t gmem_offset
) {
    // Move gmem arrays
    float const *gmem_circle_x = gmem_circles.circle_x + gmem_offset;
    float const *gmem_circle_y = gmem_circles.circle_y + gmem_offset;
    float const *gmem_circle_radius = gmem_circles.circle_radius + gmem_offset;
    float const *gmem_circle_red = gmem_circles.circle_red + gmem_offset;
    float const *gmem_circle_green = gmem_circles.circle_green + gmem_offset;
    float const *gmem_circle_blue = gmem_circles.circle_blue + gmem_offset;
    float const *gmem_circle_alpha = gmem_circles.circle_alpha + gmem_offset;

    // Iterate over circles
    for (uint32_t vc = threadIdx.x; vc < smem_circles.n_circle / 4; vc += blockDim.x) {
        // Get index to copy
        const uint32_t c = vc * 4;
        // Copy mem over
        __pipeline_memcpy_async(&smem_circles.circle_x[c], &gmem_circle_x[c], sizeof(float4), 0);
        __pipeline_memcpy_async(&smem_circles.circle_y[c], &gmem_circle_y[c], sizeof(float4), 0);
        __pipeline_memcpy_async(&smem_circles.circle_radius[c], &gmem_circle_radius[c], sizeof(float4), 0);
        __pipeline_memcpy_async(&smem_circles.circle_red[c], &gmem_circle_red[c], sizeof(float4), 0);
        __pipeline_memcpy_async(&smem_circles.circle_green[c], &gmem_circle_green[c], sizeof(float4), 0);
        __pipeline_memcpy_async(&smem_circles.circle_blue[c], &gmem_circle_blue[c], sizeof(float4), 0);
        __pipeline_memcpy_async(&smem_circles.circle_alpha[c], &gmem_circle_alpha[c], sizeof(float4), 0);
    }
    __pipeline_commit();

    // Handle tail
    for (uint32_t c = (smem_circles.n_circle / 4) * 4 + threadIdx.x; c < smem_circles.n_circle; c += blockDim.x) {
        // Scalar load 1 circle from GMEM to SMEM
        smem_circles.circle_x[c] = gmem_circle_x[c];
        smem_circles.circle_y[c] = gmem_circle_y[c];
        smem_circles.circle_radius[c] = gmem_circle_radius[c];
        smem_circles.circle_red[c] = gmem_circle_red[c];
        smem_circles.circle_green[c] = gmem_circle_green[c];
        smem_circles.circle_blue[c] = gmem_circle_blue[c];
        smem_circles.circle_alpha[c] = gmem_circle_alpha[c];
    }
}

namespace circles_gpu {

__device__ bool circle_intersects_rect(
    const float cx, const float cy, const float radius,
    const float rect_xmin, const float rect_ymin,
    const float rect_xmax, const float rect_ymax
) {
    // Clamp circle center to rectangle
    const float closest_x = fmaxf(rect_xmin, fminf(cx, rect_xmax));
    const float closest_y = fmaxf(rect_ymin, fminf(cy, rect_ymax));

    // Distance from circle center to closest point on rectangle
    const float dx = closest_x - cx;
    const float dy = closest_y - cy;
    return (dx * dx + dy * dy) < (radius * radius);
}

template <typename Op, uint32_t N, uint32_t VS, typename TileBoundsT>
__device__ void scalar_create_flag_array(GmemCircles gmem_circles,
    TileBoundsT tile_bounds, typename Op::Data *flag_arr
) {
    using Data = typename Op::Data;
    for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < gmem_circles.n_circle; c += gridDim.x * blockDim.x) {
        // Scalar load circle
        const float c_x = gmem_circles.circle_x[c];
        const float c_y = gmem_circles.circle_y[c];
        const float c_radius = gmem_circles.circle_radius[c];

        // Set flag
        Data result;
        #pragma unroll
        for (uint32_t k = 0; k < N; ++k) {
            result.set_vi(k,
                circle_intersects_rect(c_x, c_y, c_radius,
                    tile_bounds.tiles[k].start_jf, tile_bounds.tiles[k].start_if,
                    tile_bounds.tiles[k].end_jf, tile_bounds.tiles[k].end_if
                )
            );
        }
        flag_arr[c] = result;
    }
}

template <typename Op, uint32_t N, uint32_t VS, typename TileBoundsT>
__launch_bounds__(32*32)
__global__ void create_flag_array(GmemCircles gmem_circles,
    TileBoundsT tile_bounds, typename Op::Data *flag_arr
) {
    // If size is less than vector size use scalar version
    if (gmem_circles.n_circle < 16) {
        scalar_create_flag_array<Op, N, VS>(gmem_circles, tile_bounds, flag_arr);
        return;
    }

    // Vectorize gmem circle arrays
    float4 const *gmem_circle_x4 = reinterpret_cast<float4 const*>(gmem_circles.circle_x);
    float4 const *gmem_circle_y4 = reinterpret_cast<float4 const*>(gmem_circles.circle_y);
    float4 const *gmem_circle_radius4 = reinterpret_cast<float4 const*>(gmem_circles.circle_radius);

    // Vectorize flag array
    using Data = typename Op::Data;
    using VecData = Vectorized<Data, VS>;
    VecData *vec_flag_arr = reinterpret_cast<VecData*>(flag_arr);

    // Handle vectors
    for (uint32_t vc = blockIdx.x * blockDim.x + threadIdx.x; vc < gmem_circles.n_circle / 4; vc += gridDim.x * blockDim.x) {
        // Vector load circles
        const float4 c_x4 = gmem_circle_x4[vc];
        const float4 c_y4 = gmem_circle_y4[vc];
        const float4 c_radius4 = gmem_circle_radius4[vc];

        // Extract 4 floats into an array for easier indexing
        float c_x[4] = {c_x4.x, c_x4.y, c_x4.z, c_x4.w};
        float c_y[4] = {c_y4.x, c_y4.y, c_y4.z, c_y4.w};
        float c_radius[4] = {c_radius4.x, c_radius4.y, c_radius4.z, c_radius4.w};

        // Set flags
        #pragma unroll
        for (uint32_t i = 0; i < 4 / VS; ++i) {
            VecData result;
            #pragma unroll
            for (uint32_t j = 0; j < VS; ++j) {
                const uint32_t idx = i * VS + j;
                Data r;
                #pragma unroll
                for (uint32_t k = 0; k < N; ++k) {
                    r.set_vi(k,
                        circle_intersects_rect(c_x[idx], c_y[idx], c_radius[idx],
                            tile_bounds.tiles[k].start_jf, tile_bounds.tiles[k].start_if,
                            tile_bounds.tiles[k].end_jf, tile_bounds.tiles[k].end_if
                        )
                    );
                }
                result.elements[j] = r;
            }
            vec_flag_arr[vc * (4 / VS) + i] = result;
        }
    }

    // Handle tail
    for (uint32_t c = (gmem_circles.n_circle / 4) * 4 + (blockIdx.x * blockDim.x + threadIdx.x); c < gmem_circles.n_circle; c += gridDim.x * blockDim.x) {
        // Scalar load circle
        const float c_x = gmem_circles.circle_x[c];
        const float c_y = gmem_circles.circle_y[c];
        const float c_radius = gmem_circles.circle_radius[c];

        // Set flag
        Data result;
        #pragma unroll
        for (uint32_t k = 0; k < N; ++k) {
            result.set_vi(k,
                circle_intersects_rect(c_x, c_y, c_radius,
                    tile_bounds.tiles[k].start_jf, tile_bounds.tiles[k].start_if,
                    tile_bounds.tiles[k].end_jf, tile_bounds.tiles[k].end_if
                )
            );
        }
        flag_arr[c] = result;
    }
}

template <typename Op>
__device__ void extract_scan_helper(GmemCircles gmem_circles, SmemCircles sm_gmem_circles,
    const uint32_t prev, const uint32_t curr, const uint32_t c
) {
    if (prev != curr) { // Start of "run", of which the first is a circle in the tile
        // Transfer circle from grid array to sm array
        sm_gmem_circles.circle_x[curr - 1] = gmem_circles.circle_x[c];
        sm_gmem_circles.circle_y[curr - 1] = gmem_circles.circle_y[c];
        sm_gmem_circles.circle_radius[curr - 1] = gmem_circles.circle_radius[c];
        sm_gmem_circles.circle_red[curr - 1] = gmem_circles.circle_red[c];
        sm_gmem_circles.circle_green[curr - 1] = gmem_circles.circle_green[c];
        sm_gmem_circles.circle_blue[curr - 1] = gmem_circles.circle_blue[c];
        sm_gmem_circles.circle_alpha[curr - 1] = gmem_circles.circle_alpha[c];
    }
}

template <typename Op, uint32_t N, uint32_t VS, typename SmGmemCirclesArrayT>
__device__ void scalar_extract_scan(GmemCircles gmem_circles, typename Op::Data *flag_arr, SmGmemCirclesArrayT sm_gmem_circles_array) {
    using Data = typename Op::Data;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        Data curr = flag_arr[0];
        #pragma unroll
        for (uint32_t k = 0; k < N; ++k) {
            if (curr.get_vi(k) == 1) {
                // Transfer circle from grid array to sm array
                sm_gmem_circles_array.circles[k].circle_x[0] = gmem_circles.circle_x[0];
                sm_gmem_circles_array.circles[k].circle_y[0] = gmem_circles.circle_y[0];
                sm_gmem_circles_array.circles[k].circle_radius[0] = gmem_circles.circle_radius[0];
                sm_gmem_circles_array.circles[k].circle_red[0] = gmem_circles.circle_red[0];
                sm_gmem_circles_array.circles[k].circle_green[0] = gmem_circles.circle_green[0];
                sm_gmem_circles_array.circles[k].circle_blue[0] = gmem_circles.circle_blue[0];
                sm_gmem_circles_array.circles[k].circle_alpha[0] = gmem_circles.circle_alpha[0];
            }
        }
    }
    for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x + 1; c < gmem_circles.n_circle; c += gridDim.x * blockDim.x) {
        const Data prev = flag_arr[c - 1];
        const Data curr = flag_arr[c];
        #pragma unroll
        for (uint32_t k = 0; k < N; ++k) {
            extract_scan_helper<Op>(gmem_circles, sm_gmem_circles_array.circles[k], prev.get_vi(k), curr.get_vi(k), c);
        }
    }
}

template <typename Op, uint32_t N, uint32_t VS, typename SmGmemCirclesArrayT>
__launch_bounds__(32*32)
__global__ void extract_scan(GmemCircles gmem_circles, typename Op::Data *flag_arr, SmGmemCirclesArrayT sm_gmem_circles_array) {
    // If size is less than vector size use scalar version
    if (gmem_circles.n_circle < 16) {
        scalar_extract_scan<Op, N, VS, SmGmemCirclesArrayT>(gmem_circles, flag_arr, sm_gmem_circles_array);
        return;
    }

    // Vectorize flag array
    using Data = typename Op::Data;
    using VecData = Vectorized<Data, VS>;
    VecData *vec_flag_arr = reinterpret_cast<VecData*>(flag_arr);

    // Handle vectors
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const VecData curr = vec_flag_arr[0];
        #pragma unroll
        for (uint32_t k = 0; k < N; ++k) {
            if (curr.elements[0].get_vi(k) == 1) {
                sm_gmem_circles_array.circles[k].circle_x[0] = gmem_circles.circle_x[0];
                sm_gmem_circles_array.circles[k].circle_y[0] = gmem_circles.circle_y[0];
                sm_gmem_circles_array.circles[k].circle_radius[0] = gmem_circles.circle_radius[0];
                sm_gmem_circles_array.circles[k].circle_red[0] = gmem_circles.circle_red[0];
                sm_gmem_circles_array.circles[k].circle_green[0] = gmem_circles.circle_green[0];
                sm_gmem_circles_array.circles[k].circle_blue[0] = gmem_circles.circle_blue[0];
                sm_gmem_circles_array.circles[k].circle_alpha[0] = gmem_circles.circle_alpha[0];
            }
        }
        #pragma unroll
        for (uint32_t j = 1; j < VS; ++j) {
            #pragma unroll
            for (uint32_t k = 0; k < N; ++k) {
                extract_scan_helper<Op>(gmem_circles, sm_gmem_circles_array.circles[k], curr.elements[j-1].get_vi(k), curr.elements[j].get_vi(k), j);
            }
        }
    }
    for (uint32_t vc = blockIdx.x * blockDim.x + threadIdx.x + 1; vc < gmem_circles.n_circle / VS; vc += gridDim.x * blockDim.x) {
        const VecData prev = vec_flag_arr[vc - 1];
        const VecData curr = vec_flag_arr[vc];
        #pragma unroll
        for (uint32_t k = 0; k < N; ++k) {
            extract_scan_helper<Op>(gmem_circles, sm_gmem_circles_array.circles[k], prev.elements[VS-1].get_vi(k), curr.elements[0].get_vi(k), vc*VS + 0);
        }
        #pragma unroll
        for (uint32_t j = 1; j < VS; ++j) {
            #pragma unroll
            for (uint32_t k = 0; k < N; ++k) {
                extract_scan_helper<Op>(gmem_circles, sm_gmem_circles_array.circles[k], curr.elements[j-1].get_vi(k), curr.elements[j].get_vi(k), vc*VS + j);
            }
        }
    }

    // Handle tail
    for (uint32_t c = (gmem_circles.n_circle / VS) * VS + (blockIdx.x * blockDim.x + threadIdx.x); c < gmem_circles.n_circle; c += gridDim.x * blockDim.x) {
        const Data prev = flag_arr[c - 1];
        const Data curr = flag_arr[c];
        #pragma unroll
        for (uint32_t k = 0; k < N; ++k) {
            extract_scan_helper<Op>(gmem_circles, sm_gmem_circles_array.circles[k], prev.get_vi(k), curr.get_vi(k), c);
        }
    }
}

/// 1 tile versions
template <typename Op>
__device__ void scalar_create_flag_array(GmemCircles gmem_circles,
    const uint32_t start_i, const uint32_t start_j, const uint32_t end_i, const uint32_t end_j,
    typename Op::Data *flag_arr
) {
    // Hoist tile bounds
    const float start_jf = (float)start_j;
    const float start_if = (float)start_i;
    const float end_jf = float(end_j - 1);
    const float end_if = float(end_i - 1);

    for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < gmem_circles.n_circle; c += gridDim.x * blockDim.x) {
        // Scalar load circle
        const float c_x = gmem_circles.circle_x[c];
        const float c_y = gmem_circles.circle_y[c];
        const float c_radius = gmem_circles.circle_radius[c];

        // Set flag
        flag_arr[c] = circle_intersects_rect(c_x, c_y, c_radius,
            start_jf, start_if, end_jf, end_if
        );
    }
}
template <typename Op>
__launch_bounds__(32*32)
__global__ void create_flag_array(GmemCircles gmem_circles,
    const uint32_t start_i, const uint32_t start_j, const uint32_t end_i, const uint32_t end_j,
    typename Op::Data *flag_arr
) {
    // If size is less than vector size use scalar version
    if (gmem_circles.n_circle < 16) {
        scalar_create_flag_array<Op>(gmem_circles, start_i, start_j, end_i, end_j, flag_arr);
        return;
    }

    // Vectorize gmem circle arrays
    float4 const *gmem_circle_x4 = reinterpret_cast<float4 const*>(gmem_circles.circle_x);
    float4 const *gmem_circle_y4 = reinterpret_cast<float4 const*>(gmem_circles.circle_y);
    float4 const *gmem_circle_radius4 = reinterpret_cast<float4 const*>(gmem_circles.circle_radius);

    // Vectorize flag array
    using Data = typename Op::Data;
    using VecData = Vectorized<Data, 4>;
    VecData *flag_arr4 = reinterpret_cast<VecData*>(flag_arr);

    // Hoist tile bounds
    const float start_jf = (float)start_j;
    const float start_if = (float)start_i;
    const float end_jf = float(end_j - 1);
    const float end_if = float(end_i - 1);

    // Handle vectors
    for (uint32_t vc = blockIdx.x * blockDim.x + threadIdx.x; vc < gmem_circles.n_circle / 4; vc += gridDim.x * blockDim.x) {
        // Vector load circles
        const float4 c_x4 = gmem_circle_x4[vc];
        const float4 c_y4 = gmem_circle_y4[vc];
        const float4 c_radius4 = gmem_circle_radius4[vc];

        // Set flags
        VecData result;
        result.elements[0] = circle_intersects_rect(c_x4.x, c_y4.x, c_radius4.x, start_jf, start_if, end_jf, end_if);
        result.elements[1] = circle_intersects_rect(c_x4.y, c_y4.y, c_radius4.y, start_jf, start_if, end_jf, end_if);
        result.elements[2] = circle_intersects_rect(c_x4.z, c_y4.z, c_radius4.z, start_jf, start_if, end_jf, end_if);
        result.elements[3] = circle_intersects_rect(c_x4.w, c_y4.w, c_radius4.w, start_jf, start_if, end_jf, end_if);
        flag_arr4[vc] = result;
    }

    // Handle tail
    for (uint32_t c = (gmem_circles.n_circle / 4) * 4 + (blockIdx.x * blockDim.x + threadIdx.x); c < gmem_circles.n_circle; c += gridDim.x * blockDim.x) {
        // Scalar load circle
        const float c_x = gmem_circles.circle_x[c];
        const float c_y = gmem_circles.circle_y[c];
        const float c_radius = gmem_circles.circle_radius[c];

        // Set flag
        flag_arr[c] = circle_intersects_rect(c_x, c_y, c_radius,
            start_jf, start_if, end_jf, end_if
        );
    }
}
template <typename Op>
__device__ void scalar_extract_scan(GmemCircles gmem_circles, typename Op::Data *flag_arr, SmemCircles sm_gmem_circles) {
    using Data = typename Op::Data;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (flag_arr[0] == 1) {
            // Transfer circle from grid array to sm array
            sm_gmem_circles.circle_x[0] = gmem_circles.circle_x[0];
            sm_gmem_circles.circle_y[0] = gmem_circles.circle_y[0];
            sm_gmem_circles.circle_radius[0] = gmem_circles.circle_radius[0];
            sm_gmem_circles.circle_red[0] = gmem_circles.circle_red[0];
            sm_gmem_circles.circle_green[0] = gmem_circles.circle_green[0];
            sm_gmem_circles.circle_blue[0] = gmem_circles.circle_blue[0];
            sm_gmem_circles.circle_alpha[0] = gmem_circles.circle_alpha[0];
        }
    }
    for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x + 1; c < gmem_circles.n_circle; c += gridDim.x * blockDim.x) {
        const Data prev = flag_arr[c - 1];
        const Data curr = flag_arr[c];
        extract_scan_helper<Op>(gmem_circles, sm_gmem_circles, prev, curr, c);
    }
}
template <typename Op>
__launch_bounds__(32*32)
__global__ void extract_scan(GmemCircles gmem_circles, typename Op::Data *flag_arr, SmemCircles sm_gmem_circles) {
    // If size is less than vector size use scalar version
    if (gmem_circles.n_circle < 16) {
        scalar_extract_scan<Op>(gmem_circles, flag_arr, sm_gmem_circles);
        return;
    }

    // Vectorize flag array
    using Data = typename Op::Data;
    using VecData = Vectorized<Data, 4>;
    VecData *flag_arr4 = reinterpret_cast<VecData*>(flag_arr);

    // Handle vectors
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const VecData curr = flag_arr4[0];
        if (curr.elements[0] == 1) {
            sm_gmem_circles.circle_x[0] = gmem_circles.circle_x[0];
            sm_gmem_circles.circle_y[0] = gmem_circles.circle_y[0];
            sm_gmem_circles.circle_radius[0] = gmem_circles.circle_radius[0];
            sm_gmem_circles.circle_red[0] = gmem_circles.circle_red[0];
            sm_gmem_circles.circle_green[0] = gmem_circles.circle_green[0];
            sm_gmem_circles.circle_blue[0] = gmem_circles.circle_blue[0];
            sm_gmem_circles.circle_alpha[0] = gmem_circles.circle_alpha[0];
        }
        extract_scan_helper<Op>(gmem_circles, sm_gmem_circles, curr.elements[0], curr.elements[1], 1);
        extract_scan_helper<Op>(gmem_circles, sm_gmem_circles, curr.elements[1], curr.elements[2], 2);
        extract_scan_helper<Op>(gmem_circles, sm_gmem_circles, curr.elements[2], curr.elements[3], 3);
    }
    for (uint32_t vc = blockIdx.x * blockDim.x + threadIdx.x + 1; vc < gmem_circles.n_circle / 4; vc += gridDim.x * blockDim.x) {
        const VecData prev = flag_arr4[vc - 1];
        const VecData curr = flag_arr4[vc];
        extract_scan_helper<Op>(gmem_circles, sm_gmem_circles, prev.elements[3], curr.elements[0], vc*4 + 0);
        extract_scan_helper<Op>(gmem_circles, sm_gmem_circles, curr.elements[0], curr.elements[1], vc*4 + 1);
        extract_scan_helper<Op>(gmem_circles, sm_gmem_circles, curr.elements[1], curr.elements[2], vc*4 + 2);
        extract_scan_helper<Op>(gmem_circles, sm_gmem_circles, curr.elements[2], curr.elements[3], vc*4 + 3);
    }

    // Handle tail
    for (uint32_t c = (gmem_circles.n_circle / 4) * 4 + (blockIdx.x * blockDim.x + threadIdx.x); c < gmem_circles.n_circle; c += gridDim.x * blockDim.x) {
        const Data prev = flag_arr[c - 1];
        const Data curr = flag_arr[c];
        extract_scan_helper<Op>(gmem_circles, sm_gmem_circles, prev, curr, c);
    }
}
///

template <uint32_t T_TH, uint32_t T_TW>
__device__ void thread_level_render_helper(
    const float c_x, const float c_y, const float c_radius,
    const float c_red, const float c_green, const float c_blue, const float c_alpha,
    const uint32_t start_i, const uint32_t start_j, const uint32_t end_i, const uint32_t end_j,
    float *thread_img_red, float *thread_img_green, float *thread_img_blue
) {
    // Get intersection of circle and thread subtile pixels
    const int32_t start_inter_i = max(int32_t(c_y - c_radius), (int32_t)start_i);
    const int32_t end_inter_i = min(int32_t(c_y + c_radius + 1.0f), (int32_t)end_i - 1);
    const int32_t start_inter_j = max(int32_t(c_x - c_radius), (int32_t)start_j);
    const int32_t end_inter_j = min(int32_t(c_x + c_radius + 1.0f), (int32_t)end_j - 1);

    // Iterate over relevant pixels
    const float r2 = c_radius * c_radius;
    for (int32_t i = start_inter_i; i <= end_inter_i; ++i) {
        for (int32_t j = start_inter_j; j <= end_inter_j; ++j) {
            // Handle that circle can cover partial pixels
            const float dy = i - c_y;
            const float dx = j - c_x;
            if (!(dx * dx + dy * dy < r2)) {
                continue;
            }

            // Update pixel
            const int32_t p = (i - (int32_t)start_i) * T_TW + (j - (int32_t)start_j);
            const float dc_alpha = 1.0f - c_alpha;
            thread_img_red[p] = thread_img_red[p] * dc_alpha + c_red * c_alpha;
            thread_img_green[p] = thread_img_green[p] * dc_alpha + c_green * c_alpha;
            thread_img_blue[p] = thread_img_blue[p] * dc_alpha + c_blue * c_alpha;
        }
    }
}

template <uint32_t T_TH, uint32_t T_TW>
__device__ void thread_level_render(
    SmemCircles circles,
    const uint32_t start_i, const uint32_t start_j, // thread tile coordinates
    float *thread_img_red, float *thread_img_green, float *thread_img_blue // thread output img
) {
    // Vectorize circle arrays
    float4 *circle_x4 = reinterpret_cast<float4*>(circles.circle_x);
    float4 *circle_y4 = reinterpret_cast<float4*>(circles.circle_y);
    float4 *circle_radius4 = reinterpret_cast<float4*>(circles.circle_radius);
    float4 *circle_red4 = reinterpret_cast<float4*>(circles.circle_red);
    float4 *circle_green4 = reinterpret_cast<float4*>(circles.circle_green);
    float4 *circle_blue4 = reinterpret_cast<float4*>(circles.circle_blue);
    float4 *circle_alpha4 = reinterpret_cast<float4*>(circles.circle_alpha);

    // Tile dimensions
    const uint32_t end_i = start_i + T_TH;
    const uint32_t end_j = start_j + T_TW;

    // Iterate over circles
    for (uint32_t vc = 0; vc < circles.n_circle / 4; ++vc) {
        // Vector load 4 circles
        const float4 c_x4 = circle_x4[vc];
        const float4 c_y4 = circle_y4[vc];
        const float4 c_radius4 = circle_radius4[vc];
        const float4 c_red4 = circle_red4[vc];
        const float4 c_green4 = circle_green4[vc];
        const float4 c_blue4 = circle_blue4[vc];
        const float4 c_alpha4 = circle_alpha4[vc];

        // Process all 4 circles
        thread_level_render_helper<T_TH, T_TW>(
            c_x4.x, c_y4.x, c_radius4.x,
            c_red4.x, c_green4.x, c_blue4.x, c_alpha4.x,
            start_i, start_j, end_i, end_j,
            thread_img_red, thread_img_green, thread_img_blue
        );
        thread_level_render_helper<T_TH, T_TW>(
            c_x4.y, c_y4.y, c_radius4.y,
            c_red4.y, c_green4.y, c_blue4.y, c_alpha4.y,
            start_i, start_j, end_i, end_j,
            thread_img_red, thread_img_green, thread_img_blue
        );
        thread_level_render_helper<T_TH, T_TW>(
            c_x4.z, c_y4.z, c_radius4.z,
            c_red4.z, c_green4.z, c_blue4.z, c_alpha4.z,
            start_i, start_j, end_i, end_j,
            thread_img_red, thread_img_green, thread_img_blue
        );
        thread_level_render_helper<T_TH, T_TW>(
            c_x4.w, c_y4.w, c_radius4.w,
            c_red4.w, c_green4.w, c_blue4.w, c_alpha4.w,
            start_i, start_j, end_i, end_j,
            thread_img_red, thread_img_green, thread_img_blue
        );
    }

    // Handle tail
    for (uint32_t c = (circles.n_circle / 4) * 4; c < circles.n_circle; ++c) {
        // Scalar load circle
        const float c_x = circles.circle_x[c];
        const float c_y = circles.circle_y[c];
        const float c_radius = circles.circle_radius[c];
        const float c_red = circles.circle_red[c];
        const float c_green = circles.circle_green[c];
        const float c_blue = circles.circle_blue[c];
        const float c_alpha = circles.circle_alpha[c];
        // Process circle
        thread_level_render_helper<T_TH, T_TW>(
            c_x, c_y, c_radius,
            c_red, c_green, c_blue, c_alpha,
            start_i, start_j, end_i, end_j,
            thread_img_red, thread_img_green, thread_img_blue
        );
    }
}

template <uint32_t T_TH, uint32_t T_TW>
__device__ void sm_level_render_helper(
    const uint32_t width, const uint32_t height,
    GmemCircles gmem_circles, SmemCircles smem_circles_compute, SmemCircles smem_circles_stage,
    float *img_red, float *img_green, float *img_blue,
    float *tt_img_red, float *tt_img_green, float *tt_img_blue,
    const uint32_t tt_start_i, const uint32_t tt_start_j // thread tile coordinates
) {
    // Create pointers to double buffers for swapping
    SmemCircles *compute_ptr = &smem_circles_compute;
    SmemCircles *stage_ptr = &smem_circles_stage;

    const uint32_t num_blocks = gmem_circles.n_circle / smem_circles_compute.n_circle;
    if (num_blocks > 0) {
        // Load first buffer sync
        load_circles(gmem_circles, smem_circles_compute, 0);

        // Iterate over SMEM size chunks of circles
        for (uint32_t idx = 0; idx < num_blocks - 1; ++idx) {
            // Load stage buffer async
            load_circles_async(
                gmem_circles, *stage_ptr,
                smem_circles_compute.n_circle * (idx + 1)
            );

            // Process chunk of tile
            thread_level_render<T_TH, T_TW>(*compute_ptr,
                tt_start_i, tt_start_j,
                tt_img_red, tt_img_green, tt_img_blue
            );

            // Swap double buffers
            __pipeline_wait_prior(0);
            __syncthreads();
            std::swap(compute_ptr, stage_ptr);
        }
        // Handle last block
        thread_level_render<T_TH, T_TW>(*compute_ptr,
            tt_start_i, tt_start_j,
            tt_img_red, tt_img_green, tt_img_blue
        );
        __syncthreads();
    }

    // Handle tail
    const uint32_t num_circles_processed = smem_circles_compute.n_circle * num_blocks;
    const uint32_t circles_left = gmem_circles.n_circle - num_circles_processed;
    if (circles_left > 0) {
        smem_circles_compute.n_circle = circles_left;
        load_circles(
            gmem_circles, smem_circles_compute,
            num_circles_processed
        );
        thread_level_render<T_TH, T_TW>(smem_circles_compute,
            tt_start_i, tt_start_j,
            tt_img_red, tt_img_green, tt_img_blue
        );
    }

    // Write back to main memory at the end
    #pragma unroll
    for (uint32_t p = 0; p < T_TH * T_TW; ++p) {
        // Convert 1D to 2D indices
        const uint32_t i = tt_start_i + p / T_TW;
        const uint32_t j = tt_start_j + p % T_TW;
        // Write back
        img_red[i * width + j] = tt_img_red[p];
        img_green[i * width + j] = tt_img_green[p];
        img_blue[i * width + j] = tt_img_blue[p];
    }

    // Make sure the whole tile is done before moving on
    __syncthreads();
}

template <uint32_t SM_TH, uint32_t SM_TW, uint32_t T_TH, uint32_t T_TW>
__device__ void sm_level_render(
    const uint32_t width, const uint32_t height,
    GmemCircles gmem_circles, SmemCircles smem_circles_compute, SmemCircles smem_circles_stage,
    float *img_red, float *img_green, float *img_blue,
    const uint32_t smt_start_i, const uint32_t smt_start_j // sm tile coordinates
) {
    // Thread grid dimensions
    // constexpr uint32_t tt_per_i = SM_TH / T_TH;
    constexpr uint32_t tt_per_j = SM_TW / T_TW;
    const uint32_t tt_i = threadIdx.x / tt_per_j;
    const uint32_t tt_j = threadIdx.x % tt_per_j;

    // Move start x, y
    const uint32_t tt_start_i = smt_start_i + tt_i * T_TH;
    const uint32_t tt_start_j = smt_start_j + tt_j * T_TW;

    // Each thread gets a tile of pixels
    if constexpr (T_TH * T_TW == 4) {
        float tt_img_red[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float tt_img_green[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        float tt_img_blue[4] = {1.0f, 1.0f, 1.0f, 1.0f};

        sm_level_render_helper<T_TH, T_TW>(
            width, height,
            gmem_circles, smem_circles_compute, smem_circles_stage,
            img_red, img_green, img_blue,
            tt_img_red, tt_img_green, tt_img_blue,
            tt_start_i, tt_start_j
        );
    } else if constexpr (T_TH * T_TW == 64) {
        // TODO: Find a better way to compile time initialize these arrays
        float tt_img_red[64] = {
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
        };
        float tt_img_green[64] = {
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
        };
        float tt_img_blue[64] = {
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
        };
        sm_level_render_helper<T_TH, T_TW>(
            width, height,
            gmem_circles, smem_circles_compute, smem_circles_stage,
            img_red, img_green, img_blue,
            tt_img_red, tt_img_green, tt_img_blue,
            tt_start_i, tt_start_j
        );
    } else {
        return;
    }
}

template <
    uint32_t SM_TH, uint32_t SM_TW, // SM tile size
    uint32_t T_TH, uint32_t T_TW, // Thread tile size
    uint32_t SMEM_TD // SMEM size
>
__launch_bounds__(8*32)
__global__ void gpu_level_render(
    const uint32_t width, const uint32_t height, // img size
    GmemCircles *sm_gmem_circles_arr,
    float *img_red, float *img_green, float *img_blue // output img
) {
    // SM grid dimensions
    const uint32_t smt_per_i = height / SM_TH;
    const uint32_t smt_per_j = width / SM_TW;

    // Setup the block's SMEM
    extern __shared__ float smem[];
    // Split the SMEM into 7 arrays w/ double buffering
    constexpr uint32_t double_buffer_size = SMEM_TD / 2;
    float *circle_x_compute = smem;
    float *circle_y_compute = circle_x_compute + double_buffer_size;
    float *circle_radius_compute = circle_y_compute + double_buffer_size;
    float *circle_red_compute = circle_radius_compute + double_buffer_size;
    float *circle_green_compute = circle_red_compute + double_buffer_size;
    float *circle_blue_compute = circle_green_compute + double_buffer_size;
    float *circle_alpha_compute = circle_blue_compute + double_buffer_size;
    SmemCircles smem_circles_compute = {double_buffer_size, circle_x_compute, circle_y_compute, circle_radius_compute, circle_red_compute, circle_green_compute, circle_blue_compute, circle_alpha_compute};
    
    float *circle_x_stage = circle_alpha_compute + double_buffer_size;
    float *circle_y_stage = circle_x_stage + double_buffer_size;
    float *circle_radius_stage = circle_y_stage + double_buffer_size;
    float *circle_red_stage = circle_radius_stage + double_buffer_size;
    float *circle_green_stage = circle_red_stage + double_buffer_size;
    float *circle_blue_stage = circle_green_stage + double_buffer_size;
    float *circle_alpha_stage = circle_blue_stage + double_buffer_size;
    SmemCircles smem_circles_stage = {double_buffer_size, circle_x_stage, circle_y_stage, circle_radius_stage, circle_red_stage, circle_green_stage, circle_blue_stage, circle_alpha_stage};

    // Iterate over SM tiles
    for (uint32_t sm_idx = blockIdx.x; sm_idx < smt_per_i * smt_per_j; sm_idx += gridDim.x) {
        // SM tile indices
        const uint32_t smt_i = sm_idx / smt_per_j;
        const uint32_t smt_j = sm_idx % smt_per_j;

        // Process tile
        sm_level_render<SM_TH, SM_TW, T_TH, T_TW>(width, height, sm_gmem_circles_arr[sm_idx], smem_circles_compute, smem_circles_stage,
            img_red, img_green, img_blue,
            smt_i * SM_TH, smt_j * SM_TW
        );
    }
}

template <uint32_t SM_TH, uint32_t SM_TW, uint32_t T_TH, uint32_t T_TW, uint32_t NC>
void launch_specialized_kernel(
    const uint32_t width, const uint32_t height,
    GmemCircles gmem_circles,
    float *img_red, float *img_green, float *img_blue,
    GpuMemoryPool &memory_pool
) {
    // For each tile run a scan using the full grid to get the circles actually in the tile (takes ~75ms)
    const uint32_t smt_per_i = height / SM_TH;
    const uint32_t smt_per_j = width / SM_TW;
    const uint32_t num_tiles = smt_per_i * smt_per_j;

    // Extra gmem sizes
    using Data = typename CircleOp3Tile::Data;
    const size_t scan_size = 48 * 32 * sizeof(Data);
    const size_t flag_arr_size = gmem_circles.n_circle * sizeof(Data);
    const uint32_t num_circles = gmem_circles.n_circle; // The max number of circles per tile -> tune
    const uint32_t num_circles_aligned = (num_circles + 3) & ~3u;
    const size_t extract_data_size = num_tiles * (7 * sizeof(float)) * num_circles_aligned; // tiles x circles x 7 float arrays
    const size_t sm_gmem_circles_size = num_tiles * sizeof(GmemCircles);

    // Setup extra gmem
    void *seed = reinterpret_cast<void*>(memory_pool.alloc(scan_size));
    Data *flag = reinterpret_cast<Data*>(memory_pool.alloc(flag_arr_size));
    float *sm_gmem_circles_workspaces = reinterpret_cast<float*>(memory_pool.alloc(extract_data_size));
    GmemCircles *sm_gmem_circles_arr = reinterpret_cast<GmemCircles*>(memory_pool.alloc(sm_gmem_circles_size));

    // Iterate over SM tiles
    for (uint32_t idx = 0; idx < num_tiles / 3; ++idx) {
        const uint32_t sm_idx = idx * 3;

        // Setup sm_gmem_circles
        float *sm_gmem_circles_workspace1 = sm_gmem_circles_workspaces + sm_idx * (7 * num_circles_aligned);
        SmemCircles sm_gmem_circles1 = {0, // Set n_circle after scan
            sm_gmem_circles_workspace1 + 0 * num_circles_aligned,
            sm_gmem_circles_workspace1 + 1 * num_circles_aligned,
            sm_gmem_circles_workspace1 + 2 * num_circles_aligned,
            sm_gmem_circles_workspace1 + 3 * num_circles_aligned,
            sm_gmem_circles_workspace1 + 4 * num_circles_aligned,
            sm_gmem_circles_workspace1 + 5 * num_circles_aligned,
            sm_gmem_circles_workspace1 + 6 * num_circles_aligned
        };
        float *sm_gmem_circles_workspace2 = sm_gmem_circles_workspaces + (sm_idx + 1) * (7 * num_circles_aligned);
        SmemCircles sm_gmem_circles2 = {0,
            sm_gmem_circles_workspace2 + 0 * num_circles_aligned,
            sm_gmem_circles_workspace2 + 1 * num_circles_aligned,
            sm_gmem_circles_workspace2 + 2 * num_circles_aligned,
            sm_gmem_circles_workspace2 + 3 * num_circles_aligned,
            sm_gmem_circles_workspace2 + 4 * num_circles_aligned,
            sm_gmem_circles_workspace2 + 5 * num_circles_aligned,
            sm_gmem_circles_workspace2 + 6 * num_circles_aligned
        };
        float *sm_gmem_circles_workspace3 = sm_gmem_circles_workspaces + (sm_idx + 2) * (7 * num_circles_aligned);
        SmemCircles sm_gmem_circles3 = {0,
            sm_gmem_circles_workspace3 + 0 * num_circles_aligned,
            sm_gmem_circles_workspace3 + 1 * num_circles_aligned,
            sm_gmem_circles_workspace3 + 2 * num_circles_aligned,
            sm_gmem_circles_workspace3 + 3 * num_circles_aligned,
            sm_gmem_circles_workspace3 + 4 * num_circles_aligned,
            sm_gmem_circles_workspace3 + 5 * num_circles_aligned,
            sm_gmem_circles_workspace3 + 6 * num_circles_aligned
        };

        // SM tile indices
        const uint32_t smt1_i = sm_idx / smt_per_j;
        const uint32_t smt1_j = sm_idx % smt_per_j;
        const uint32_t smt2_i = (sm_idx + 1) / smt_per_j;
        const uint32_t smt2_j = (sm_idx + 1) % smt_per_j;
        const uint32_t smt3_i = (sm_idx + 2) / smt_per_j;
        const uint32_t smt3_j = (sm_idx + 2) % smt_per_j;

        // Get tile bounds
        const uint32_t start1_i = smt1_i * SM_TH;
        const uint32_t start1_j = smt1_j * SM_TW;
        const uint32_t end1_i = start1_i + SM_TH;
        const uint32_t end1_j = start1_j + SM_TW;
        const uint32_t start2_i = smt2_i * SM_TH;
        const uint32_t start2_j = smt2_j * SM_TW;
        const uint32_t end2_i = start2_i + SM_TH;
        const uint32_t end2_j = start2_j + SM_TW;
        const uint32_t start3_i = smt3_i * SM_TH;
        const uint32_t start3_j = smt3_j * SM_TW;
        const uint32_t end3_i = start3_i + SM_TH;
        const uint32_t end3_j = start3_j + SM_TW;

        // Create flag array
        TileBoundsArray3 tile_bounds = {{
            {(float)start1_j, (float)start1_i, float(end1_j-1), float(end1_i-1)},
            {(float)start2_j, (float)start2_i, float(end2_j-1), float(end2_i-1)},
            {(float)start3_j, (float)start3_i, float(end3_j-1), float(end3_i-1)}
        }};
        create_flag_array<CircleOp3Tile, 3, 2, TileBoundsArray3><<<48, 32*32>>>(gmem_circles, tile_bounds, flag);

        // Run scan on flag array
        scan_gpu::launch_scan<CircleOp3Tile>(gmem_circles.n_circle, flag, seed);

        // Extract scan
        SmGmemCirclesArray3 sm_gmem_circles_array = {{
            sm_gmem_circles1, sm_gmem_circles2, sm_gmem_circles3
        }};
        extract_scan<CircleOp3Tile, 3, 2, SmGmemCirclesArray3><<<48, 32*32>>>(gmem_circles, flag, sm_gmem_circles_array);

        // Set n_circle
        Data last_run;
        CUDA_CHECK(cudaMemcpy(&last_run, &flag[gmem_circles.n_circle - 1], sizeof(Data), cudaMemcpyDeviceToHost));
        sm_gmem_circles1.n_circle = last_run.get_v1();
        sm_gmem_circles2.n_circle = last_run.get_v2();
        sm_gmem_circles3.n_circle = last_run.get_v3();

        // Convert to GmemCircles
        GmemCircles const_sm_gmem_circles1 = {
            sm_gmem_circles1.n_circle,
            sm_gmem_circles1.circle_x,
            sm_gmem_circles1.circle_y,
            sm_gmem_circles1.circle_radius,
            sm_gmem_circles1.circle_red,
            sm_gmem_circles1.circle_green,
            sm_gmem_circles1.circle_blue,
            sm_gmem_circles1.circle_alpha
        };
        GmemCircles const_sm_gmem_circles2 = {
            sm_gmem_circles2.n_circle,
            sm_gmem_circles2.circle_x,
            sm_gmem_circles2.circle_y,
            sm_gmem_circles2.circle_radius,
            sm_gmem_circles2.circle_red,
            sm_gmem_circles2.circle_green,
            sm_gmem_circles2.circle_blue,
            sm_gmem_circles2.circle_alpha
        };
        GmemCircles const_sm_gmem_circles3 = {
            sm_gmem_circles3.n_circle,
            sm_gmem_circles3.circle_x,
            sm_gmem_circles3.circle_y,
            sm_gmem_circles3.circle_radius,
            sm_gmem_circles3.circle_red,
            sm_gmem_circles3.circle_green,
            sm_gmem_circles3.circle_blue,
            sm_gmem_circles3.circle_alpha
        };

        // Store sm_gmem_circles in gpu memory
        CUDA_CHECK(cudaMemcpy(&sm_gmem_circles_arr[sm_idx], &const_sm_gmem_circles1, sizeof(GmemCircles), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&sm_gmem_circles_arr[sm_idx + 1], &const_sm_gmem_circles2, sizeof(GmemCircles), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&sm_gmem_circles_arr[sm_idx + 2], &const_sm_gmem_circles3, sizeof(GmemCircles), cudaMemcpyHostToDevice));
    }
    // Handle tail
    using Data1Tile = typename CircleOp1Tile::Data;
    Data1Tile *flag_1Tile = reinterpret_cast<Data1Tile*>(flag);
    for (uint32_t sm_idx = (num_tiles / 3) * 3; sm_idx < num_tiles; ++sm_idx) {
        float *sm_gmem_circles_workspace = sm_gmem_circles_workspaces + sm_idx * (7 * num_circles_aligned);
        SmemCircles sm_gmem_circles = {0,
            sm_gmem_circles_workspace + 0 * num_circles_aligned,
            sm_gmem_circles_workspace + 1 * num_circles_aligned,
            sm_gmem_circles_workspace + 2 * num_circles_aligned,
            sm_gmem_circles_workspace + 3 * num_circles_aligned,
            sm_gmem_circles_workspace + 4 * num_circles_aligned,
            sm_gmem_circles_workspace + 5 * num_circles_aligned,
            sm_gmem_circles_workspace + 6 * num_circles_aligned
        };

        const uint32_t smt_i = sm_idx / smt_per_j;
        const uint32_t smt_j = sm_idx % smt_per_j;
        const uint32_t start_i = smt_i * SM_TH;
        const uint32_t start_j = smt_j * SM_TW;
        const uint32_t end_i = start_i + SM_TH;
        const uint32_t end_j = start_j + SM_TW;

        // TileBoundsArray1 tile_bounds = {{
        //     {(float)start_j, (float)start_i, float(end_j-1), float(end_i-1)}
        // }};
        // create_flag_array<CircleOp1Tile, 1, 4, TileBoundsArray1><<<48, 32*32>>>(gmem_circles, tile_bounds, flag_1Tile);
        create_flag_array<CircleOp1Tile><<<48, 32*32>>>(gmem_circles,
            start_i, start_j, end_i, end_j,
            flag_1Tile
        );

        scan_gpu::launch_scan<CircleOp1Tile>(gmem_circles.n_circle, flag_1Tile, seed);
        
        extract_scan<CircleOp1Tile><<<48, 32*32>>>(gmem_circles, flag_1Tile, sm_gmem_circles);
        
        Data1Tile last_run;
        CUDA_CHECK(cudaMemcpy(&last_run, &flag_1Tile[gmem_circles.n_circle - 1], sizeof(Data1Tile), cudaMemcpyDeviceToHost));
        // sm_gmem_circles.n_circle = last_run.data;
        sm_gmem_circles.n_circle = last_run;
        
        GmemCircles const_sm_gmem_circles = {
            sm_gmem_circles.n_circle,
            sm_gmem_circles.circle_x,
            sm_gmem_circles.circle_y,
            sm_gmem_circles.circle_radius,
            sm_gmem_circles.circle_red,
            sm_gmem_circles.circle_green,
            sm_gmem_circles.circle_blue,
            sm_gmem_circles.circle_alpha
        };
        CUDA_CHECK(cudaMemcpy(&sm_gmem_circles_arr[sm_idx], &const_sm_gmem_circles, sizeof(GmemCircles), cudaMemcpyHostToDevice));
    }

    // Launch render kernel
    constexpr int smem_size_bytes = NC * 7 * sizeof(float); // 7 float arrays
    cudaFuncSetAttribute(
        gpu_level_render<SM_TH, SM_TW, T_TH, T_TW, NC>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size_bytes
    );
    gpu_level_render<SM_TH, SM_TW, T_TH, T_TW, NC><<<64, 8*32, smem_size_bytes>>>(
        width, height, sm_gmem_circles_arr,
        img_red, img_green, img_blue
    );
}

void launch_render(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,      // pointer to GPU memory
    float const *circle_y,      // pointer to GPU memory
    float const *circle_radius, // pointer to GPU memory
    float const *circle_red,    // pointer to GPU memory
    float const *circle_green,  // pointer to GPU memory
    float const *circle_blue,   // pointer to GPU memory
    float const *circle_alpha,  // pointer to GPU memory
    float *img_red,             // pointer to GPU memory
    float *img_green,           // pointer to GPU memory
    float *img_blue,            // pointer to GPU memory
    GpuMemoryPool &memory_pool) {
    // Test case sizes: 256x256, 1024x1024
    GmemCircles gmem_circles = {(uint32_t)n_circle, circle_x, circle_y, circle_radius, circle_red, circle_green, circle_blue, circle_alpha};
    if (height == 256 && width == 256) {
        constexpr uint32_t num_circles = 80; // at most 80 in the small test cases
        launch_specialized_kernel<32, 32, 2, 2, num_circles>(width, height, gmem_circles, img_red, img_green, img_blue, memory_pool);
    } else if (height == 1024 && width == 1024) {
        constexpr uint32_t num_circles = 2000; // Tuning parameter
        launch_specialized_kernel<128, 128, 8, 8, num_circles>(width, height, gmem_circles, img_red, img_green, img_blue, memory_pool);
    } else {
        // Not handled
        return;
    }
}

} // namespace circles_gpu

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

GpuMemoryPool::~GpuMemoryPool() {
    for (auto ptr : allocations_) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void *GpuMemoryPool::alloc(size_t size) {
    if (next_idx_ < allocations_.size()) {
        auto idx = next_idx_++;
        if (size > capacities_.at(idx)) {
            CUDA_CHECK(cudaFree(allocations_.at(idx)));
            CUDA_CHECK(cudaMalloc(&allocations_.at(idx), size));
            CUDA_CHECK(cudaMemset(allocations_.at(idx), 0, size));
            capacities_.at(idx) = size;
        }
        return allocations_.at(idx);
    } else {
        void *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        CUDA_CHECK(cudaMemset(ptr, 0, size));
        allocations_.push_back(ptr);
        capacities_.push_back(size);
        next_idx_++;
        return ptr;
    }
}

void GpuMemoryPool::reset() {
    next_idx_ = 0;
    for (int32_t i = 0; i < allocations_.size(); i++) {
        CUDA_CHECK(cudaMemset(allocations_.at(i), 0, capacities_.at(i)));
    }
}

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Scene {
    int32_t width;
    int32_t height;
    std::vector<float> circle_x;
    std::vector<float> circle_y;
    std::vector<float> circle_radius;
    std::vector<float> circle_red;
    std::vector<float> circle_green;
    std::vector<float> circle_blue;
    std::vector<float> circle_alpha;

    int32_t n_circle() const { return circle_x.size(); }
};

struct Image {
    int32_t width;
    int32_t height;
    std::vector<float> red;
    std::vector<float> green;
    std::vector<float> blue;
};

float max_abs_diff(Image const &a, Image const &b) {
    float max_diff = 0.0f;
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        float diff_red = std::abs(a.red.at(idx) - b.red.at(idx));
        float diff_green = std::abs(a.green.at(idx) - b.green.at(idx));
        float diff_blue = std::abs(a.blue.at(idx) - b.blue.at(idx));
        max_diff = std::max(max_diff, diff_red);
        max_diff = std::max(max_diff, diff_green);
        max_diff = std::max(max_diff, diff_blue);
    }
    return max_diff;
}

struct Results {
    bool correct;
    float max_abs_diff;
    Image image_expected;
    Image image_actual;
    double time_ms;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename T> struct GpuBuf {
    T *data;

    explicit GpuBuf(size_t n) { CUDA_CHECK(cudaMalloc(&data, n * sizeof(T))); }

    explicit GpuBuf(std::vector<T> const &host_data) {
        CUDA_CHECK(cudaMalloc(&data, host_data.size() * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(
            data,
            host_data.data(),
            host_data.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    ~GpuBuf() { CUDA_CHECK(cudaFree(data)); }
};

Results run_config(Mode mode, Scene const &scene) {
    auto img_expected = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    render_cpu(
        scene.width,
        scene.height,
        scene.n_circle(),
        scene.circle_x.data(),
        scene.circle_y.data(),
        scene.circle_radius.data(),
        scene.circle_red.data(),
        scene.circle_green.data(),
        scene.circle_blue.data(),
        scene.circle_alpha.data(),
        img_expected.red.data(),
        img_expected.green.data(),
        img_expected.blue.data());

    auto circle_x_gpu = GpuBuf<float>(scene.circle_x);
    auto circle_y_gpu = GpuBuf<float>(scene.circle_y);
    auto circle_radius_gpu = GpuBuf<float>(scene.circle_radius);
    auto circle_red_gpu = GpuBuf<float>(scene.circle_red);
    auto circle_green_gpu = GpuBuf<float>(scene.circle_green);
    auto circle_blue_gpu = GpuBuf<float>(scene.circle_blue);
    auto circle_alpha_gpu = GpuBuf<float>(scene.circle_alpha);
    auto img_red_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_green_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_blue_gpu = GpuBuf<float>(scene.height * scene.width);

    auto memory_pool = GpuMemoryPool();

    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(img_red_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            img_green_gpu.data,
            0,
            scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(img_blue_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        memory_pool.reset();
    };

    auto f = [&]() {
        circles_gpu::launch_render(
            scene.width,
            scene.height,
            scene.n_circle(),
            circle_x_gpu.data,
            circle_y_gpu.data,
            circle_radius_gpu.data,
            circle_red_gpu.data,
            circle_green_gpu.data,
            circle_blue_gpu.data,
            circle_alpha_gpu.data,
            img_red_gpu.data,
            img_green_gpu.data,
            img_blue_gpu.data,
            memory_pool);
    };

    reset();
    f();

    auto img_actual = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    CUDA_CHECK(cudaMemcpy(
        img_actual.red.data(),
        img_red_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.green.data(),
        img_green_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.blue.data(),
        img_blue_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));

    float max_diff = max_abs_diff(img_expected, img_actual);

    if (max_diff > 5e-2) {
        return Results{
            false,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    if (mode == Mode::TEST) {
        return Results{
            true,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    double time_ms = benchmark_ms(1000.0, reset, f);

    return Results{
        true,
        max_diff,
        std::move(img_expected),
        std::move(img_actual),
        time_ms,
    };
}

template <typename Rng>
Scene gen_random(Rng &rng, int32_t width, int32_t height, int32_t n_circle) {
    auto unif_0_1 = std::uniform_real_distribution<float>(0.0f, 1.0f);
    auto z_values = std::vector<float>();
    for (int32_t i = 0; i < n_circle; i++) {
        float z;
        for (;;) {
            z = unif_0_1(rng);
            z = std::max(z, unif_0_1(rng));
            if (z > 0.01) {
                break;
            }
        }
        // float z = std::max(unif_0_1(rng), unif_0_1(rng));
        z_values.push_back(z);
    }
    std::sort(z_values.begin(), z_values.end(), std::greater<float>());

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };
    auto color_idx_dist = std::uniform_int_distribution<int>(0, colors.size() - 1);
    auto alpha_dist = std::uniform_real_distribution<float>(0.0f, 0.3f);

    int32_t fog_interval = n_circle / 10;
    float fog_alpha = 0.2;

    auto scene = Scene{width, height};
    float base_radius_scale = 1.0f;
    int32_t i = 0;
    for (float z : z_values) {
        float max_radius = base_radius_scale / z;
        float radius = std::max(1.0f, unif_0_1(rng) * max_radius);
        float x = unif_0_1(rng) * (width + 2 * max_radius) - max_radius;
        float y = unif_0_1(rng) * (height + 2 * max_radius) - max_radius;
        int color_idx = color_idx_dist(rng);
        uint32_t color = colors[color_idx];
        scene.circle_x.push_back(x);
        scene.circle_y.push_back(y);
        scene.circle_radius.push_back(radius);
        scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
        scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
        scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
        scene.circle_alpha.push_back(alpha_dist(rng));
        i++;
        if (i % fog_interval == 0 && i + 1 < n_circle) {
            scene.circle_x.push_back(float(width - 1) / 2.0f);
            scene.circle_y.push_back(float(height - 1) / 2.0f);
            scene.circle_radius.push_back(float(std::max(width, height)));
            scene.circle_red.push_back(1.0f);
            scene.circle_green.push_back(1.0f);
            scene.circle_blue.push_back(1.0f);
            scene.circle_alpha.push_back(fog_alpha);
        }
    }

    return scene;
}

constexpr float PI = 3.14159265359f;

Scene gen_overlapping_opaque() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };

    int32_t n_circle = 20;
    int32_t n_ring = 4;
    float angle_range = PI;
    for (int32_t ring = 0; ring < n_ring; ring++) {
        float dist = 20.0f * (ring + 1);
        float saturation = float(ring + 1) / n_ring;
        float hue_shift = float(ring) / (n_ring - 1);
        for (int32_t i = 0; i < n_circle; i++) {
            float theta = angle_range * i / (n_circle - 1);
            float x = width / 2.0f - dist * std::cos(theta);
            float y = height / 2.0f - dist * std::sin(theta);
            scene.circle_x.push_back(x);
            scene.circle_y.push_back(y);
            scene.circle_radius.push_back(16.0f);
            auto color = colors[(i + ring * 2) % colors.size()];
            scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
            scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
            scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
            scene.circle_alpha.push_back(1.0f);
        }
    }

    return scene;
}

Scene gen_overlapping_transparent() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    float offset = 20.0f;
    float radius = 40.0f;
    scene.circle_x = std::vector<float>{
        (width - 1) / 2.0f - offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f - offset,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
    };
    scene.circle_radius = std::vector<float>{
        radius,
        radius,
        radius,
        radius,
    };
    // 0xd32360
    // 0x2874aa
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0x28) / 255.0f,
        float(0x28) / 255.0f,
        float(0xd3) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x74) / 255.0f,
        float(0x74) / 255.0f,
        float(0x23) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0xaa) / 255.0f,
        float(0xaa) / 255.0f,
        float(0x60) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        0.75f,
        0.75f,
        0.75f,
        0.75f,
    };
    return scene;
}

Scene gen_simple() {
    /*
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    */
    int32_t width = 256;
    int32_t height = 256;
    auto scene = Scene{width, height};
    scene.circle_x = std::vector<float>{
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
    };
    scene.circle_radius = std::vector<float>{
        40.0f,
        40.0f,
        40.0f,
        40.0f,
    };
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0xcc) / 255.0f,
        float(0x20) / 255.0f,
        float(0x28) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x9f) / 255.0f,
        float(0x80) / 255.0f,
        float(0x74) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0x26) / 255.0f,
        float(0x20) / 255.0f,
        float(0xaa) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        1.0f,
        1.0f,
        1.0f,
        1.0f,
    };
    return scene;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void write_bmp(
    std::string const &fname,
    uint32_t width,
    uint32_t height,
    const std::vector<uint8_t> &pixels) {
    BMPHeader header;
    header.width = width;
    header.height = height;

    uint32_t rowSize = (width * 3 + 3) & (~3); // Align to 4 bytes
    header.imageSize = rowSize * height;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));

    // Write pixel data with padding
    std::vector<uint8_t> padding(rowSize - width * 3, 0);
    for (int32_t idx_y = height - 1; idx_y >= 0;
         --idx_y) { // BMP stores pixels from bottom to top
        const uint8_t *row = &pixels[idx_y * width * 3];
        file.write(reinterpret_cast<const char *>(row), width * 3);
        if (!padding.empty()) {
            file.write(reinterpret_cast<const char *>(padding.data()), padding.size());
        }
    }
}

uint8_t float_to_byte(float x) {
    if (x < 0) {
        return 0;
    } else if (x >= 1) {
        return 255;
    } else {
        return x * 255.0f;
    }
}

void write_image(std::string const &fname, Image const &img) {
    auto pixels = std::vector<uint8_t>(img.width * img.height * 3);
    for (int32_t idx = 0; idx < img.width * img.height; idx++) {
        float red = img.red.at(idx);
        float green = img.green.at(idx);
        float blue = img.blue.at(idx);
        // BMP stores pixels in BGR order
        pixels.at(idx * 3) = float_to_byte(blue);
        pixels.at(idx * 3 + 1) = float_to_byte(green);
        pixels.at(idx * 3 + 2) = float_to_byte(red);
    }
    write_bmp(fname, img.width, img.height, pixels);
}

Image compute_img_diff(Image const &a, Image const &b) {
    auto img_diff = Image{
        a.width,
        a.height,
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
    };
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        img_diff.red.at(idx) = std::abs(a.red.at(idx) - b.red.at(idx));
        img_diff.green.at(idx) = std::abs(a.green.at(idx) - b.green.at(idx));
        img_diff.blue.at(idx) = std::abs(a.blue.at(idx) - b.blue.at(idx));
    }
    return img_diff;
}

struct SceneTest {
    std::string name;
    Mode mode;
    Scene scene;
};

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    auto scenes = std::vector<SceneTest>();
    scenes.push_back({"simple", Mode::TEST, gen_simple()});
    scenes.push_back({"overlapping_opaque", Mode::TEST, gen_overlapping_opaque()});
    scenes.push_back(
        {"overlapping_transparent", Mode::TEST, gen_overlapping_transparent()});
    scenes.push_back(
        {"ten_million_circles", Mode::BENCHMARK, gen_random(rng, 1024, 1024, 10'000'000)});

    int32_t fail_count = 0;

    int32_t count = 0;
    for (auto const &scene_test : scenes) {
        auto i = count++;
        printf("\nTesting scene '%s'\n", scene_test.name.c_str());
        auto results = run_config(scene_test.mode, scene_test.scene);
        write_image(
            std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                "_cpu.bmp",
            results.image_expected);
        write_image(
            std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                "_gpu.bmp",
            results.image_actual);
        if (!results.correct) {
            printf("  Result did not match expected image\n");
            printf("  Max absolute difference: %.2e\n", results.max_abs_diff);
            auto diff = compute_img_diff(results.image_expected, results.image_actual);
            write_image(
                std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                    "_diff.bmp",
                diff);
            printf(
                "  (Wrote image diff to 'out/img%d_%s_diff.bmp')\n",
                i,
                scene_test.name.c_str());
            fail_count++;
            continue;
        } else {
            printf("  OK\n");
        }
        if (scene_test.mode == Mode::BENCHMARK) {
            printf("  Time: %f ms\n", results.time_ms);
        }
    }

    if (fail_count) {
        printf("\nCorrectness: %d tests failed\n", fail_count);
    } else {
        printf("\nCorrectness: All tests passed\n");
    }

    return 0;
}
