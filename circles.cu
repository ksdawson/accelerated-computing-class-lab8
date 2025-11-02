#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
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
// Optimized GPU Implementation

namespace circles_gpu {

// template <uint32_t T_TH, uint32_t T_TW>
// __device__ void thread_level_render_helper(
//     const float c_x, const float c_y, const float c_radius,
//     const float c_red, const float c_green, const float c_blue, const float c_alpha,
//     const uint32_t start_i, const uint32_t start_j, const uint32_t end_i, const uint32_t end_j,
//     float *thread_img_red, float *thread_img_green, float *thread_img_blue
// ) {
//     // Iterate over **all pixels in the thread tile**, ignoring circle bounds
//     for (int32_t i = 0; i < T_TH; ++i) {
//         for (int32_t j = 0; j < T_TW; ++j) {
//             const int32_t global_i = start_i + i;
//             const int32_t global_j = start_j + j;

//             // Compute dx, dy for circle
//             const float dy = global_i - c_y;
//             const float dx = global_j - c_x;

//             // Only update pixel if inside circle
//             if (!(int32_t(c_x - c_radius) <= global_j && global_j <= int32_t(c_x + c_radius + 1.0f) &&
//                 int32_t(c_y - c_radius) <= global_i && global_i <= int32_t(c_y + c_radius + 1.0f) &&
//                 dx * dx + dy * dy < c_radius * c_radius)) {
//                 continue;
//             }

//             // Update pixel
//             const uint32_t p = i * T_TW + j;
//             thread_img_red[p]   = thread_img_red[p] * (1.0f - c_alpha) + c_red * c_alpha;
//             thread_img_green[p] = thread_img_green[p] * (1.0f - c_alpha) + c_green * c_alpha;
//             thread_img_blue[p]  = thread_img_blue[p] * (1.0f - c_alpha) + c_blue * c_alpha;
//         }
//     }
// }

template <uint32_t T_TH, uint32_t T_TW>
__device__ void thread_level_render_helper(
    const float c_x, const float c_y, const float c_radius,
    const float c_red, const float c_green, const float c_blue, const float c_alpha,
    const uint32_t start_i, const uint32_t start_j, const uint32_t end_i, const uint32_t end_j,
    float *thread_img_red, float *thread_img_green, float *thread_img_blue
) {
    // Might be faster + simpler to just iterate over tile pixels?
    // Get intersection of circle and thread subtile pixels
    const int32_t start_inter_i = max(int32_t(c_y - c_radius), (int32_t)start_i);
    const int32_t end_inter_i = min(int32_t(c_y + c_radius + 1.0f), (int32_t)end_i - 1);
    const int32_t start_inter_j = max(int32_t(c_x - c_radius), (int32_t)start_j);
    const int32_t end_inter_j = min(int32_t(c_x + c_radius + 1.0f), (int32_t)end_j - 1);

    // Iterate over relevant pixels
    for (int32_t i = start_inter_i; i <= end_inter_i; ++i) {
        for (int32_t j = start_inter_j; j <= end_inter_j; ++j) {
            // Handle that circle can cover partial pixels
            const float dy = i - c_y;
            const float dx = j - c_x;
            if (!(dx * dx + dy * dy < c_radius * c_radius)) {
                continue;
            }

            // Update pixel
            const int32_t p = (i - (int32_t)start_i) * T_TW + (j - (int32_t)start_j);
            thread_img_red[p] = thread_img_red[p] * (1.0f - c_alpha) + c_red * c_alpha;
            thread_img_green[p] = thread_img_green[p] * (1.0f - c_alpha) + c_green * c_alpha;
            thread_img_blue[p] = thread_img_blue[p] * (1.0f - c_alpha) + c_blue * c_alpha;
        }
    }
}

template <uint32_t T_TH, uint32_t T_TW>
__device__ void thread_level_render(
    const uint32_t n_circle,
    float const *circle_x, float const *circle_y, float const *circle_radius,
    float const *circle_red, float const *circle_green, float const *circle_blue, float const *circle_alpha,
    const uint32_t start_i, const uint32_t start_j, // thread tile coordinates
    float *thread_img_red, float *thread_img_green, float *thread_img_blue // thread output img
) {
    // Vectorize circle arrays
    float4 const *circle_x4 = reinterpret_cast<float4 const*>(circle_x);
    float4 const *circle_y4 = reinterpret_cast<float4 const*>(circle_y);
    float4 const *circle_radius4 = reinterpret_cast<float4 const*>(circle_radius);
    float4 const *circle_red4 = reinterpret_cast<float4 const*>(circle_red);
    float4 const *circle_green4 = reinterpret_cast<float4 const*>(circle_green);
    float4 const *circle_blue4 = reinterpret_cast<float4 const*>(circle_blue);
    float4 const *circle_alpha4 = reinterpret_cast<float4 const*>(circle_alpha);

    // Tile dimensions
    const uint32_t end_i = start_i + T_TH;
    const uint32_t end_j = start_j + T_TW;

    // Iterate over circles
    for (uint32_t vc = 0; vc < n_circle / 4; ++vc) {
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
    for (uint32_t c = (n_circle / 4) * 4; c < n_circle; ++c) {
        // Scalar load circle
        const float c_x = circle_x[c];
        const float c_y = circle_y[c];
        const float c_radius = circle_radius[c];
        const float c_red = circle_red[c];
        const float c_green = circle_green[c];
        const float c_blue = circle_blue[c];
        const float c_alpha = circle_alpha[c];
        // Process circle
        thread_level_render_helper<T_TH, T_TW>(
            c_x, c_y, c_radius,
            c_red, c_green, c_blue, c_alpha,
            start_i, start_j, end_i, end_j,
            thread_img_red, thread_img_green, thread_img_blue
        );
    }
}

template <uint32_t SM_TH, uint32_t SM_TW, uint32_t T_TH, uint32_t T_TW>
__device__ void sm_level_render(
    const uint32_t width, const uint32_t height,
    const uint32_t n_circle,
    float const *circle_x, float const *circle_y, float const *circle_radius,
    float const *circle_red, float const *circle_green, float const *circle_blue, float const *circle_alpha,
    float *img_red, float *img_green, float *img_blue,
    const uint32_t smt_start_i, const uint32_t smt_start_j // sm tile coordinates
) {
    // Thread grid dimensions
    constexpr uint32_t tt_per_i = SM_TH / T_TH;
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

        // Process tile
        thread_level_render<T_TH, T_TW>(n_circle,
            circle_x, circle_y, circle_radius,
            circle_red, circle_green, circle_blue, circle_alpha,
            tt_start_i, tt_start_j,
            tt_img_red, tt_img_green, tt_img_blue
        );

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
    } else if constexpr (T_TH * T_TW == 64) {
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
        thread_level_render<T_TH, T_TW>(n_circle,
            circle_x, circle_y, circle_radius,
            circle_red, circle_green, circle_blue, circle_alpha,
            tt_start_i, tt_start_j,
            tt_img_red, tt_img_green, tt_img_blue
        );
        #pragma unroll
        for (uint32_t p = 0; p < T_TH * T_TW; ++p) {
            const uint32_t i = tt_start_i + p / T_TW;
            const uint32_t j = tt_start_j + p % T_TW;
            img_red[i * width + j] = tt_img_red[p];
            img_green[i * width + j] = tt_img_green[p];
            img_blue[i * width + j] = tt_img_blue[p];
        }
    } else {
        return;
    }
}

template <
    uint32_t SM_TH, uint32_t SM_TW, // SM tile size
    uint32_t T_TH, uint32_t T_TW // Thread tile size
>
__global__ void gpu_level_render(
    uint32_t width, uint32_t height, // img size
    uint32_t n_circle, // num circles
    float const *circle_x, float const *circle_y, float const *circle_radius, // circle size arrays
    float const *circle_red, float const *circle_green, float const *circle_blue, float const *circle_alpha, // circle color arrays
    float *img_red, float *img_green, float *img_blue // output img
) {
    // SM grid dimensions
    const uint32_t smt_per_i = height / SM_TH;
    const uint32_t smt_per_j = width / SM_TW;

    // Iterate over SM tiles
    for (uint32_t sm_idx = blockIdx.x; sm_idx < smt_per_i * smt_per_j; sm_idx += gridDim.x) {
        // SM tile indices
        const uint32_t smt_i = sm_idx / smt_per_j;
        const uint32_t smt_j = sm_idx % smt_per_j;

        // Process tile
        sm_level_render<SM_TH, SM_TW, T_TH, T_TW>(width, height, n_circle,
            circle_x, circle_y, circle_radius,
            circle_red, circle_green, circle_blue, circle_alpha,
            img_red, img_green, img_blue,
            smt_i * SM_TH, smt_j * SM_TW
        );
    }
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
    if (height == 256 && width == 256) {
        gpu_level_render<32, 32, 2, 2><<<48, 8*32>>>(width, height, n_circle, circle_x, circle_y, circle_radius,
            circle_red, circle_green, circle_blue, circle_alpha,
            img_red, img_green, img_blue
        );
    } else if (height == 1024 && width == 1024) {
        gpu_level_render<128, 128, 8, 8><<<48, 8*32>>>(width, height, n_circle, circle_x, circle_y, circle_radius,
            circle_red, circle_green, circle_blue, circle_alpha,
            img_red, img_green, img_blue
        );
    } else {
        // Not handled
        return;
    }
}

} // namespace circles_gpu

// How do we use that a circle is a bounding box telling you which pixels it touches?

// Simple idea:
// (1) Divide image into SM tiles
// (2) Entire grid computes a scan of a boolean array similar to RLE for each tile over circles
//     (a) 1 indicates a circle is in the tile; 0 if not (use the boundary of the tile to check)
//     (b) Prefix sum indicates index in output of "runs"
//     (c) The "start" of a run is the circle that's actually in the tile
// (3) Now each tile has a list of only relevant circles
// (4) At this point could simply have each thread handle a pixel and iterate over its tile's circles
//     (a) Threads should handle pixels because then can accumulate results in registers and no need for atomic ops
//     (b) Threads should work on tiles so load data for some circles SMEM->reg then use it for all the pixels
//     (c) Tile's circles are in order but may not fit completely in SMEM, so need to buffer
// (5) Alt: recursively scan?

// Plan:
// (1) Use RLE from lab7 to get relevant circles for each SM tile
// (2) Simple SM-level impl: iterate over pixels and iterate over relevant circles
// (3) Improve SM-level by thread tiling
// (4) Improve SM-level by SMEM loading

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
