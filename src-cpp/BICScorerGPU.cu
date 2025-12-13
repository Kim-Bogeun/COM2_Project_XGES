#include "BICScorerGPU.h"
#include "set_ops.h" // For FlatSet, though we mainly use its contents

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <numeric>

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error in %s at line %d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// PImpl struct to hide CUDA implementation details
struct BICScorerGPUImpl {
    cublasHandle_t cublas_handle;
    double* d_data = nullptr;
    double* d_covariance = nullptr;
    double* d_means = nullptr;
    double* d_centered_data = nullptr;

    // Reusable buffers for batched scoring
    int* d_targets = nullptr;
    int* d_parent_sets_flat = nullptr;
    int* d_parent_set_sizes = nullptr;
    int* d_parent_set_offsets = nullptr;
    double* d_out_scores = nullptr;

    size_t capacity_targets = 0;
    size_t capacity_parent_sets_flat = 0;
    size_t capacity_parent_set_sizes = 0;
    size_t capacity_parent_set_offsets = 0;
    size_t capacity_out_scores = 0;

    long n_variables;
    long n_samples;
};

// =================================================================================
// CUDA Kernels
// =================================================================================

__global__ void column_means_kernel(const double* data, double* means, long n_samples, long n_variables) {
    int var_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (var_idx >= n_variables) return;

    double sum = 0.0;
    for (long i = 0; i < n_samples; ++i) {
        sum += data[i * n_variables + var_idx];
    }
    means[var_idx] = sum / n_samples;
}

__global__ void center_data_kernel(const double* data, const double* means, double* centered_data, long n_samples, long n_variables) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_samples && col < n_variables) {
        long index = row * n_variables + col;
        centered_data[index] = data[index] - means[col];
    }
}

__global__ void compute_covariance_kernel(const double* data, double* covariance, long n_samples, long n_variables) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_variables && col < n_variables) {
        double sum = 0.0;
        for (long k = 0; k < n_samples; ++k) {
            sum += data[k * n_variables + row] * data[k * n_variables + col];
        }
        covariance[row * n_variables + col] = sum / n_samples;
    }
}


// A device-side function for a simple linear solver (Gaussian elimination)
// This is intentionally naive for Version 1. It will run entirely within a single thread.
// It solves Ax = b for a symmetric positive definite matrix A.
// A is stored in `matrix_A`, b in `vector_b`. Result `x` is stored in `vector_b`.
// All are pointers to small, thread-local arrays.
__device__ void solve_linear_system_naive(double* matrix_A, double* vector_b, int size) {
    if (size == 0) return;
    if (size > 32) { // Safety break for a naive implementation
        // printf("Parent set too large for naive solver\n"); 
        return;
    }

    // Forward elimination
    for (int k = 0; k < size; ++k) {
        // Find pivot
        int pivot = k;
        double max_val = fabs(matrix_A[k * size + k]);
        for (int i = k + 1; i < size; ++i) {
            if (fabs(matrix_A[i * size + k]) > max_val) {
                max_val = fabs(matrix_A[i * size + k]);
                pivot = i;
            }
        }
        // Swap rows if needed
        if (pivot != k) {
            for (int j = k; j < size; ++j) {
                double temp = matrix_A[k * size + j];
                matrix_A[k * size + j] = matrix_A[pivot * size + j];
                matrix_A[pivot * size + j] = temp;
            }
            double temp = vector_b[k];
            vector_b[k] = vector_b[pivot];
            vector_b[pivot] = temp;
        }

        // Elimination
        for (int i = k + 1; i < size; ++i) {
            double factor = matrix_A[i * size + k] / matrix_A[k * size + k];
            for (int j = k; j < size; ++j) {
                matrix_A[i * size + j] -= factor * matrix_A[k * size + j];
            }
            vector_b[i] -= factor * vector_b[k];
        }
    }

    // Back substitution
    for (int i = size - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < size; ++j) {
            sum += matrix_A[i * size + j] * vector_b[j];
        }
        vector_b[i] = (vector_b[i] - sum) / matrix_A[i * size + i];
    }
}


__global__ void batched_score_kernel_v1(
    const double* d_covariance,
    const int* d_targets,
    const int* d_parent_sets_flat,
    const int* d_parent_set_offsets, // Use offsets instead of sizes
    const int* d_parent_set_sizes,
    double* d_out_scores,
    long n_variables,
    long n_samples,
    double alpha,
    int num_tasks) 
{
    // Max parent set size this naive kernel can handle
    const int MAX_P_SIZE = 32;

    // Grid-stride loop
    for (int task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < num_tasks; task_idx += blockDim.x * gridDim.x) {
        int target = d_targets[task_idx];
        int p_size = d_parent_set_sizes[task_idx];
        int p_offset = d_parent_set_offsets[task_idx];

        double cov_target_target = d_covariance[target * n_variables + target];
        double sigma;

        if (p_size == 0) {
            sigma = cov_target_target;
        } else if (p_size > MAX_P_SIZE) {
            sigma = 1e9; // Indicate error/failure
        } else {
            // Thread-local storage for sub-matrices
            double cov_parents_parents[MAX_P_SIZE * MAX_P_SIZE];
            double cov_parents_target[MAX_P_SIZE];
            
            // Extract sub-matrices from global memory
            for (int i = 0; i < p_size; ++i) {
                int parent_i = d_parent_sets_flat[p_offset + i];
                // cov_parents_target
                cov_parents_target[i] = d_covariance[parent_i * n_variables + target];
                // cov_parents_parents
                for (int j = 0; j < p_size; ++j) {
                    int parent_j = d_parent_sets_flat[p_offset + j];
                    cov_parents_parents[i * p_size + j] = d_covariance[parent_i * n_variables + parent_j];
                }
            }

            // Solve for beta (beta is stored in cov_parents_target after this call)
            solve_linear_system_naive(cov_parents_parents, cov_parents_target, p_size);

            // Calculate sigma
            double beta_dot_cov = 0.0;
            for(int i = 0; i < p_size; ++i) {
                int parent_i = d_parent_sets_flat[p_offset + i];
                double original_cov_val = d_covariance[parent_i * n_variables + target];
                beta_dot_cov += cov_parents_target[i] * original_cov_val;
            }
            sigma = cov_target_target - beta_dot_cov;
        }
        
        // Final BIC score calculation
        if (sigma <= 0) sigma = 1e-9; // Avoid log(0)
        double log_likelihood_no_constant = -0.5 * n_samples * (1.0 + log(sigma));
        double bic_regularization = 0.5 * log((double)n_samples) * (p_size + 1.0) * alpha;
        d_out_scores[task_idx] = log_likelihood_no_constant - bic_regularization;
    }
}


// =================================================================================
// BICScorerGPU Class Implementation
// =================================================================================

BICScorerGPU::BICScorerGPU(const Eigen::MatrixXd& host_data, double alpha) : alpha(alpha) {
    impl = new BICScorerGPUImpl();
    n_samples = host_data.rows();
    n_variables = host_data.cols();
    impl->n_samples = n_samples;
    impl->n_variables = n_variables;

    CUBLAS_CHECK(cublasCreate(&impl->cublas_handle));

    // Allocate memory
    size_t data_size = n_samples * n_variables * sizeof(double);
    size_t cov_size = n_variables * n_variables * sizeof(double);
    CUDA_CHECK(cudaMalloc(&impl->d_data, data_size));
    CUDA_CHECK(cudaMalloc(&impl->d_covariance, cov_size));
    CUDA_CHECK(cudaMalloc(&impl->d_means, n_variables * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&impl->d_centered_data, data_size));

    // Copy data to device (convert from Eigen's column-major to row-major)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> host_data_rowmajor = host_data;
    CUDA_CHECK(cudaMemcpy(impl->d_data, host_data_rowmajor.data(), data_size, cudaMemcpyHostToDevice));

    // --- Compute Covariance Matrix on GPU ---
    // 1. Compute means
    dim3 threads_per_block_1d(256);
    dim3 num_blocks_1d((n_variables + threads_per_block_1d.x - 1) / threads_per_block_1d.x);
    column_means_kernel<<<num_blocks_1d, threads_per_block_1d>>>(impl->d_data, impl->d_means, n_samples, n_variables);
    CUDA_CHECK(cudaGetLastError());
    
    // 2. Center data
    dim3 threads_per_block_2d(16, 16);
    dim3 num_blocks_2d(
        (n_variables + threads_per_block_2d.x - 1) / threads_per_block_2d.x,
        (n_samples + threads_per_block_2d.y - 1) / threads_per_block_2d.y
    );
    center_data_kernel<<<num_blocks_2d, threads_per_block_2d>>>(impl->d_data, impl->d_means, impl->d_centered_data, n_samples, n_variables);
    CUDA_CHECK(cudaGetLastError());

    // 3. Compute covariance: S = (1/n) * X_centered^T * X_centered
    // We implement this manually to avoid cuBLAS layout confusion.
    dim3 threads_cov(16, 16);
    dim3 blocks_cov(
        (n_variables + threads_cov.x - 1) / threads_cov.x,
        (n_variables + threads_cov.y - 1) / threads_cov.y
    );
    compute_covariance_kernel<<<blocks_cov, threads_cov>>>(impl->d_centered_data, impl->d_covariance, n_samples, n_variables);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

BICScorerGPU::~BICScorerGPU() {
    // std::cout << "Destructing BICScorerGPU..." << std::endl;
    cudaFree(impl->d_data);
    cudaFree(impl->d_covariance);
    cudaFree(impl->d_means);
    cudaFree(impl->d_centered_data);

    // Free reusable buffers
    if (impl->d_targets) cudaFree(impl->d_targets);
    if (impl->d_parent_sets_flat) cudaFree(impl->d_parent_sets_flat);
    if (impl->d_parent_set_sizes) cudaFree(impl->d_parent_set_sizes);
    if (impl->d_parent_set_offsets) cudaFree(impl->d_parent_set_offsets);
    if (impl->d_out_scores) cudaFree(impl->d_out_scores);

    cublasDestroy(impl->cublas_handle);
    delete impl;
    // std::cout << "Destructed BICScorerGPU." << std::endl;
}

double BICScorerGPU::local_score(int target, const FlatSet& parents) {
    // Inefficient, runs a batch of 1. For ScorerInterface compatibility.
    std::vector<double> out_scores(1);
    std::vector<int> targets = {target};
    
    std::vector<int> parent_sets_flat;
    parent_sets_flat.reserve(parents.size());
    for(int p : parents) {
        parent_sets_flat.push_back(p);
    }
    std::vector<int> parent_set_sizes = {(int)parents.size()};

    local_score_batched_internal(out_scores, targets, parent_sets_flat, parent_set_sizes);
    
    return out_scores[0];
}

void BICScorerGPU::local_score_batched(
    std::vector<double>& out_scores,
    const std::vector<int>& targets,
    const std::vector<FlatSet>& parents_list) 
{
    std::vector<int> parent_sets_flat;
    std::vector<int> parent_set_sizes;
    parent_set_sizes.reserve(targets.size());
    
    size_t total_parents = 0;
    for(const auto& p : parents_list) total_parents += p.size();
    parent_sets_flat.reserve(total_parents);

    for(const auto& parents : parents_list) {
        parent_set_sizes.push_back((int)parents.size());
        for(int p : parents) {
            parent_sets_flat.push_back(p);
        }
    }

    local_score_batched_internal(out_scores, targets, parent_sets_flat, parent_set_sizes);
}

void BICScorerGPU::local_score_batched_internal(
    std::vector<double>& out_scores,
    const std::vector<int>& targets,
    const std::vector<int>& parent_sets_flat,
    const std::vector<int>& parent_set_sizes) 
{
    int num_tasks = targets.size();
    if (num_tasks == 0) return;

    out_scores.resize(num_tasks);

    // Create offsets vector
    std::vector<int> parent_set_offsets(num_tasks, 0);
    for(int i = 1; i < num_tasks; ++i) {
        parent_set_offsets[i] = parent_set_offsets[i-1] + parent_set_sizes[i-1];
    }

    // Manage memory for inputs and outputs using reusable buffers
    if (num_tasks > impl->capacity_targets) {
        if (impl->d_targets) cudaFree(impl->d_targets);
        impl->capacity_targets = std::max((size_t)num_tasks, (size_t)(impl->capacity_targets * 1.5));
        CUDA_CHECK(cudaMalloc(&impl->d_targets, impl->capacity_targets * sizeof(int)));
    }

    if (parent_sets_flat.size() > impl->capacity_parent_sets_flat) {
        if (impl->d_parent_sets_flat) cudaFree(impl->d_parent_sets_flat);
        impl->capacity_parent_sets_flat = std::max(parent_sets_flat.size(), (size_t)(impl->capacity_parent_sets_flat * 1.5));
        CUDA_CHECK(cudaMalloc(&impl->d_parent_sets_flat, impl->capacity_parent_sets_flat * sizeof(int)));
    }

    if (num_tasks > impl->capacity_parent_set_sizes) {
        if (impl->d_parent_set_sizes) cudaFree(impl->d_parent_set_sizes);
        impl->capacity_parent_set_sizes = std::max((size_t)num_tasks, (size_t)(impl->capacity_parent_set_sizes * 1.5));
        CUDA_CHECK(cudaMalloc(&impl->d_parent_set_sizes, impl->capacity_parent_set_sizes * sizeof(int)));
    }

    if (num_tasks > impl->capacity_parent_set_offsets) {
        if (impl->d_parent_set_offsets) cudaFree(impl->d_parent_set_offsets);
        impl->capacity_parent_set_offsets = std::max((size_t)num_tasks, (size_t)(impl->capacity_parent_set_offsets * 1.5));
        CUDA_CHECK(cudaMalloc(&impl->d_parent_set_offsets, impl->capacity_parent_set_offsets * sizeof(int)));
    }

    if (num_tasks > impl->capacity_out_scores) {
        if (impl->d_out_scores) cudaFree(impl->d_out_scores);
        impl->capacity_out_scores = std::max((size_t)num_tasks, (size_t)(impl->capacity_out_scores * 1.5));
        CUDA_CHECK(cudaMalloc(&impl->d_out_scores, impl->capacity_out_scores * sizeof(double)));
    }

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(impl->d_targets, targets.data(), num_tasks * sizeof(int), cudaMemcpyHostToDevice));
    if (!parent_sets_flat.empty()) {
        CUDA_CHECK(cudaMemcpy(impl->d_parent_sets_flat, parent_sets_flat.data(), parent_sets_flat.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(impl->d_parent_set_sizes, parent_set_sizes.data(), num_tasks * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl->d_parent_set_offsets, parent_set_offsets.data(), num_tasks * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threads(256);
    dim3 blocks((num_tasks + threads.x - 1) / threads.x);
    
    batched_score_kernel_v1<<<blocks, threads>>>(
        impl->d_covariance,
        impl->d_targets,
        impl->d_parent_sets_flat,
        impl->d_parent_set_offsets,
        impl->d_parent_set_sizes,
        impl->d_out_scores,
        n_variables,
        n_samples,
        alpha,
        num_tasks
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(out_scores.data(), impl->d_out_scores, num_tasks * sizeof(double), cudaMemcpyDeviceToHost));

    // No need to free here, we reuse the buffers

}
