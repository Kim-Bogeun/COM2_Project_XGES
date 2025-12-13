#include "BICScorerGPU_Optimized.h"
#include "set_ops.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <map>
#include <algorithm>

// Error checking macros
#define CUDA_CHECK_OPT(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK_OPT(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error in %s at line %d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// PImpl struct to hide CUDA implementation details
struct BICScorerGPUOptimizedImpl {
    cublasHandle_t cublas_handle;
    double* d_data = nullptr;
    double* d_covariance = nullptr;
    double* d_means = nullptr;
    double* d_centered_data = nullptr;

    long n_variables;
    long n_samples;
};

// =================================================================================
// CUDA Kernels for Optimized Version
// =================================================================================

// Helper kernels for data preprocessing (shared with V1)
static __global__ void column_means_kernel(const double* data, double* means, long n_samples, long n_variables) {
    int var_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (var_idx >= n_variables) return;

    double sum = 0.0;
    for (long i = 0; i < n_samples; ++i) {
        sum += data[i * n_variables + var_idx];
    }
    means[var_idx] = sum / n_samples;
}

static __global__ void center_data_kernel(const double* data, const double* means, double* centered_data, long n_samples, long n_variables) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_samples && col < n_variables) {
        long index = row * n_variables + col;
        centered_data[index] = data[index] - means[col];
    }
}

static __global__ void compute_covariance_kernel(const double* data, double* covariance, long n_samples, long n_variables) {
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

// Kernel to fill the d_A_pointers and d_B_pointers arrays
static __global__ void setup_batched_pointers(double** d_A_pointers, double* d_batch_A, int matrix_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        d_A_pointers[idx] = d_batch_A + idx * matrix_size;
    }
}

// Kernel to extract the sub-matrices for a batch of tasks with the same parent set size
// Debug kernel to check pointers
__global__ void debug_print_pointers(double** A_ptrs, double** b_ptrs, int* pivots, int p_size, int batch_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("GPU DEBUG: p_size=%d, batch_size=%d\n", p_size, batch_size);
        printf("GPU DEBUG: A_ptrs=%p, b_ptrs=%p\n", A_ptrs, b_ptrs);
        if (batch_size > 0) {
            printf("GPU DEBUG: A_ptrs[0]=%p\n", A_ptrs[0]);
            printf("GPU DEBUG: b_ptrs[0]=%p\n", b_ptrs[0]);
            // Attempt to read data
            // printf("GPU DEBUG: A[0][0]=%f\n", A_ptrs[0][0]); 
            // printf("GPU DEBUG: b[0][0]=%f\n", b_ptrs[0][0]);
        }
    }
}

static __global__ void extract_matrices_kernel(
    const double* d_covariance,
    const int* d_task_indices,
    const int* d_all_targets,
    const int* d_all_parent_sets_flat,
    const int* d_all_parent_set_offsets,
    double* d_batch_A,
    double* d_batch_b,
    int p_size,
    int num_tasks_in_batch,
    long n_variables)
{
    for (int task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < num_tasks_in_batch; task_idx += blockDim.x * gridDim.x) {
        int original_task_idx = d_task_indices[task_idx];
        int target = d_all_targets[original_task_idx];
        int p_offset = d_all_parent_set_offsets[original_task_idx];

        double* matrix_A = d_batch_A + task_idx * p_size * p_size;
        double* vector_b = d_batch_b + task_idx * p_size;

        for (int i = 0; i < p_size; ++i) {
            int parent_i = d_all_parent_sets_flat[p_offset + i];
            vector_b[i] = d_covariance[parent_i * n_variables + target];
            for (int j = 0; j < p_size; ++j) {
                int parent_j = d_all_parent_sets_flat[p_offset + j];
                matrix_A[i * p_size + j] = d_covariance[parent_i * n_variables + parent_j];
            }
        }
    }
}

#define MAX_P_SIZE 32

static __global__ void solve_batch_kernel(double* d_batch_A, double* d_batch_b, int p_size, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    double A[MAX_P_SIZE * MAX_P_SIZE];
    double b[MAX_P_SIZE];
    
    // Load A and b
    double* A_src = d_batch_A + idx * p_size * p_size;
    double* b_src = d_batch_b + idx * p_size;
    
    for (int i = 0; i < p_size * p_size; ++i) A[i] = A_src[i];
    for (int i = 0; i < p_size; ++i) b[i] = b_src[i];
    
    // Gaussian elimination with partial pivoting
    for (int i = 0; i < p_size; ++i) {
        // Pivot
        int pivot = i;
        double max_val = fabs(A[i * p_size + i]);
        for (int k = i + 1; k < p_size; ++k) {
            double val = fabs(A[k * p_size + i]);
            if (val > max_val) {
                max_val = val;
                pivot = k;
            }
        }
        
        // Swap rows
        if (pivot != i) {
            for (int j = i; j < p_size; ++j) {
                double temp = A[i * p_size + j];
                A[i * p_size + j] = A[pivot * p_size + j];
                A[pivot * p_size + j] = temp;
            }
            double temp_b = b[i];
            b[i] = b[pivot];
            b[pivot] = temp_b;
        }
        
        if (fabs(A[i * p_size + i]) < 1e-12) {
            continue; 
        }
        
        // Eliminate
        for (int k = i + 1; k < p_size; ++k) {
            double factor = A[k * p_size + i] / A[i * p_size + i];
            for (int j = i; j < p_size; ++j) {
                A[k * p_size + j] -= factor * A[i * p_size + j];
            }
            b[k] -= factor * b[i];
        }
    }
    
    // Back substitution
    for (int i = p_size - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < p_size; ++j) {
            sum += A[i * p_size + j] * b[j];
        }
        if (fabs(A[i * p_size + i]) > 1e-12) {
            b[i] = (b[i] - sum) / A[i * p_size + i];
        } else {
            b[i] = 0.0;
        }
    }
    
    // Store result
    for (int i = 0; i < p_size; ++i) b_src[i] = b[i];
}

static __global__ void debug_pivots(int* pivots, int p_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("GPU DEBUG: Pivots for first element: ");
        for(int i=0; i<p_size; ++i) printf("%d ", pivots[i]);
        printf("\n");
    }
}

static __global__ void calculate_score_p0_kernel(
    const double* d_covariance,
    const int* d_task_indices,
    const int* d_all_targets,
    double* d_out_scores,
    int num_tasks_in_batch,
    long n_variables,
    long n_samples,
    double alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tasks_in_batch) {
        int original_task_idx = d_task_indices[idx];
        int target = d_all_targets[original_task_idx];
        
        double sigma = d_covariance[target * n_variables + target];
        if (sigma <= 0) sigma = 1e-9;
        
        double log_likelihood_no_constant = -0.5 * n_samples * (1.0 + log(sigma));
        double bic_regularization = 0.5 * log((double)n_samples) * (1.0) * alpha;
        d_out_scores[original_task_idx] = log_likelihood_no_constant - bic_regularization;
    }
}

// Kernel to compute final scores after batched solve
static __global__ void calculate_final_scores_kernel(
    const double* d_covariance,
    const double* d_solved_betas, // This is d_batch_b after solving
    const int* d_task_indices,
    const int* d_all_targets,
    const int* d_all_parent_sets_flat,
    const int* d_all_parent_set_offsets,
    double* d_out_scores,
    int p_size,
    int num_tasks_in_batch,
    long n_variables,
    long n_samples,
    double alpha)
{
     for (int task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < num_tasks_in_batch; task_idx += blockDim.x * gridDim.x) {
        int original_task_idx = d_task_indices[task_idx];
        int target = d_all_targets[original_task_idx];
        int p_offset = d_all_parent_set_offsets[original_task_idx];

        double cov_target_target = d_covariance[target * n_variables + target];
        const double* beta = d_solved_betas + task_idx * p_size;
        
        double beta_dot_cov = 0.0;
        for(int i = 0; i < p_size; ++i) {
            int parent_i = d_all_parent_sets_flat[p_offset + i];
            double original_cov_val = d_covariance[parent_i * n_variables + target];
            beta_dot_cov += beta[i] * original_cov_val;
        }
        double sigma = cov_target_target - beta_dot_cov;

        if (sigma <= 0) sigma = 1e-9;
        double log_likelihood = -0.5 * n_samples * (1.0 + log(sigma));
        double regularization = 0.5 * log((double)n_samples) * (p_size + 1.0) * alpha;
        
        // Scatter result to the correct original position
        d_out_scores[original_task_idx] = log_likelihood - regularization;
    }
}


// =================================================================================
// BICScorerGPUOptimized Class Implementation
// =================================================================================

BICScorerGPUOptimized::BICScorerGPUOptimized(const Eigen::MatrixXd& host_data, double alpha) : alpha(alpha) {
    // std::cerr << "[DEBUG] BICScorerGPUOptimized Constructor Start" << std::endl;
    impl = new BICScorerGPUOptimizedImpl();
    n_samples = host_data.rows();
    n_variables = host_data.cols();
    impl->n_samples = n_samples;
    impl->n_variables = n_variables;

    CUBLAS_CHECK_OPT(cublasCreate(&impl->cublas_handle));

    size_t data_size = n_samples * n_variables * sizeof(double);
    size_t cov_size = n_variables * n_variables * sizeof(double);
    // std::cerr << "[DEBUG] Allocating GPU memory..." << std::endl;
    CUDA_CHECK_OPT(cudaMalloc(&impl->d_data, data_size));
    CUDA_CHECK_OPT(cudaMalloc(&impl->d_covariance, cov_size));
    CUDA_CHECK_OPT(cudaMalloc(&impl->d_means, n_variables * sizeof(double)));
    CUDA_CHECK_OPT(cudaMalloc(&impl->d_centered_data, data_size));

    // Copy data to device (convert from Eigen's column-major to row-major)
    // std::cerr << "[DEBUG] Copying data to GPU..." << std::endl;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> host_data_rowmajor = host_data;
    CUDA_CHECK_OPT(cudaMemcpy(impl->d_data, host_data_rowmajor.data(), data_size, cudaMemcpyHostToDevice));

    // --- Compute Covariance Matrix on GPU ---
    // std::cerr << "[DEBUG] Computing covariance..." << std::endl;
    // X_centered is row-major n_samples x n_variables
    // When interpreted as column-major by cuBLAS, it becomes n_variables x n_samples (transposed)
    dim3 threads_1d(256);
    dim3 blocks_1d((n_variables + threads_1d.x - 1) / threads_1d.x);
    column_means_kernel<<<blocks_1d, threads_1d>>>(impl->d_data, impl->d_means, n_samples, n_variables);

    dim3 threads_2d(16, 16);
    dim3 blocks_2d((n_variables + threads_2d.x - 1) / threads_2d.x, (n_samples + threads_2d.y - 1) / threads_2d.y);
    center_data_kernel<<<blocks_2d, threads_2d>>>(impl->d_data, impl->d_means, impl->d_centered_data, n_samples, n_variables);
    
    // --- Compute Covariance Matrix manually ---
    dim3 threads_cov(16, 16);
    dim3 blocks_cov(
        (n_variables + threads_cov.x - 1) / threads_cov.x,
        (n_variables + threads_cov.y - 1) / threads_cov.y
    );
    compute_covariance_kernel<<<blocks_cov, threads_cov>>>(impl->d_centered_data, impl->d_covariance, n_samples, n_variables);
    CUDA_CHECK_OPT(cudaGetLastError());
    CUDA_CHECK_OPT(cudaDeviceSynchronize());
    // std::cerr << "[DEBUG] BICScorerGPUOptimized Constructor End" << std::endl;
}

BICScorerGPUOptimized::~BICScorerGPUOptimized() {
    cudaFree(impl->d_data);
    cudaFree(impl->d_covariance);
    cudaFree(impl->d_means);
    cudaFree(impl->d_centered_data);
    cublasDestroy(impl->cublas_handle);
    delete impl;
}

double BICScorerGPUOptimized::local_score(int target, const FlatSet& parents) {
    std::vector<double> out_scores(1);
    std::vector<int> targets = {target};
    std::vector<FlatSet> parents_list = {parents};
    local_score_batched(out_scores, targets, parents_list);
    return out_scores[0];
}

void BICScorerGPUOptimized::local_score_batched(
    std::vector<double>& out_scores,
    const std::vector<int>& targets,
    const std::vector<FlatSet>& parents_list)
{
    // std::cerr << "[DEBUG] local_score_batched Start" << std::endl;
    int total_tasks = targets.size();
    if (total_tasks == 0) return;
    out_scores.resize(total_tasks);

    // Flatten the parents
    std::vector<int> parent_sets_flat;
    std::vector<int> parent_set_sizes;
    parent_set_sizes.reserve(total_tasks);
    
    size_t total_parents = 0;
    for(const auto& p : parents_list) total_parents += p.size();
    parent_sets_flat.reserve(total_parents);

    for(const auto& p : parents_list) {
        parent_set_sizes.push_back(p.size());
        for(int val : p) {
            parent_sets_flat.push_back(val);
        }
    }

    // --- Host-side setup ---
    std::vector<int> parent_set_offsets(total_tasks, 0);
    for(int i = 1; i < total_tasks; ++i) {
        parent_set_offsets[i] = parent_set_offsets[i-1] + parent_set_sizes[i-1];
    }
    
    // Group tasks by parent set size
    std::map<int, std::vector<int>> grouped_tasks;
    for(int i = 0; i < total_tasks; ++i) {
        grouped_tasks[parent_set_sizes[i]].push_back(i);
    }
    // std::cerr << "[DEBUG] Tasks grouped. Number of groups: " << grouped_tasks.size() << std::endl;

    // --- Device-side global allocations ---
    int *d_all_targets, *d_all_parent_sets_flat, *d_all_parent_set_offsets;
    double* d_out_scores;
    // std::cerr << "[DEBUG] Allocating global device memory..." << std::endl;
    CUDA_CHECK_OPT(cudaMalloc(&d_all_targets, targets.size() * sizeof(int)));
    CUDA_CHECK_OPT(cudaMalloc(&d_all_parent_sets_flat, parent_sets_flat.size() * sizeof(int)));
    CUDA_CHECK_OPT(cudaMalloc(&d_all_parent_set_offsets, parent_set_offsets.size() * sizeof(int)));
    CUDA_CHECK_OPT(cudaMalloc(&d_out_scores, total_tasks * sizeof(double)));

    // std::cerr << "[DEBUG] Copying global data to device..." << std::endl;
    CUDA_CHECK_OPT(cudaMemcpy(d_all_targets, targets.data(), targets.size() * sizeof(int), cudaMemcpyHostToDevice));
    // std::cerr << "[DEBUG] Copied targets." << std::endl;
    CUDA_CHECK_OPT(cudaMemcpy(d_all_parent_sets_flat, parent_sets_flat.data(), parent_sets_flat.size() * sizeof(int), cudaMemcpyHostToDevice));
    // std::cerr << "[DEBUG] Copied parent_sets_flat." << std::endl;
    CUDA_CHECK_OPT(cudaMemcpy(d_all_parent_set_offsets, parent_set_offsets.data(), parent_set_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    // std::cerr << "[DEBUG] Copied parent_set_offsets." << std::endl;


    // --- Process each group ---
    for(auto const& [p_size, task_indices] : grouped_tasks) {
        int batch_size = task_indices.size();
        // std::cerr << "[DEBUG] Processing group p_size=" << p_size << ", batch_size=" << batch_size << std::endl;
        
        // --- Device-side batch allocations ---
        int* d_task_indices;
        CUDA_CHECK_OPT(cudaMalloc(&d_task_indices, batch_size * sizeof(int)));
        CUDA_CHECK_OPT(cudaMemcpy(d_task_indices, task_indices.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice));

        if (p_size == 0) {
            dim3 threads(256);
            dim3 blocks((batch_size + threads.x - 1) / threads.x);
            calculate_score_p0_kernel<<<blocks, threads>>>(impl->d_covariance, d_task_indices, d_all_targets, d_out_scores, batch_size, n_variables, n_samples, alpha);
            cudaFree(d_task_indices);
            continue; 
        }
        
        double *d_batch_A, *d_batch_b;

        // std::cerr << "[DEBUG] Allocating batch memory for p_size=" << p_size << std::endl;
        CUDA_CHECK_OPT(cudaMalloc(&d_batch_A, (size_t)batch_size * p_size * p_size * sizeof(double)));
        CUDA_CHECK_OPT(cudaMalloc(&d_batch_b, (size_t)batch_size * p_size * sizeof(double)));
        
        dim3 threads(256);
        dim3 blocks((batch_size + threads.x - 1) / threads.x);

        // 1. Extract matrices into batch arrays
        // std::cerr << "[DEBUG] Extracting matrices..." << std::endl;
        extract_matrices_kernel<<<blocks, threads>>>(impl->d_covariance, d_task_indices, d_all_targets, d_all_parent_sets_flat,
                                                     d_all_parent_set_offsets, d_batch_A, d_batch_b, p_size, batch_size, n_variables);
        CUDA_CHECK_OPT(cudaGetLastError());
        
        // 2. Solve Ax=b using custom kernel
        solve_batch_kernel<<<blocks, threads>>>(d_batch_A, d_batch_b, p_size, batch_size);
        CUDA_CHECK_OPT(cudaGetLastError());
        CUDA_CHECK_OPT(cudaDeviceSynchronize());

        // 3. Calculate final scores from solved systems
        // std::cerr << "[DEBUG] Calculating final scores..." << std::endl;
        calculate_final_scores_kernel<<<blocks, threads>>>(impl->d_covariance, d_batch_b, d_task_indices, d_all_targets, 
                                                         d_all_parent_sets_flat, d_all_parent_set_offsets, d_out_scores,
                                                         p_size, batch_size, n_variables, n_samples, alpha);
        CUDA_CHECK_OPT(cudaGetLastError());
        
        // --- Free batch-specific memory ---
        // std::cerr << "[DEBUG] Freeing batch memory..." << std::endl;
        cudaFree(d_task_indices);
        cudaFree(d_batch_A);
        cudaFree(d_batch_b);
    }

    // --- Copy all results back to host ---
    // std::cerr << "[DEBUG] Copying results back to host..." << std::endl;
    CUDA_CHECK_OPT(cudaDeviceSynchronize());
    CUDA_CHECK_OPT(cudaMemcpy(out_scores.data(), d_out_scores, total_tasks * sizeof(double), cudaMemcpyDeviceToHost));

    // --- Free global device memory ---
    cudaFree(d_all_targets);
    cudaFree(d_all_parent_sets_flat);
    cudaFree(d_all_parent_set_offsets);
    cudaFree(d_out_scores);
    // std::cerr << "[DEBUG] local_score_batched End" << std::endl;
}
