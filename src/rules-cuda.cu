#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <iostream>

#include "cuda-common.cuh"
#include "rules-cuda.cuh"
#include "cub/cub.cuh"

using namespace std;

__global__ void ker_modularity_score_calculate(
        const int k,
        const int m,
        const int *stats_int,
        const int *stats_ext,
        float *scores) {
    int i = get_global_id();
    if (i >= k) return;

    int int_vol = stats_int[i];
    int ext_vol = stats_ext[i];

    float actual = int_vol / float(2 * m);
    float expect = pow((int_vol + ext_vol) / float(2 * m), 2);

    scores[i] = actual - expect;
}


float ModularityCudaRule::calculate(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p) {
    const int c = p.num_labels;
    const int m = g.m;

    size_t temp_size;
    CUDA_CALL(cub::DeviceReduce::Sum,
            NULL,
            temp_size,
            (float*) NULL,
            (float*) NULL,
            c,
            stream);

    float *mem_scores;
    float *mem_result;
    void *mem_temp;

    CUDA_CALL(cudaMalloc, &mem_scores, sizeof(float) * c);
    CUDA_CALL(cudaMalloc, &mem_result, sizeof(float));
    CUDA_CALL(cudaMalloc, &mem_temp, temp_size);

    CUDA_LAUNCH(stream, c, GROUP_SIZE,
            ker_modularity_score_calculate,
            c,
            m,
            p.mem_stats_int,
            p.mem_stats_ext,
            mem_scores);

    CUDA_CALL(cub::DeviceReduce::Sum,
            mem_temp,
            temp_size,
            mem_scores,
            mem_result,
            c,
            stream);

    float result;
    CUDA_CALL(cudaMemcpy, &result, mem_result, sizeof(float), cudaMemcpyDeviceToHost);

    CUDA_CALL(cudaFree, mem_scores);
    CUDA_CALL(cudaFree, mem_result);
    CUDA_CALL(cudaFree, mem_temp);

    return result;
}

__global__ void ker_modularity_score_merge(
        const int k,
        const int m,
        const int2 *__restrict__ label_pairs,
        const int *__restrict__ counts,
        const int *__restrict__ stats_int,
        const int *__restrict__ stats_ext,
        float *__restrict__ scores) {
    int i = get_global_id();
    if (i >= k) return;

    int2 p = label_pairs[i];
    int a = p.x;
    int b = p.y;
    int f = counts[i];

    float e_ij = f / float(m);
    float a_i = (stats_int[a] + stats_ext[a]) / float(2 * m);
    float a_j = (stats_int[b] + stats_ext[b]) / float(2 * m);

    scores[i] = e_ij - 2 * a_i * a_j;
}

void ModularityCudaRule::score_merge(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p,
        const int num_pairs, const int2 *mem_label_pairs, const int *mem_counts, float *mem_scores) {
    CUDA_LAUNCH(stream, num_pairs, GROUP_SIZE,
            ker_modularity_score_merge,
            num_pairs,
            g.m,
            mem_label_pairs,
            mem_counts,
            p.mem_stats_int,
            p.mem_stats_ext,
            mem_scores);
}

__global__ void ker_modularity_score_join(
        const int k,
        const int m,
        const int *__restrict__ labels,
        const int *__restrict__ deg,
        const int *__restrict__ vertices,
        const int *__restrict__ candidate_labels,
        const int *__restrict__ counts,
        const int *__restrict__ stats_int,
        const int *__restrict__ stats_ext,
        float *scores) {
    int i = get_global_id();
    if (i >= k) return;

    int v = vertices[i];
    int c = candidate_labels[i];
    int f = counts[i];
    int d = deg[v];

    bool is_invalid = (c == INVALID_LABEL);
    if (is_invalid) c = labels[v];

    int vol = stats_ext[c] + stats_int[c];
    if (is_invalid) vol -= d;

    float e_ij = f / float(m);
    float a_i = vol / float(2 * m);
    float a_j = d / float(2 * m);

    scores[i] = e_ij - 2 * a_i * a_j;
}

void ModularityCudaRule::score_join(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p,
        const int num_pairs, const int *mem_vertices, const int *mem_labels, const int *mem_counts, float *mem_scores) {
    CUDA_LAUNCH(stream, num_pairs, GROUP_SIZE,
            ker_modularity_score_join,
            num_pairs,
            g.m,
            p.mem_labels,
            g.mem_deg,
            mem_vertices,
            mem_labels,
            mem_counts,
            p.mem_stats_int,
            p.mem_stats_ext,
            mem_scores);
}


__global__ void ker_cpm_score_calculate(
        const int k,
        const int m,
        const float lambda,
        const int *stats_int,
        const int *stats_sizes,
        float *scores) {
    int i = get_global_id();
    if (i >= k) return;

    int int_vol = stats_int[i];
    int size = stats_sizes[i];

    float actual = int_vol / float(2 * m);
    float expect = lambda * size * (size - 1) / float(2 * m);

    scores[i] = actual - expect;
}


float CPMCudaRule::calculate(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p) {
    const int c = p.num_labels;
    const int m = g.m;

    size_t temp_size;
    CUDA_CALL(cub::DeviceReduce::Sum,
            NULL,
            temp_size,
            (float*) NULL,
            (float*) NULL,
            c,
            stream);

    float *mem_scores;
    float *mem_result;
    void *mem_temp;

    CUDA_CALL(cudaMalloc, &mem_scores, sizeof(float) * c);
    CUDA_CALL(cudaMalloc, &mem_result, sizeof(float));
    CUDA_CALL(cudaMalloc, &mem_temp, temp_size);

    CUDA_LAUNCH(stream, c, GROUP_SIZE,
            ker_cpm_score_calculate,
            c,
            m,
            lambda,
            p.mem_stats_int,
            p.mem_stats_ext,
            mem_scores);

    CUDA_CALL(cub::DeviceReduce::Sum,
            mem_temp,
            temp_size,
            mem_scores,
            mem_result,
            c,
            stream);

    float result;
    CUDA_CALL(cudaMemcpy, &result, mem_result, sizeof(float), cudaMemcpyDeviceToHost);

    CUDA_CALL(cudaFree, mem_scores);
    CUDA_CALL(cudaFree, mem_result);
    CUDA_CALL(cudaFree, mem_temp);

    return result;
}

__global__ void ker_cpm_score_merge(
        const int k,
        const int m,
        const float lambda,
        const int2 *__restrict__ label_pairs,
        const int *__restrict__ counts,
        const int *__restrict__ stats_sizes,
        float *__restrict__ scores) {
    int i = get_global_id();
    if (i >= k) return;

    int2 p = label_pairs[i];
    int a = p.x;
    int b = p.y;
    int f = counts[i];
    int size_a = stats_sizes[a];
    int size_b = stats_sizes[b];

    scores[i] = f - lambda * size_a * size_b;
}

void CPMCudaRule::score_merge(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p,
        const int num_pairs, const int2 *mem_label_pairs, const int *mem_counts, float *mem_scores) {
    CUDA_LAUNCH(stream, num_pairs, GROUP_SIZE,
            ker_cpm_score_merge,
            num_pairs,
            g.m,
            lambda,
            mem_label_pairs,
            mem_counts,
            p.mem_stats_sizes,
            mem_scores);
}

__global__ void ker_cpm_score_join(
        const int k,
        const int m,
        const float lambda,
        const int *__restrict__ labels,
        const int *__restrict__ vertices,
        const int *__restrict__ candidate_labels,
        const int *__restrict__ counts,
        const int *__restrict__ stats_sizes,
        float *scores) {
    int i = get_global_id();
    if (i >= k) return;

    int v = vertices[i];
    int c = candidate_labels[i];
    int f = counts[i];

    bool is_invalid = (c == INVALID_LABEL);
    if (is_invalid) c = labels[v];

    int size = stats_sizes[c];
    if (is_invalid) size -= 1;

    scores[i] = f - lambda * size;
}

void CPMCudaRule::score_join(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p,
        const int num_pairs, const int *mem_vertices, const int *mem_labels, const int *mem_counts, float *mem_scores) {
    CUDA_LAUNCH(stream, num_pairs, GROUP_SIZE,
            ker_cpm_score_join,
            num_pairs,
            g.m,
            lambda,
            p.mem_labels,
            mem_vertices,
            mem_labels,
            mem_counts,
            p.mem_stats_sizes,
            mem_scores);
}
