#ifndef CUDA_RULES_H
#define CUDA_RULES_H

#include <cuda.h>

#define INVALID_LABEL (-1)

struct CudaGraph {
    int n;
    int m;
    int2 *mem_edges;
    int *mem_deg;
};

struct CudaPartition {
    int num_labels;
    int *mem_labels;
    int *mem_stats_sizes;
    int *mem_stats_ext;
    int *mem_stats_int;
};

class CudaRule {
    public:
        virtual float calculate(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p) = 0;
        virtual void score_merge(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p,
                const int num_pairs, const int2 *mem_label_pairs, const int *mem_counts, float *mem_scores) = 0;
        virtual void score_join(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p,
                const int num_pairs, const int *mem_vertices, const int *mem_labels, const int *mem_counts, float *mem_scores) = 0;
};


class ModularityCudaRule: public CudaRule {
    public:
        virtual float calculate(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p);
        virtual void score_merge(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p,
                const int num_pairs, const int2 *mem_label_pairs, const int *mem_counts, float *mem_scores);
        virtual void score_join(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p,
                const int num_pairs, const int *mem_vertices, const int *mem_labels, const int *mem_counts, float *mem_scores);
};


class CPMCudaRule: public CudaRule {
    private:
        double lambda;

    public:
        CPMCudaRule(double l) : lambda(l) {};
        virtual float calculate(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p);
        virtual void score_merge(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p,
                const int num_pairs, const int2 *mem_label_pairs, const int *mem_counts, float *mem_scores);
        virtual void score_join(cudaStream_t stream, const CudaGraph &g, const CudaPartition &p,
                const int num_pairs, const int *mem_vertices, const int *mem_labels, const int *mem_counts, float *mem_scores);
};


#endif
