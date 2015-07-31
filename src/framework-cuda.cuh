#ifndef FRAMEWORK_CUDA
#define FRAMEWORK_CUDA

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <curand.h>
#include <iostream>

#include "graph.hpp"
#include "rules-cuda.cuh"


bool cuda_init(int device_id, int group_size);
bool cuda_deinit(void);
bool cuda_transfer_graph(const Graph &g);
bool cuda_labels_to_device(int *labels);
bool cuda_labels_from_device(int *labels);

bool cuda_collect_statistics(void);
bool cuda_propagation_step(double frac, CudaRule &r);
bool cuda_merging_step(int rounds, CudaRule &r);
float cuda_calculate_metric(CudaRule &r);


#endif
