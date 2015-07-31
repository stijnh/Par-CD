#ifndef FRAMEWORK_HPP
#define FRAMEWORK_HPP

#include "graph.hpp"
#include "common.hpp"
#include "rules.hpp"

void create_singleton_partition(const Graph &g, int *labels);
void create_clustering_partition(const Graph &g, int *labels);
int compress_labels(const Graph &g, int *labels);
void collect_statistics(const Graph &g, Partition &p, int num_threads=1);

#endif
