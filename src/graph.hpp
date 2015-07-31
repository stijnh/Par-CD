#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <cstddef>

class Graph {
    public:
        const int n;
        const int m;
        const int *const indices;
        const int *const adj;
        const int *const tri;
        const int *const tri_deg;
        const int *const map;
        const double clustering_coef;

        Graph();
        Graph(int num_nodes, int num_edges, int *adj, int *indices,
                int *tri_deg, int *tri, int *map = NULL);
        ~Graph();

        static const Graph *read(const char *filename);
        void count_triangles(const int num_threads=1) const;

        inline const int *neighbors(int i) const {
            return adj + indices[i];
        }

        inline const int *begin_neighbors(int i) const {
            return adj + indices[i];
        }

        inline const int *end_neighbors(int i) const {
            return adj + indices[i + 1];
        }

        inline int degree(int i) const {
            return indices[i + 1] - indices[i];
        }
};


#endif
