#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include "graph.hpp"

#define MAGIC_WORD ("PCD")

using namespace std;

Graph::Graph() :
    n(0),
    m(0),
    indices(NULL),
    adj(NULL),
    tri(NULL),
    tri_deg(NULL),
    map(NULL),
    clustering_coef(0.0) {
}


Graph::Graph(int num_nodes, int num_edges, int *edge_adj, int *edge_indices,
        int *triangles_deg, int *triangles, int *mp) :
    n(num_nodes),
    m(num_edges),
    indices(edge_indices),
    adj(edge_adj),
    tri(triangles),
    tri_deg(triangles_deg),
    map(mp),
    clustering_coef(NAN) {
        //
}


const Graph *Graph::read(const char *filename) {
    int n, m;
    int *adj, *indices;
    //int *tri, *tri_deg;

    ifstream f(filename, ios_base::binary | ios_base::in);

    if (!f.is_open()) {
        cerr << "failed to open " << filename << endl;
        return NULL;
    }

    char word[16];
    f.read((char*) word, sizeof(char) * strlen(MAGIC_WORD));

    if (!f.good()) {
        cerr << "error while reading " << filename << endl;
        return NULL;
    }

    if (strncmp(word, MAGIC_WORD, strlen(MAGIC_WORD)) != 0) {
        cerr << "invalid file type" << endl;
        return NULL;
    }

    f.read((char*) &n, sizeof(int));
    f.read((char*) &m, sizeof(int));

    if (!f.good()) {
        cerr << "error while reading " << filename << endl;
        return NULL;
    }

    indices = new int[n + 1];
    f.read((char*) indices, sizeof(int) * (n + 1));

    adj = new int[2 * m];
    f.read((char*) adj, sizeof(int) * (2 * m));

    //tri = NULL;
    //f.read((char*) tri, sizeof(int) * n);
    //f.ignore(sizeof(int) * n);

    //tri_deg = NULL;
    //f.read((char*) tri_deg, sizeof(int) * n);
    //f.ignore(sizeof(int) * n);

    int *map = new int[n];
    f.read((char*) map, sizeof(int) * n);

    bool err = !f.good() || f.eof();
    f.ignore(sizeof(char));
    err |= !f.eof();
    f.close();

    if (err) {
        delete[] adj;
        delete[] indices;
        delete[] map;

        cerr << "error while reading " << filename << endl;
        return NULL;
    }

    return new Graph(n, m, adj, indices, NULL, NULL, map);
}

void Graph::count_triangles(const int num_threads) const {
    double cc = 0;
    int *et = new int[2 * m];
    int *t = new int[n];
    int *td = new int[n];

    fill(et, et + (2 * m), 0);
    fill(t, t + n, 0);
    fill(td, td + n, 0);

#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 128)
    for (int v = 0; v < n; v++) {
        for (const int *r = begin_neighbors(v); r != end_neighbors(v); r++) {
            int u = *r;

            if (u >= v) {
                break;
            }

            const int *x = begin_neighbors(v);
            const int *y = begin_neighbors(u);
            const int *x_end = end_neighbors(v);
            const int *y_end = end_neighbors(u);

            while (x != x_end && y != y_end && *x < u && *y < u) {
                int d = *x - *y;

                if (d == 0) {
                    __sync_fetch_and_add(et + (x - adj), 1);
                    __sync_fetch_and_add(et + (y - adj), 1);
                    __sync_fetch_and_add(et + (r - adj), 1);
                }

                if (d <= 0) x++;
                if (d >= 0) y++;
            }
        }
    }

#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 128)
    for (int v = 0; v < n; v++) {
        for (const int *r = begin_neighbors(v); r != end_neighbors(v); r++) {
            int u = *r;
            int tri = et[r - adj];

            if (tri > 0) {
                __sync_fetch_and_add(t + v, tri);
                __sync_fetch_and_add(t + u, tri);
                __sync_fetch_and_add(td + v, 1);
                __sync_fetch_and_add(td + u, 1);
            }
        }
    }

#pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+:cc)
    for (int v = 0; v < n; v++) {
        int tri = t[v];
        int deg = degree(v);

        if (deg > 1) {
            cc += (tri / (deg * (deg - 1.0))) / n;
        }
    }

    delete[] et;
    *const_cast<int**>(&tri) = t;
    *const_cast<int**>(&tri_deg) = td;
    *const_cast<double*>(&clustering_coef) = cc;
}

Graph::~Graph() {
    delete[] adj;
    delete[] indices;
    if (tri)     delete[] tri;
    if (tri_deg) delete[] tri_deg;
}
