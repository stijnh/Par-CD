#include <algorithm>
#include <cassert>
#include <iostream>
#include <functional>
#include <map>
#include <memory>
#include <omp.h>
#include <vector>

#include "framework.hpp"

using namespace std;

void create_singleton_partition(const Graph &g, int *labels) {
    int n = g.n;

    for (int v = 0; v < n; v++) {
        labels[v] = v;
    }

    for (int v = 0; v < n; v++) {
        swap(labels[v], labels[rand() % n]);
    }
}


void create_clustering_partition(const Graph &g, int *labels) {
    int n = g.n;
    scoped_ptr<pair<pair<double, int>, int> > pairs(new pair<pair<double, int>, int>[n]);

    for (int v = 0; v < n; v++) {
        int tri = g.tri[v];
        int deg = g.degree(v);
        double cc = deg > 1 ? tri / double(deg * (deg - 1)) : 0.0;

        pairs[v] = make_pair(make_pair(-cc, -deg), v);
    }

    sort(pairs.get(), pairs.get() + n);

    int num_comms = 0;
    fill(labels, labels + n, NEW_LABEL);

    for (int i = 0; i < n; i++) {
        int v = pairs[i].second;

        if (labels[v] == NEW_LABEL) {
            labels[v] = num_comms;

            for (const int *p = g.begin_neighbors(v); p != g.end_neighbors(v); p++) {
                if (labels[*p] == NEW_LABEL) {
                    labels[*p] = num_comms;
                }
            }

            num_comms++;
        }
    }
}



void propagation_step(const Graph &g, const Partition &p, Rule &r, double frac, int *new_labels, int num_threads) {
}

void merging_step(const Graph &g, const Partition &p, Rule &r, int rounds, int *new_labels, int num_threads) {
}

INLINE static void collect_statistics_noatomic(const Graph &g, Partition &p) {
    const int n = g.n;
    const int *const labels = p.labels;
    const int k = *max_element(labels, labels + n) + 1;

    p.num_labels = k;
    p.sizes = new int[k];
    p.int_volume = new int[k];
    p.ext_volume = new int[k];

    fill(p.sizes, p.sizes + k, 0);
    fill(p.int_volume, p.int_volume + k, 0);
    fill(p.ext_volume, p.ext_volume + k, 0);

    for (int i = 0; i < n; i++) {
        int l = labels[i];
        int d_int = 0;
        int d_ext = 0;

        for (const int *u = g.begin_neighbors(i); u != g.end_neighbors(i); u++) {
            if (l == labels[*u]) {
                d_int++;
            } else {
                d_ext++;
            }
        }

        p.sizes[l]++;
        p.int_volume[l] += d_int;
        p.ext_volume[l] += d_ext;
    }
}


void collect_statistics(const Graph &g, Partition &p, int num_threads) {
    if (num_threads == 1) {
        return collect_statistics_noatomic(g, p);
    }

    const int n = g.n;
    const int *const labels = p.labels;
    const int k = *max_element(labels, labels + n) + 1;

    int *sizes = new int[k];
    int *int_volume = new int[k];
    int *ext_volume = new int[k];

    fill(sizes, sizes + k, 0);
    fill(int_volume, int_volume + k, 0);
    fill(ext_volume, ext_volume + k, 0);

#pragma omp parallel for schedule(static, 512) num_threads(num_threads)
    for (int i = 0; i < n; i++) {
        int l = labels[i];
        int d_int = 0;
        int d_ext = 0;

        for (const int *u = g.begin_neighbors(i); u != g.end_neighbors(i); u++) {
            if (l == labels[*u]) {
                d_int++;
            } else {
                d_ext++;
            }
        }

        __sync_fetch_and_add(&sizes[l], 1);
        __sync_fetch_and_add(&int_volume[l], d_int);
        __sync_fetch_and_add(&ext_volume[l], d_ext);
    }

    p.num_labels = k;
    p.sizes = sizes;
    p.int_volume = int_volume;
    p.ext_volume = ext_volume;
}


int compress_labels(const Graph &g, int *labels) {
    int n = g.n;
    int num_labels = *max_element(labels, labels + g.n) + 1;
    int num_new_labels = 0;
    scoped_ptr<int> relabel(new int[num_labels]);

    fill(relabel.get(), relabel.get() + num_labels, NEW_LABEL);

    for (int v = 0; v < n; v++) {
        int old_label = labels[v];
        int new_label;

        if (old_label == NEW_LABEL) {
            new_label = num_new_labels++;
        } else if (relabel[old_label] != NEW_LABEL) {
            new_label = relabel[old_label];
        } else {
            new_label = relabel[old_label] = num_new_labels++;
        }

        labels[v] = new_label;
    }

    return num_new_labels;
}
