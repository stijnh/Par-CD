#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <sys/time.h>

#include "common.hpp"

using namespace std;

static vector<struct timeval> timer_vals;

void timer_start() {
    struct timeval val;
    gettimeofday(&val, NULL);
    timer_vals.push_back(val);
}

double timer_end() {
    if (timer_vals.empty()) {
        return 0.0;
    }

    struct timeval before = timer_vals.back();
    struct timeval after;
    gettimeofday(&after, NULL);

    timer_vals.pop_back();

    double diff = (after.tv_sec  - before.tv_sec) +
                  (after.tv_usec - before.tv_usec) / 1000000.0;

    return diff;
}

double timer() {
    struct timeval val;
    gettimeofday(&val, NULL);
    return val.tv_sec + val.tv_usec / 1000000.0;
}

Partition::Partition(int n) {
    labels = new int[n];
    fill(labels, labels + n, 0);
    num_labels = -1;
    sizes = NULL;
    int_volume = NULL;
    ext_volume = NULL;
}

Partition::Partition(int *l) {
    num_labels = -1;
    labels = l;
    sizes = NULL;
    int_volume = NULL;
    ext_volume = NULL;
}

Partition::~Partition() {
    if (labels) delete[](labels);
    if (sizes) delete[](sizes);
    if (int_volume) delete[](int_volume);
    if (ext_volume) delete[](ext_volume);
}

double norm_mutual_info(const Graph &g, map<pair<int, int>, int> &cm) {
    typedef map<pair<int, int>, int>::iterator iter_t;
    int ca = 0;
    int cb = 0;

    for (iter_t it = cm.begin(); it != cm.end(); it++) {
        int i = it->first.first;
        int j = it->first.second;

        ca = max(ca, i + 1);
        cb = max(cb, j + 1);
    }

    int *ni = new int[ca];
    int *nj = new int[cb];

    fill(ni, ni + ca, 0);
    fill(nj, nj + cb, 0);

    for (iter_t it = cm.begin(); it != cm.end(); it++) {
        int i = it->first.first;
        int j = it->first.second;
        int val = it->second;

        ni[i] += val;
        nj[j] += val;
    }

    double nom = 0.0;
    double den = 0.0;

    for (iter_t it = cm.begin(); it != cm.end(); it++) {
        int i = it->first.first;
        int j = it->first.second;
        int val = it->second;

        nom += val * log(((double) val * g.n) / (ni[i] * nj[j]));
    }

    for (int i = 0; i < ca; i++) {
        if (ni[i] != 0) {
            den += ni[i] * log(ni[i] / ((double) g.n));
        }
    }

    for (int j = 0; j < cb; j++) {
        if (nj[j] != 0) {
            den += nj[j] * log(nj[j] / ((double) g.n));
        }
    }

    return -2 * nom / den;
}

double compare_communities(const Graph &g, int *labels, string filename) {
    ifstream f(filename.c_str(), ios_base::in);

    if (!f.is_open()) {
        cerr << "failed to open " << filename << endl;
        exit(-1);
    }

    string line;
    map<pair<int, int>, int> cm;
    int comm = 0;

    while (getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        comm++;
        stringstream ss(line);
        int v;

        while (ss >> v) {
            cm[make_pair(comm, labels[v])]++;
        }
    }

    if (f.fail() && !f.eof()) {
        cerr << "error while reading " << filename << endl;
        exit(-1);
    }

    f.close();

    return norm_mutual_info(g, cm);
}

double compare_communities(const Graph &g, int *a, int *b) {
    map<pair<int, int>, int> n;

    for (int i = 0; i < g.n; i++) {
        n[make_pair(a[i], b[i])] += 1;
    }

    return norm_mutual_info(g, n);
}
