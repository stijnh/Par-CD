#include <algorithm>
#include <cmath>
#include <iostream>

#include "rules.hpp"

using namespace std;

double ModularityRule::calculate(const Graph &g, const Partition &p) const {
    const int c = p.num_labels;
    const int m = g.m;
    double mod = 0.0;

    for (int i = 0; i < c; i++) {
        double actual = p.int_volume[i] / (2.0 * m);
        double expect = pow((p.int_volume[i] + p.ext_volume[i]) / (2.0 * m), 2);

        mod += actual - expect;
    }

    return mod;
}

double ModularityRule::score_merge(const Graph &g, const Partition &p,
        int i, int j, int num_links) const {
    int m = g.m;
    double e_ij = num_links / double(m);
    double a_i = (p.int_volume[i] + p.ext_volume[i]) / (2.0 * m);
    double a_j = (p.int_volume[j] + p.ext_volume[j]) / (2.0 * m);

    return e_ij - 2 * a_i * a_j;
}

double ModularityRule::score_join(const Graph &g, const Partition &p,
        int v, int c, int num_neighbors) const {
    int m = g.m;

    double e_ij = num_neighbors / double(m);
    double a_i = (p.int_volume[c] + p.ext_volume[c]) / (2.0 * m);
    double a_j = g.degree(v) / (2.0 * m);

    return e_ij - 2 * a_i * a_j;
}

double ModularityRule::score_leave(const Graph &g, const Partition &p,
        int v, int num_neighbors) const {
    int m = g.m;
    int c = p.labels[v];
    int deg = g.degree(v);

    double e_ij = num_neighbors / double(m);
    double a_i = (p.int_volume[c] + p.ext_volume[c] - deg) / (2.0 * m);
    double a_j = g.degree(v) / (2.0 * m);

    return -(e_ij - 2 * a_i * a_j);
}

double ConductanceRule::calculate(const Graph &g, const Partition &p) const {
    double score = 0.0;

    for (int c = 0; c < p.num_labels; c++) {
        int s = p.sizes[c];
        int i = p.int_volume[c];
        int v = i + p.ext_volume[c];

        score += s * (i / double(v));
    }

    return score / g.n;
}

string ModularityRule::get_name() const {
    return "modularity";
}

double ConductanceRule::score_merge(const Graph &g, const Partition &p,
        int i, int j, int num_links) const {
    int si = p.sizes[i];
    int sj = p.sizes[j];

    int ii = p.int_volume[i];
    int ij = p.int_volume[j];

    int vi = ii + p.ext_volume[i];
    int vj = ij + p.ext_volume[j];


    return (si + sj) * (ii + ij + num_links) / double(vi + vj + num_links)
        - si * ii / double(vi)
        - sj * ij / double(vj);
}

double ConductanceRule::score_join(const Graph &g, const Partition &p,
        int x, int c, int num_neighbors) const {
    int deg = g.degree(x);
    int s = p.sizes[c];
    int i = p.int_volume[c];
    int v = i + p.ext_volume[c];

    return (s + 1) * (i + 2 * num_neighbors) / double(v + deg)
        - s * (i / double(v));
}

double ConductanceRule::score_leave(const Graph &g, const Partition &p,
        int x, int num_neighbors) const {
    int deg = g.degree(x);
    int c = p.labels[x];
    int s = p.sizes[c];
    int i = p.int_volume[c];
    int v = i + p.ext_volume[c];

    return (s - 1) * (i - 2 * num_neighbors) / double(v - deg)
        - s * (i / double(v));
}

string ConductanceRule::get_name() const {
    return "conductance";
}


double CPMRule::calculate(const Graph &g, const Partition &p) const {
    double score = 0.0;

    for (int i = 0; i < p.num_labels; i++) {
        score += p.int_volume[i] - lambda * p.sizes[i] * (p.sizes[i] - 1);
    }

    return score / double(2 * g.m);
}

double CPMRule::score_merge(const Graph &g, const Partition &p,
        int i, int j, int num_links) const {
    return num_links - lambda * p.sizes[i] * p.sizes[j];
}

double CPMRule::score_join(const Graph &g, const Partition &p,
        int v, int c, int num_neighbors) const {
    return num_neighbors - lambda * p.sizes[c];
}

double CPMRule::score_leave(const Graph &g, const Partition &p,
        int v, int num_neighbors) const {
    return -(num_neighbors - lambda * (p.sizes[p.labels[v]] - 1));
}

string CPMRule::get_name() const {
    return "CPM (lambda=" + to_string(lambda) + ")";
}

double WCCRule::calculate(const Graph &g, const Partition &p) const {
    double score = 0.0;
    const int *labels = p.labels;
    int *eit = new int[2 * g.m];
    int *it = new int[g.n];
    int *itd = new int[g.n];

    fill(eit, eit + (2 * g.m), 0);
    fill(it, it + g.n, 0);
    fill(itd, itd + g.n, 0);

#pragma omp parallel for schedule(dynamic, 128)
    for (int v = 0; v < g.n; v++) {
        int l = labels[v];

        for (const int *r = g.begin_neighbors(v); r != g.end_neighbors(v); r++) {
            int u = *r;

            if (u >= v) {
                break;
            }

            if (l != labels[u]) {
                continue;
            }

            const int *x = g.begin_neighbors(v);
            const int *y = g.begin_neighbors(u);
            const int *x_end = g.end_neighbors(v);
            const int *y_end = g.end_neighbors(u);

            while (x != x_end && *x < u && y != y_end && *y < u) {
                int d = *x - *y;


                if (d == 0 && labels[*x] == l) {
                    __sync_fetch_and_add(eit + (x - g.adj), 1);
                    __sync_fetch_and_add(eit + (y - g.adj), 1);
                    __sync_fetch_and_add(eit + (r - g.adj), 1);
                }

                if (d <= 0) x++;
                if (d >= 0) y++;
            }
        }
    }

#pragma omp parallel for schedule(dynamic, 128)
    for (int v = 0; v < g.n; v++) {
        for (const int *r = g.begin_neighbors(v); r != g.end_neighbors(v); r++) {
            int u = *r;

            const int int_tri = eit[r - g.adj];

            if (int_tri > 0) {
                __sync_fetch_and_add(it + v, int_tri);
                __sync_fetch_and_add(itd + v, 1);
                __sync_fetch_and_add(it + u, int_tri);
                __sync_fetch_and_add(itd + u, 1);
            }
        }
    }

#pragma omp parallel for reduction(+:score)
    for (int v = 0; v < g.n; v++) {
        int size = p.sizes[p.labels[v]];

        int tri = g.tri[v];
        int dtri = g.tri_deg[v];
        int itri = it[v];
        int ditri = itd[v];

        if (tri > 0) {
            double a = itri / double(tri);
            double b = dtri / double(size - 1 + dtri - ditri);

            score += (a * b) / g.n;
        }
    }

    delete[] eit;
    delete[] it;
    delete[] itd;

    return score;
}




inline static double calculate_wcc_improvement(int n_x, int n_y, int c_x, int c_y, int m_x, int m_y, int m_xy, int b_x, int b_y, double cc, bool debug=false) {
    double p_x = n_x > 1 ? m_x / (n_x * (n_x - 1.0) / 2.0) : 0.0;
    double p_y = n_y > 1 ? m_y / (n_y * (n_y - 1.0) / 2.0) : 0.0;
    double p_xy = c_x > 0 && c_y > 0 ? m_xy / double(c_x * c_y) : 0.0;

    double q_x = n_x > 0 ? (b_x - m_xy) / double(n_x) : 0;
    double q_y = n_y > 0 ? (b_y - m_xy) / double(n_y) : 0;

    double score = 0.0;

    debug = n_x > 1 && n_y > 1 && 0;
    if (debug) {
        cout << "n_x=" << n_x << endl;
        cout << "n_y=" << n_y << endl;
        cout << "c_x=" << c_x << endl;
        cout << "c_y=" << c_y << endl;
        cout << "b_x=" << b_x << endl;
        cout << "b_y=" << b_y << endl;
        cout << "m_x=" << m_x << endl;
        cout << "m_y=" << m_y << endl;
        cout << "m_xy=" << m_xy << endl;
    }

    // nodes in X connected to Y
    if (c_x > 0) {
        double t_x  = (n_x - 1) * (n_x - 2) / 2 * p_x * p_x * p_x; // v - X - X
        double t_xy = t_x                                          // v - X - X
                    + (c_x - 1) * c_y * p_x * p_y * p_xy           // v - X - Y
                    + c_y * (c_y - 1) / 2 * p_xy * p_xy * p_y;     // v - Y - Y
        double t_v  = t_xy
                    + q_x * (q_x - 1) / 2 * cc                     // v - V - V
                    + q_x * (n_x - 1) * p_x * cc                   // v - X - V
                    + c_y * q_x * p_xy * cc;                       // v - Y - V

        double vt_xy = q_x;
        double vt_x  = vt_xy + p_xy * c_y;
        double vt_v  = vt_x + p_x * (n_x - 1);

        double tr = (vt_v / t_v) * (t_xy / (n_x + n_y - 1 + vt_xy))
                  - (vt_v / t_v) * (t_x / (n_x - 1 + vt_x));

        if (t_v > 0) score += c_x * tr;

        if (debug) {
            cout << "x in X, neighbor in Y" << endl;
            cout << " t_x=" << t_x << endl;
            cout << " t_xy=" << t_xy << endl;
            cout << " t_v=" << t_v << endl;
            cout << " vt_xy=" << vt_xy << endl;
            cout << " vt_x=" << vt_x << endl;
            cout << " vt_v=" << vt_v << endl;
            cout << " tr=" << tr << endl;
        }
    }

    // nodes in X not connected to Y
    if (c_x < n_x) {
        double t_x  = (n_x - 1) * (n_x - 2) / 2 * p_x * p_x * p_x; // v - X - X
        double t_xy = t_x;
        double t_v  = t_xy
                    + q_x * (q_x - 1) / 2 * cc                     // v - V - V
                    + q_x * (n_x - 1) * p_x * cc;                  // v - X - V

        double vt_xy = q_x;
        double vt_x  = vt_xy;
        double vt_v  = vt_x + p_x * (n_x - 1);

        double tr = (vt_v / t_v) * (t_xy / (n_x + n_y - 1 + vt_xy))
                  - (vt_v / t_v) * (t_x / (n_x - 1 + vt_x));

        if (t_v > 0) score += (n_x - c_x) * tr;

        if (debug) {
            cout << "x in X, not neighbor in Y" << endl;
            cout << " t_x=" << t_x << endl;
            cout << " t_xy=" << t_xy << endl;
            cout << " t_v=" << t_v << endl;
            cout << " vt_xy=" << vt_xy << endl;
            cout << " vt_x=" << vt_x << endl;
            cout << " vt_v=" << vt_v << endl;
            cout << " tr=" << tr << endl;
        }
    }

    // nodes in Y connected to X
    if (c_y > 0) {
        double t_y  = (n_y - 1) * (n_y - 2) / 2 * p_y * p_y * p_y; // v - Y - Y
        double t_xy = t_y                                          // v - Y - Y
                    + (c_y - 1) * c_x * p_y * p_x * p_xy           // v - Y - X
                    + c_x * (c_x - 1) / 2 * p_xy * p_xy * p_x;     // v - X - X
        double t_v  = t_xy
                    + q_y * (q_y - 1) / 2 * cc                     // v - V - V
                    + q_y * (n_y - 1) * p_y * cc                   // v - Y - V
                    + c_x * q_y * p_xy * cc;                       // v - X - V

        double vt_xy = q_y;
        double vt_y  = vt_xy + p_xy * c_x;
        double vt_v  = vt_y + p_y * (n_y - 1);

        double tr = (vt_v / t_v) * (t_xy / (n_y + n_x - 1 + vt_xy))
                  - (vt_v / t_v) * (t_y / (n_y - 1 + vt_y));

        if (t_v > 0) score += c_y * tr;

        if (debug) {
            cout << "x in Y, neighbor in Y" << endl;
            cout << " t_y=" << t_y << endl;
            cout << " t_xy=" << t_xy << endl;
            cout << " t_v=" << t_v << endl;
            cout << " vt_xy=" << vt_xy << endl;
            cout << " vt_y=" << vt_y << endl;
            cout << " vt_v=" << vt_v << endl;
            cout << " tr=" << tr << endl;
        }
    }

    // nodes in Y not connected to X
    if (c_y < n_y) {
        double t_y  = (n_y - 1) * (n_y - 2) / 2 * p_y * p_y * p_y; // v - Y - Y
        double t_xy = t_y;                                         // v - Y - Y
        double t_v  = t_xy
                    + q_y * (q_y - 1) / 2 * cc                     // v - V - V
                    + q_y * (n_y - 1) * p_y * cc;                  // v - Y - V

        double vt_xy = q_y;
        double vt_y  = vt_xy;
        double vt_v  = vt_y + p_y * (n_y - 1);

        double tr = (vt_v / t_v) * (t_xy / (n_y + n_x - 1 + vt_xy))
                  - (vt_v / t_v) * (t_y / (n_y - 1 + vt_y));

        if (t_v > 0) score += (n_y - c_y) * tr;

        if (debug) {
            cout << "x in Y, not neighbor in Y" << endl;
            cout << " t_y=" << t_y << endl;
            cout << " t_xy=" << t_xy << endl;
            cout << " t_v=" << t_v << endl;
            cout << " vt_xy=" << vt_xy << endl;
            cout << " vt_y=" << vt_y << endl;
            cout << " vt_v=" << vt_v << endl;
            cout << " tr=" << tr << endl;
        }
    }

    if (debug)
        cout << "score=" << score << endl;

    return score;
}

double WCCRule::score_merge(const Graph &g, const Partition &p,
        int i, int j, int num_links) const {

    return calculate_wcc_improvement(
        p.sizes[i], p.sizes[j],
        min(num_links, p.sizes[i]), min(num_links, p.sizes[j]),
        p.int_volume[i] / 2, p.int_volume[j] / 2,
        num_links,
        p.ext_volume[i], p.ext_volume[j],
        g.clustering_coef);
}

double WCCRule::score_join(const Graph &g, const Partition &p,
        int v, int c, int num_neighbors) const {
    int deg = g.degree(v);
    int size = p.sizes[c];
    int int_edge = p.int_volume[c] / 2;
    int ext_edge = p.ext_volume[c];

   return calculate_wcc_improvement(
            1, size,
            1, num_neighbors,
            0, int_edge,
            num_neighbors,
            deg, ext_edge,
            g.clustering_coef);
}

double WCCRule::score_leave(const Graph &g, const Partition &p,
        int v, int num_neighbors) const {

    int deg = g.degree(v);
    int c = p.labels[v];
    int size = p.sizes[c];
    int int_edge = p.int_volume[c] / 2;
    int ext_edge = p.ext_volume[c];


    return -calculate_wcc_improvement(
            1, size - 1,
            1, num_neighbors,
            0, int_edge - num_neighbors,
            num_neighbors,
            deg, ext_edge - deg + 2 * num_neighbors,
            g.clustering_coef);
}

string WCCRule::get_name() const {
    return "WCC";
}

