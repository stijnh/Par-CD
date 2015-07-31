#include <vector>
#include <cstdlib>

#include "refinement.hpp"
#include "graph.hpp"
#include "rules.hpp"

using namespace std;

INLINE static int ceil_power_of_two(int a) {
    //unsigned int v = (unsigned int)a;
    //v--;
    //v |= v >> 1;
    //v |= v >> 2;
    //v |= v >> 4;
    //v |= v >> 8;
    //v |= v >> 16;
    //v++;
    //return (int) v;
    return 1 << (32 - __builtin_clz(a - 1));
}


class FastHashTable {
    public:
        static const int load_factor = 2;
        static const int init_cap = 4096;
        int capacity;
        int size;
        int max_size;
        int mask;
        vector<pair<int, int> > entries;
        vector<int> indices;

        FastHashTable(int cap) {
            capacity = init_cap;
            size = 0;
            max_size = init_cap;
            mask = max_size - 1;

            indices.resize(capacity);
            entries.resize(capacity, make_pair(-1, 0));
        }

        INLINE void clear() {
            for (int i = 0; i < size; i++) {
                entries[indices[i]] = make_pair(-1, 0);
            }

            size = 0;
            max_size = init_cap;
            mask = max_size - 1;
        }

        INLINE void insert(int key, int value=1) {
            int index = key & mask;
            int curr_key = entries[index].first;

            while(1) {
                if (curr_key == -1) {
                    entries[index] = make_pair(key, value);
                    indices[size++] = index;
                    break;
                } else if (curr_key == key) {
                    entries[index].second += value;
                    break;
                }

                index = (index + 1) & mask;
                curr_key = entries[index].first;
            }

            if (size * load_factor > max_size) {
                resize();
            }
        }

        INLINE void resize() {
            max_size *= 2;
            mask = max_size - 1;
            bool changes = true;

            if (max_size > capacity) {
                entries.resize(2 * capacity, make_pair(-1, 0));
                indices.resize(2 * capacity);
                capacity *= 2;
            }

            while (changes) {
                changes = false;

                for (int i = 0; i < size; i++) {
                    int old_index = indices[i];
                    int key = entries[old_index].first;
                    int new_index = key & mask;

                    while (new_index != old_index && entries[new_index].first != -1) {
                        new_index = (new_index + 1) & mask;
                    }

                    if (new_index != old_index) {
                        indices[i] = new_index;
                        swap(entries[old_index], entries[new_index]);
                        changes = true;
                    }

                }
            }
        }
};



void PropagationRefinement::refine(const Graph &g, const Partition &p, const Rule &r, int *new_labels) {
    const int n = g.n;
    const int f = min(frac, 1.0) * RAND_MAX;
    const int *const labels = p.labels;

#pragma omp parallel num_threads(num_threads)
    {
        unsigned int seed;

#pragma omp critical
        seed = rand();
        FastHashTable htable(g.degree(0));

#pragma omp for schedule(static, 512)
        for (int v = 0; v < n; v++) {
            if (rand_r(&seed) > f) {
                new_labels[v] = p.labels[v];
                continue;
            }

            for (const int *u = g.begin_neighbors(v); u != g.end_neighbors(v); u++) {
                htable.insert(labels[*u]);
            }

            int curr_label = p.labels[v];
            int best_label = -1;

            int best_count = 0;
            double best_score = 0.0;
            double leave_score = 0.0;

            int curr_freq = 0;

            for (int i = 0; i < htable.size; i++) {
                int label = htable.entries[htable.indices[i]].first;
                int freq = htable.entries[htable.indices[i]].second;

                if (label != curr_label) {
                    double score = r.score_join(g, p, v, label, freq);

                    if (score > best_score) {
                        best_label = label;
                        best_score = score;
                        best_count = 1;
                    } else if (score == best_score) {
                        if (rand_r(&seed) % (++best_count) == 0) {
                            best_label = label;
                        }
                    }
                } else {
                    curr_freq = freq;
                }
            }

            htable.clear();
            leave_score = r.score_leave(g, p, v, curr_freq);

            best_score += leave_score;

            if (best_score > 0 && best_score > leave_score) {
                new_labels[v] = best_label;
            } else if (leave_score > 0) {
                new_labels[v] = NEW_LABEL;
            } else {
                new_labels[v] = curr_label;
            }
        }
    }
}


void MergingRefinement::refine(const Graph &g, const Partition &p, const Rule &r, int *new_labels) {
    const int n = g.n;
    const int num_comms = p.num_labels;
    const int *const labels = p.labels;

    scoped_ptr<int> comm_members(n);
    scoped_ptr<int> comm_indices(num_comms + 1);

    comm_indices[0] = 0;
    comm_indices[1] = 0;

    for (int i = 2; i < num_comms + 1; i++) {
        comm_indices[i] = comm_indices[i - 1] + p.sizes[i - 2];
    }

    for (int v = 0; v < n; v++) {
        comm_members[comm_indices[p.labels[v] + 1]++] = v;
    }

    scoped_ptr<int> best_partner(num_comms);
    scoped_ptr<int> found_partner(num_comms);

    fill(best_partner.get(), best_partner.get() + num_comms, -1);
    fill(found_partner.get(), found_partner.get() + num_comms, false);

    bool all_done = false;

#pragma omp parallel num_threads(num_threads)
    {
        FastHashTable htable(num_comms);
        unsigned int seed;

#pragma omp critical
        seed = rand();

        for (int rnd = 0; rnd < rounds && !all_done; rnd++) {

#pragma omp for schedule(dynamic, 256)
            for (int l = 0; l < num_comms; l++) {
                if (found_partner[l] || (best_partner[l] != -1 && !found_partner[best_partner[l]])) {
                    continue;
                }

                for (int i = comm_indices[l]; i != comm_indices[l + 1]; i++) {
                    int v = comm_members[i];

                    for (const int *u = g.begin_neighbors(v); u != g.end_neighbors(v); u++) {
                        int k = labels[*u];

                        if (k != l && !found_partner[k]) {
                            htable.insert(k);
                        }
                    }
                }

                int best_label = -1;
                double best_score = 0;
                int count = 1;

                for (int i = 0; i < htable.size; i++) {
                    int label = htable.entries[htable.indices[i]].first;
                    int freq = htable.entries[htable.indices[i]].second;
                    double score = r.score_merge(g, p, l, label, freq);

                    if (score == best_score) {
                        if (rand_r(&seed) % (count++) == 0) {
                            best_label = label;
                        }
                    } else if (score > best_score) {
                        best_label = label;
                        best_score = score;
                        count = 1;
                    }
                }

                best_partner[l] = best_label;
                htable.clear();
            }

#pragma omp single
            all_done = true;

#pragma omp for schedule(static, 512) reduction(&&:all_done)
            for (int l = 0; l < num_comms; l++) {
                if (!found_partner[l] && (best_partner[l] == -1 || best_partner[best_partner[l]] == l)) {
                    found_partner[l] = true;
                }

                all_done &= found_partner[l];
            }
        }

#pragma omp for schedule(static, 512)
        for (int v = 0; v < n; v++) {
            int l = labels[v];

            if (found_partner[l] && best_partner[l] != -1) {
                new_labels[v] = min(l, best_partner[l]);
            } else {
                new_labels[v] = l;
            }
        }
    }
}


void FindConnectedRefinement::refine(const Graph &g, const Partition &p, const Rule &r, int *new_labels) {
    const int n = g.n;

    scoped_ptr<int> tags(n);
    scoped_ptr<int> new_tags(n);

    for (int i = 0; i < n; i++) {
        tags[i] = i;
    }

    bool done = false;

    while (!done) {
        done = true;

#pragma omp parallel for schedule(dynamic, 128) reduction(&&:done)
        for (int v = 0; v < n; v++) {
            int t = tags[v];

            for (const int *u = g.begin_neighbors(v); u != g.end_neighbors(v); u++) {
                t = min(t, tags[*u]);
            }

            new_tags[v] = t;
            done &= (tags[v] == t);
        }

        swap(tags, new_tags);
    }

    copy(tags.get(), tags.get() + n, new_labels);
}

