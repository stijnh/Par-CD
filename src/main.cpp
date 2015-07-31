#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <omp.h>
#include <vector>

#include "common.hpp"
#include "framework.hpp"
#include "refinement.hpp"
#include "rules.hpp"

using namespace std;

#define STRATEGY_PROP (0)
#define STRATEGY_MERGE (1)
#define STRATEGY_ALTERNATE (2)

void print_usage(char *prog) {
    cerr <<
        "usage: " << prog << " <graph> [options]                        \n"
        "                                                               \n"
        "options:                                                       \n"
        "-m <metric>                                                    \n"
        "  Set the metric to optimize for:                              \n"
        "    mod: Modularity by Newman & Girvan                         \n"
        "    wcc: Weighted Community Clustering by Prat et al.          \n"
        "    cpm-<gamma>: Constant Potts Model by Traag et al.          \n"
        "-p <#threads>                                                  \n"
        "  Set the number of threads to use.                            \n"
        "-f <fraction>                                                  \n"
        "  Set the fraction of vertices to update for label propagation.\n"
        "-r <#rounds>                                                   \n"
        "  Set the number of rounds for parallel merging.               \n"
        "-t <threshold>                                                 \n"
        "  Set the iteration threshold.                                 \n"
        "-n <#iterations>                                               \n"
        "  Set the total number of iteration. This will disable the     \n"
        "  iteration threshold set with -t.                             \n"
        "-i 0/1                                                         \n"
        "  Set the heuristic for creating the initial partitioning:     \n"
        "    0: unique labels heuristic                                 \n"
        "    1: triangle-based heuristic                                \n"
        "-s <schedule>                                                  \n"
        "  Set the schedule for the refinement phase:                   \n"
        "    0: Semi-parallel label propagation                         \n"
        "    1: Best-neighbor parallel merging                          \n"
        "    2: Alternate the two strategies                            \n"
        "-h                                                             \n"
        "  Print this help message.                                     \n"
        << endl;
}

int main(int argc, char *argv[]) {

#ifdef DEBUG
    cerr << "------------------------------------------------" << endl;
    cerr << "                     WARNING                    " << endl;
    cerr << " DEBUG MODE ENABLED. PERFORMANCE IS NOT OPTIMAL " << endl;
    cerr << "                                                " << endl;
    cerr << "------------------------------------------------" << endl;
#endif

    int c;
    int num_threads = omp_get_max_threads();
    double frac = 0.5;
    int rounds = 1;
    int max_iter = -1;
    double threshold = 0;
    int initial = 0;
    int strategy = 0;
    Rule *r = new ModularityRule();

    while ((c = getopt(argc, argv, "?hr:f:m:p:t:s:i:m:n:")) != -1) {
        switch (c) {
            case 'p':
                num_threads = atoi(optarg);

                if (num_threads < 0) {
                    cerr << "number of threads should be positive" << endl;
                    exit(-1);
                }
                break;

            case 'r':
                rounds = atoi(optarg);

                if (rounds <= 0) {
                    cerr << "number of rounds should be positive" << endl;
                    exit(-1);
                }
                break;

            case 'n':
                max_iter = atoi(optarg);
                break;

            case 'i':
                initial = atoi(optarg);
                break;

            case 'f':
                frac = atof(optarg);

                if (frac <= 0 || frac > 1) {
                    cerr << "fraction should be between 0.0 and 1.0" << endl;
                    exit(-1);
                }
                break;

            case 't':
                threshold = atof(optarg);

                if (threshold < 0 || threshold > 1) {
                    cerr << "threshold should be between 0.0 and 1.0" << endl;
                    exit(-1);
                }
                break;

            case 's':
                strategy = atoi(optarg);
                break;

            case 'm':
                for (int i = 0; optarg[i]; i++) optarg[i] = tolower(optarg[i]);

                if (r) delete r;

                if (strcmp(optarg, "mod") == 0 || strcmp(optarg, "modularity") == 0) {
                    r = new ModularityRule();
                } else if (strcmp(optarg, "wcc") == 0) {
                    r = new WCCRule();
                } else if (strncmp(optarg, "cpm-", 4) == 0 && optarg[4] != 0) {
                    double gamma = atof(optarg + 4);
                    r = new CPMRule(gamma);
                } else {
                    cerr << "unknown metric: " << optarg << endl;
                    exit(-1);
                }
                break;

            case 'h':
            case '?':
                print_usage(argv[0]);
                exit(-1);
                break;

            default:
                cerr << "unknown option" << endl << endl;
                print_usage(argv[0]);
                exit(-1);
                break;
        }
    }

    double time_load;
    double time_initial;
    double time_refine;

    if (optind >= argc) {
        print_usage(argv[0]);
        exit(-1);
    }

    char *graph_file = argv[optind];
    cout << "loading graph..." << endl;

    timer_start();
    const Graph *const g = Graph::read(graph_file);
    time_load = timer_end();

    if (g == NULL) {
        cerr << "error while loading graph" << endl;
        exit(-1);
    }

    cout << "-------------------------------------------------" << endl;
    cout << "loaded graph from " << graph_file << endl;
    cout << "num nodes: " << g->n << endl;
    cout << "num edges: " << g->m << endl;
    cout << "avg. degree: " << double(2 * g->m) / g->n << endl;
    cout << "max. degree: " << g->degree(0) << endl;
    cout << "-------------------------------------------------" << endl;


    const int n = g->n;
    int *labels = new int[n];

    cout << "creating initial partitioning..." << endl;

    timer_start();
    if (initial == 1 || dynamic_cast<WCCRule*>(r)) {
        timer_start();
        g->count_triangles();
        cout << "count triangles: " << timer_end() << endl;
    }

    switch (initial) {
        case 0:
            create_singleton_partition(*g, labels);
            break;

        case 1:
            create_clustering_partition(*g, labels);
            break;

        default:
            cerr << "unknown partitioning strategy" << endl;
            exit(-1);
            break;
    }
    time_initial = timer_end();

    Partition *p = new Partition(labels);
    double prev_score = 0.0;
    bool done = false;
    int it = 0;

    PropagationRefinement pr(frac, num_threads);
    MergingRefinement mr(rounds, num_threads);

    cout << "refining partitioning..." << endl;

    timer_start();
    while (!done) {

        timer_start();

        timer_start();
        compress_labels(*g, p->labels);
        collect_statistics(*g, *p, num_threads);
        cout << " - collecting statistics: " << timer_end() << " sec" << endl;

        timer_start();
        double score = r->calculate(*g, *p);
        cout << " - calculate metric: " << timer_end() << " sec" << endl;
        cout << " - " << r->get_name() << ": " << score << endl;
        cout << " - improv. of prev. iteration: " << (score / prev_score - 1) * 100 << "%" << endl;

        done = max_iter != -1 ? it >= max_iter : 
            (prev_score > 0 && score > 0 && score / prev_score - 1 <= threshold);

        if (!done) {
            int *new_labels = new int[n];

            if (strategy == STRATEGY_PROP || (strategy == STRATEGY_ALTERNATE && it % 2 == 0)) {
                timer_start();
                pr.refine(*g, *p, *r, new_labels);
                cout << " - label propagation: " << timer_end() << " sec" << endl;
            }

            else {
                timer_start();
                mr.refine(*g, *p, *r, new_labels);
                cout << " - parallel merging: " << timer_end() << " sec" << endl;
            }

            delete p;
            p = new Partition(new_labels);
        }

        cout << "iteration " << (it + 1) << ": " << timer_end() << " sec" << endl;
        it++;
        prev_score = score;
    }

    time_refine = timer_end();

    cout << "-------------------------------------------------" << endl;
    cout << "time read graph: " << time_load << " sec" << endl;
    cout << "time initial partitioning: " << time_initial << " sec" << endl;
    cout << "time refinement phase: " << time_refine << " sec" << endl;
    cout << "refinement iterations: " << it << endl;
    cout << "avg. time per iteration: " << time_refine / it << " sec" << endl;
    cout << r->get_name() << ": " << prev_score << endl;
    cout << "num. communities: " << p->num_labels << endl;
    cout << "avg. community size: " << double(n) / p->num_labels << endl;
    cout << "largest community size: " << *max_element(p->sizes, p->sizes + p->num_labels) << endl;
    cout << "-------------------------------------------------" << endl;

    for (int i = 0; i < n; i ++) {
        cerr << g->map[i] << " " << p->labels[i] << endl;
    }

    return 1;
}
