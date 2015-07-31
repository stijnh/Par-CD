#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "framework.hpp"
#include "framework-cuda.cuh"
#include "cuda-common.cuh"

using namespace std;

void print_usage(char *prog) {
    cerr <<
        "usage: " << prog << " <graph> [options]                        \n"
        "                                                               \n"
        "options:                                                       \n"
        "-m <metric>                                                    \n"
        "  Set the metric to optimize for:                              \n"
        "    mod: Modularity by Newman & Girvan                         \n"
        "    cpm-<gamma>: Constant Potts Model by Traag et al.          \n"
        "-f <fraction>                                                  \n"
        "  Set the fraction of vertices to update for label propagation.\n"
        "-n <#iterations>                                               \n"
        "  Set the total number of iteration. This will disable the     \n"
        "  iteration threshold set with -t.                             \n"
        "-s <schedule>                                                  \n"
        "  Set the schedule for the refinement phase:                   \n"
        "    0: Semi-parallel label propagation                         \n"
        "    1: Best-neighbor parallel merging                          \n"
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
    double frac = 0.5;
    double threshold = 0;
    int strategy = 0;
    int max_iter = 10;
    CudaRule *r = new ModularityCudaRule();


    while ((c = getopt(argc, argv, "?hf:m:s:m:n:")) != -1) {
        switch (c) {
            case 'f':
                frac = atof(optarg);

                if (frac <= 0 || frac > 1) {
                    cerr << "fraction should be between 0.0 and 1.0" << endl;
                    exit(-1);
                }
                break;

            case 'n':
                max_iter = atoi(optarg);

                if (max_iter <= 0) {
                    cerr << "number of iterations should be positive" << endl;
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
                    r = new ModularityCudaRule();
                } else if (strncmp(optarg, "cpm-", 4) == 0 && optarg[4] != 0) {
                    double gamma = atof(optarg + 4);
                    r = new CPMRule(gamma);
                } else {
                    cerr << "unknown metric: " << optarg << endl;
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

    if (optind >= argc) {
        print_usage(argv[0]);
        exit(-1);
    }

    timer_start();
    cuda_init(0, 256);
    cout << "CUDA init: " << timer_end() << " sec" << endl;

    timer_start();
    const Graph *const g = Graph::read(argv[1]);
    cout << "read graph: " << timer_end() << " sec" << endl;

    timer_start();
    cuda_transfer_graph(*g);
    cout << "transfer graph: " << timer_end() << " sec" << endl;

    const int n = g->n;
    int *labels = new int[n];
    create_singleton_partition(*g, labels);

    timer_start();
    cuda_labels_to_device(labels);
    cout << "transfer labels: " << timer_end() << " sec" << endl;

    for (int i = 0; i < max_iter; i++) {
        timer_start();

        cuda_collect_statistics();

        float metric = cuda_calculate_metric(*r);
        cout << " - " << r->get_name() << ": " << metric << endl;

        timer_start();
        if (strategy == 0) {
            cuda_propagation_step(frac, r);
        } else {
            cuda_merging_step(1, r);
        }

        cout << "- iteration: " << timer_end() << " sec" << endl;
    }

    timer_start();
    cuda_labels_from_device(labels);
    cout << "transfer labels: " << timer_end() << " sec" << endl;

    cuda_deinit();

    return EXIT_SUCCESS;
}
