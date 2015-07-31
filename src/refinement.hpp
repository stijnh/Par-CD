#ifndef REFINEMENT_HPP
#define REFINEMENT_HPP

#include "graph.hpp"
#include "common.hpp"
#include "rules.hpp"

class Refinement {
    public:
        virtual void refine(const Graph &g, const Partition &p, const Rule &r, int *new_labels) = 0;
};

class PropagationRefinement : public Refinement {
    private:
        double frac;
        int num_threads;

    public:
        PropagationRefinement(double f, int n=1) : frac(f), num_threads(n) { };
        virtual void refine(const Graph &g, const Partition &p, const Rule &r, int *new_labels);
};

class MergingRefinement : public Refinement {
    private:
        int rounds;
        int num_threads;

    public:
        MergingRefinement(int r, int n=1) : rounds(r), num_threads(n) {};
        virtual void refine(const Graph &g, const Partition &p, const Rule &r, int *new_labels);
};

class FindConnectedRefinement : public Refinement {
    private:
        int num_threads;

    public:
        FindConnectedRefinement(int n=1) : num_threads(n) {};
        virtual void refine(const Graph &g, const Partition &p, const Rule &r, int *new_labels);
};

#endif
