#ifndef RULES_H
#define RULES_H

#include <string>

#include "graph.hpp"
#include "common.hpp"

class Rule {
    public:
        virtual double calculate(const Graph &g, const Partition &p) const = 0;
        virtual double score_merge(const Graph &g, const Partition &p,
                int c1, int c2, int num_links) const = 0;
        virtual double score_join(const Graph &g, const Partition &p,
                int v, int c, int num_neighbors) const = 0;
        virtual double score_leave(const Graph &g, const Partition &p,
                int v, int num_neighbors) const = 0;
        virtual std::string get_name(void) const = 0;
};


class ModularityRule : public Rule {
    public:
        virtual double calculate(const Graph &g, const Partition &p) const;
        virtual double score_merge(const Graph &g, const Partition &p,
                int c1, int c2, int num_links) const;
        virtual double score_join(const Graph &g, const Partition &p,
                int v, int c, int num_neighbors) const;
        virtual double score_leave(const Graph &g, const Partition &p,
                int v, int num_neighbors) const;
        virtual std::string get_name(void) const;
};

class ConductanceRule : public Rule {
    public:
        virtual double calculate(const Graph &g, const Partition &p) const;
        virtual double score_merge(const Graph &g, const Partition &p,
                int c1, int c2, int num_links) const;
        virtual double score_join(const Graph &g, const Partition &p,
                int v, int c, int num_neighbors) const;
        virtual double score_leave(const Graph &g, const Partition &p,
                int v, int num_neighbors) const;
        virtual std::string get_name(void) const;
};


class CPMRule : public Rule {
    private:
        double lambda;

    public:
        CPMRule(double l) : lambda(l) {};
        virtual double calculate(const Graph &g, const Partition &p) const;
        virtual double score_merge(const Graph &g, const Partition &p,
                int c1, int c2, int num_links) const;
        virtual double score_join(const Graph &g, const Partition &p,
                int v, int c, int num_neighbors) const;
        virtual double score_leave(const Graph &g, const Partition &p,
                int v, int num_neighbors) const;
        virtual std::string get_name(void) const;
};


class WCCRule : public Rule {
    public:
        virtual double calculate(const Graph &g, const Partition &p) const;
        virtual double score_merge(const Graph &g, const Partition &p,
                int c1, int c2, int num_links) const;
        virtual double score_join(const Graph &g, const Partition &p,
                int v, int c, int num_neighbors) const;
        virtual double score_leave(const Graph &g, const Partition &p,
                int v, int num_neighbors) const;
        virtual std::string get_name(void) const;
};

#endif
