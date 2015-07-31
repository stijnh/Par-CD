#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstddef>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <string>
#include <utility>

#include "graph.hpp"

#if defined __NVCC__
#define INLINE __attribute__((always_inline)) __forceinline__
#else
#define INLINE __attribute__((always_inline)) __inline__
#endif

#define NEW_LABEL (~0)

double timer(void);
void timer_start(void);
double timer_end(void);

class Partition {
    public:
        int num_labels;
        int *labels;
        int *sizes;
        int *int_volume;
        int *ext_volume;

        Partition(int n);
        Partition(int *labels);
        ~Partition();
};


double compare_communities(const Graph &g, int *a, int *b);
double compare_communities(const Graph &g, int *labels, std::string filename);
double norm_mutual_info(const Graph &g, std::map<std::pair<int, int>, int> &confusion_matrix);

template <typename T> std::string to_string(const T& n) {
    std::ostringstream stm;
    stm << n;
    return stm.str();
}

template <typename T>
class scoped_ptr {
    private:
        T *ptr;

    public:
        scoped_ptr(size_t s) {
            this->ptr = new T[s];
        }

        scoped_ptr(T *p=NULL) {
            this->ptr = p;
        }

        T *get() {
            return ptr;
        }

        template <typename S>
        T *get(S i) {
            return (ptr + i);
        }

        template <typename S>
        T &operator[] (S i) {
            return *(ptr + i);
        }

        T *operator->() const {
            return ptr;
        }

        scoped_ptr<T> &operator=(scoped_ptr<T> &rhs) {
            ptr = rhs.ptr;
            rhs.ptr = NULL;
            return *this;
        }

        T &operator*() const {
            return *ptr;
        }

        ~scoped_ptr<T>() {
            if (ptr) {
                delete[] ptr;
            }
        }
};

#endif
