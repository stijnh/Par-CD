#ifndef CUDA_COMMON
#define CUDA_COMMON

#include <cuda.h>
#include <curand.h>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <typeinfo>

#include "common.hpp"

#define GROUP_SIZE (512)

#define CUDA_NAN_FLOAT (interpret_int_as_float(0x7fffffff))

#define CURAND_CHECK(msg, res) \
    __curand_check(__FILE__, __func__, __LINE__, (msg), (res))

#define CURAND_CALL(func, ...) \
    CURAND_CHECK(#func, func(__VA_ARGS__))

#define CUDA_CHECK(msg, res) \
    __cuda_check(__FILE__, __func__, __LINE__, (msg), (res))

#define CUDA_CALL(func, ...) \
    CUDA_CHECK(#func, func(__VA_ARGS__))

#define CUDA_CHECK_LAST(msg) \
   CUDA_CHECK(msg, cudaPeekAtLastError())

#define CUDA_SYNC() \
    CUDA_CALL(cudaDeviceSynchronize)

#define CUDA_SYNC_STREAM(stream) \
    CUDA_CALL(cudaStreamSynchronize, stream)

#define CUDA_LAUNCH(stream, global_size, block_size, kernel, ...) \
    do { \
        dim3 __grid(global_size / block_size + (global_size % block_size != 0), 1); \
        dim3 __block(block_size, 1); \
        if (__grid.x >= 65536) __grid.x = __grid.y = ceil(sqrt(__grid.x)); \
        kernel<<<__grid, __block, 0, stream>>>(__VA_ARGS__); \
        CUDA_CHECK_LAST(#kernel); \
    } while (0)

#define CUDA_FILL(stream, ptr, count, value) \
    CUDA_LAUNCH(stream, count, 512, kernel_fill, count, ptr, value)

#define CUDA_CLEAR(stream, ptr, count) \
    CUDA_CALL(cudaMemsetAsync, ptr, 0, sizeof(*(ptr)) * (count), stream)

INLINE static __device__ __host__ float interpret_int_as_float(const int x) {
    union {int from; float to;} tmp;
    tmp.from = x;
    return tmp.to;
}

INLINE static __device__ __host__ int interpret_float_as_int(const float x) {
    union {float from; int to;} tmp;
    tmp.from = x;
    return tmp.to;
}

INLINE static __device__ int get_global_id(void) {
      return blockIdx.x * blockDim.x + threadIdx.x +
             (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x;
}

INLINE static __device__ int get_local_id(void) {
    return threadIdx.x + threadIdx.y * blockDim.x;
}

INLINE static __device__ int get_global_size(void) {
      return blockDim.x * gridDim.x * blockDim.y * gridDim.y;
}

INLINE static __device__ int get_local_size(void) {
    return blockDim.x * blockDim.y;
}

INLINE static curandStatus_t __curand_check(const char *file, const char *func,
        int line, const char *msg, curandStatus_t code) {
    if (code != CURAND_STATUS_SUCCESS) {
        std::cerr << "CURAND fatal error: " << file << ":" << func << ":"
            << line << ": " << msg << std::endl;
        exit(code);
    }

    return code;
}

INLINE static cudaError_t __cuda_check(const char *file, const char *func,
        int line, const char *msg, cudaError_t code) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA fatal error: " << file << ":" << func << ":"
            << line << ": " << msg << ": "
            << cudaGetErrorString(code) << std::endl;
        exit(code);
    }

    return code;
}

template <typename T>
__global__ void kernel_fill(const int n, T *ptr, T val) {
    int i = get_global_id();
    if (i < n) ptr[i] = val;
}

template <typename A, typename B>
struct tuple {
    A left;
    B right;
};

template <typename A, typename B>
__host__ __device__ tuple<A, B> make_tuple(A a, B b) {
    tuple<A, B> t;
    t.left = a;
    t.right = b;
    return t;
}

template <typename ItA, typename ItB>
class zip_iterator: public std::iterator_traits<tuple<
                    typename std::iterator_traits<ItA>::value_type,
                    typename std::iterator_traits<ItB>::value_type>*> {
    typedef typename std::iterator_traits<ItA>::value_type A;
    typedef typename std::iterator_traits<ItB>::value_type B;

    private:
        ItA a_it;
        ItB b_it;

    public:
        __host__ __device__ zip_iterator(ItA a, ItB b) :
            a_it(a), b_it(b) { }

        __host__ __device__ tuple<A, B> operator[](ptrdiff_t i) const {
            return make_tuple<A, B>(a_it[i], b_it[i]);
        }

        __host__ __device__ zip_iterator<ItA, ItB> operator+(size_t i) {
            return make_zip_iterator(a_it + i, b_it + i);
        }
};

template <typename ItA, typename ItB>
__host__ __device__ static zip_iterator<ItA, ItB> make_zip_iterator(ItA a, ItB b) {
    return zip_iterator<ItA, ItB>(a, b);
}

template <typename It, typename F>
class transform_iterator: public std::iterator_traits<typename F::result_type*> {
    private:
        It p;
        F f;

    public:
        __device__ __host__ transform_iterator(It ptr, F func): p(ptr), f(func) {
            //
        }

        template <typename I>
        __device__ __host__ transform_iterator<It, F> operator+(I i) {
            return transform_iterator(p + i, f);
        }

        template <typename T>
        __device__ __host__ typename F::result_type operator[](T i) const {
            return f(p[i]);
        }
};

template <typename It, typename F>
__host__ __device__ static transform_iterator<It, F> make_transform_iterator(It it, F f) {
    return transform_iterator<It, F>(it, f);
}

#include <typeinfo>

class CudaAllocator {
    private:
        std::vector<void*> ptrs;

    public:
        CudaAllocator() {
            //
        }

        ~CudaAllocator()  {
            deallocate_all();
        }

#ifdef DEBUG
        static void print_mem_info() {
            size_t f, t;
            CUDA_CALL(cudaMemGetInfo, &f, &t);

            std::cout << "memory usage: "
                << ((t - f) / 1024.0 / 1024.0) << "MB / "
                << (t / 1024.0 / 1024.0) << "MB ["
                << int(100.0 - 100.0 * f / t) << "%]";

        }
#endif

        template <typename T>
        T *allocate(size_t n) {
            size_t size = sizeof(T) * n;

#ifdef DEBUG
            std::cout << "allocate " << n << " x " << typeid(T).name() << " = " << (size / 1024.0 / 1024.0) << "MB (";
            print_mem_info();
            std::cout << ")" << std::endl;
#endif

            T *ptr;
            CUDA_CALL(cudaMalloc, &ptr, size);
            ptrs.push_back(ptr);
            return ptr;
        }

        template <typename T>
        void deallocate(T *ptr) {
            std::vector<void*>::iterator it = find(ptrs.begin(), ptrs.end(), ptr);
            if (it == ptrs.end()) {
                std::cerr << "failed to deallocate pointer " << ptr << std::endl;
                exit(-1);
            }

            CUDA_CALL(cudaFree, ptr);
            ptrs.erase(it);

#ifdef DEBUG
            std::cout << "deallocate " << ptr << " (";
            print_mem_info();
            std::cout << ")" << std::endl;
#endif
        }

        void deallocate_all() {
            while (!ptrs.empty()) {
                deallocate(ptrs[0]);
            }
        }
};

class CudaTimer {
    private:
        cudaStream_t stream;
        std::string name;
        std::vector<std::string> timers;
        std::vector<cudaEvent_t> events;

    public:
        CudaTimer(cudaStream_t s, std::string n) {
            stream = s;
            name = n;
        }

        ~CudaTimer() {
            for (size_t i = 0; i < events.size(); i++) {
                CUDA_CALL(cudaEventDestroy, events[i]);
            }
        }

        void start() {
            cudaEvent_t evt;
            CUDA_CALL(cudaEventCreate, &evt);
            CUDA_CALL(cudaEventRecord, evt, stream);
            events.push_back(evt);
        }

        void event(std::string name) {
//#ifdef DEBUG
            cudaEvent_t evt;
            CUDA_CALL(cudaEventCreate, &evt);
            CUDA_CALL(cudaEventRecord, evt, stream);

            timers.push_back(name);
            events.push_back(evt);
//#endif
        }

        void stop() {
            cudaEvent_t evt;
            CUDA_CALL(cudaEventCreate, &evt);
            CUDA_CALL(cudaEventRecord, evt, stream);
            CUDA_CALL(cudaEventSynchronize, evt);
            events.push_back(evt);

            cudaEvent_t first = events[0];
            cudaEvent_t last = evt;

            float time;
            CUDA_CALL(cudaEventElapsedTime, &time, first, last);

            std::cout << " - " << name << ": " << (time / 1000) << " sec" << std::endl;

            for (size_t i = 0; i < timers.size(); i++) {
                std::string name = timers[i];
                cudaEvent_t before = events[i + 1];
                cudaEvent_t after = events[i + 2];

                float time;
                CUDA_CALL(cudaEventElapsedTime, &time, before, after);

                std::cout << "   - " << name << ": " << (time / 1000) << " sec" << std::endl;
            }
        }
};

INLINE static __host__ __device__ int2 operator+(int2 a, int2 b) {
    return make_int2(a.x + b.x, a.y + b.y);
}

INLINE static __host__ __device__ bool operator<(int2 a, int2 b) {
    return a.x != b.x ? a.x < b.x : a.y < b.y;
}

INLINE static __host__ __device__ bool operator==(int2 a, int2 b) {
    return a.x == b.x && a.y == b.y;
}

#endif
