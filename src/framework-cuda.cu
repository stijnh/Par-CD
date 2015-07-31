#include <assert.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#include "framework-cuda.cuh"
#include "cuda-common.cuh"
#include "cub/cub.cuh"
#include "moderngpu.cuh"

static int device_id;

static CudaAllocator global_allocator;

static int num_vertices;
static int num_endpoints;
static int num_edges;
static int num_comms;
static size_t temp_size;

static int2 *mem_edges;
static int *mem_deg;
static int *mem_labels;
static int *mem_new_labels;

static int *mem_stats_sizes;
static int *mem_stats_ext;
static int *mem_stats_int;
static void *mem_temp;
static curandState_t *mem_rng;

static cudaStream_t stream;

#ifdef DEBUG
static bool debug_flag = true;
#else
static bool debug_flag = false;
#endif

using namespace std;
using namespace cub;

#define CUB_CALL(func, ...) \
    do { \
        size_t __size__ = 0; \
        func((void*) NULL, __size__, __VA_ARGS__, stream, debug_flag); \
        if (__size__ > temp_size) { \
            global_allocator.deallocate(mem_temp); \
            mem_temp = global_allocator.allocate<char>(temp_size = __size__); \
        } \
        CUDA_CALL(func, mem_temp, temp_size, __VA_ARGS__, stream, debug_flag); \
    } while(0)

__global__ void ker_init() {
    //
}

bool cuda_init(int dev_id, int grp_size) {
    device_id = dev_id;
    //GROUP_SIZE = grp_size;

    // Set device and force initialization of this device
    CUDA_CALL(cudaSetDevice, device_id);
    CUDA_CALL(cudaStreamCreate, &stream);

    CUDA_CALL(cudaFree, (void*) NULL);
    CUDA_LAUNCH(stream, 1, 1, ker_init);
    CUDA_SYNC();

    return true;
}

bool cuda_deinit(void) {
    global_allocator.deallocate_all();
    return true;
}

__global__ void ker_seed_rng(
        const int n,
        const unsigned int seed,
        curandState_t *__restrict__ rng) {
    int i = get_global_id();
    if (i >= n) return;

    curand_init(seed, get_local_id(), 0, &rng[i]);
}

bool cuda_transfer_graph(const Graph &g) {
    const int n = g.n;
    const int m = g.m;
    const int e = 2 * g.m;
    num_comms = n;
    num_vertices = n;
    num_endpoints = e;
    num_edges = m;
    temp_size = sizeof(int) * 1024;

    mem_temp = global_allocator.allocate<char>(temp_size);
    mem_deg = global_allocator.allocate<int>(n);
    mem_labels = global_allocator.allocate<int>(n);
    mem_new_labels = global_allocator.allocate<int>(n);
    mem_edges = global_allocator.allocate<int2>(e);
    mem_stats_sizes = global_allocator.allocate<int>(n);
    mem_stats_ext = global_allocator.allocate<int>(n);
    mem_stats_int = global_allocator.allocate<int>(n);
    mem_rng = global_allocator.allocate<curandState_t>(n);

    int2 *edges = new int2[e];
    int *deg = new int[n];
    int index = 0;
    for (int v = 0; v < n; v++) {
        deg[v] = g.degree(v);

        for (const int *u = g.neighbors(v); u != g.end_neighbors(v); u++) {
            edges[index++] = make_int2(v, *u);
        }
    }

    CUDA_CALL(cudaMemcpy, mem_edges, edges, sizeof(int2) * e, cudaMemcpyHostToDevice);
    CUDA_CALL(cudaMemcpy, mem_deg, deg, sizeof(int) * n, cudaMemcpyHostToDevice);

    delete[] edges;
    delete[] deg;

    CUDA_LAUNCH(stream, n, GROUP_SIZE,
            ker_seed_rng,
            n,
            0,
            mem_rng);

    return true;
}

bool cuda_labels_to_device(int *labels) {
    CUDA_CALL(cudaMemcpy, mem_labels, labels, sizeof(int) * num_vertices, cudaMemcpyHostToDevice);
    return true;
}

bool cuda_labels_from_device(int *labels) {
    CUDA_CALL(cudaMemcpy, labels, mem_labels, sizeof(int) * num_vertices, cudaMemcpyDeviceToHost);
    return true;
}

static CudaGraph make_cuda_graph() {
    CudaGraph g;
    g.n = num_vertices;
    g.m = num_edges;
    g.mem_edges = mem_edges;
    g.mem_deg = mem_deg;
    return g;
}

static CudaPartition make_cuda_partition() {
    CudaPartition p;
    p.num_labels = num_vertices;
    p.mem_labels = mem_labels;
    p.mem_stats_sizes = mem_stats_sizes;
    p.mem_stats_ext = mem_stats_ext;
    p.mem_stats_int = mem_stats_int;
    return p;
}

float cuda_calculate_metric(CudaRule &rule) {
    CudaTimer timer(stream, "calculate metric");

    timer.start();
    float v = rule.calculate(stream, make_cuda_graph(), make_cuda_partition());
    timer.stop();

    return v;
}

__global__ void ker_compress_labels(
        const int n,
        const int k,
        int *__restrict__ labels,
        const int *__restrict__ unique_labels) {
    int id = get_global_id();
    if (id >= n) return;

    int label = labels[id];
    int lbnd = 0;
    int ubnd = k;
    int index;

    while (true) {
        index = (lbnd + ubnd) / 2;
        int l = unique_labels[index];

        if (label > l) {
            lbnd = index + 1;
        } else if (label < l) {
            ubnd = index;
        } else {
            break;
        }

        if ((lbnd + ubnd) / 2 == index) {
            printf("ERROR! %d] %d-%d -> %d (%d != %d)\n", id, lbnd, ubnd, index, unique_labels[index], label);
            break;
        }
    }

    labels[id] = index;
}

__global__ void ker_update_stats_sizes(
        const int n,
        const int *__restrict__ labels,
        int *__restrict__ sizes) {
    int id = get_global_id();
    if (id >= n) return;

    atomicAdd(sizes + labels[id], 1);
}

__global__ void ker_update_stats_edges(
        const int e,
        const int *__restrict__ labels,
        const int2 *__restrict__ edges,
        int *__restrict__ ext_edges,
        int *__restrict__ int_edges) {
    int i = get_global_id();
    if (i >= e) return;

    int2 edge = edges[i];
    int la = labels[edge.x];
    int lb = labels[edge.y];

    atomicAdd((la == lb ? int_edges : ext_edges) + la, 1);
}


bool cuda_collect_statistics() {
    const int n = num_vertices;

    CudaTimer timer(stream, "collect statistics");
    CudaAllocator allocator;

    timer.start();
    timer.event("find min/max label");
    int *mem_min = allocator.allocate<int>(1);
    int *mem_max = allocator.allocate<int>(1);

    CUB_CALL(DeviceReduce::Min,
            mem_labels,
            mem_min,
            n);

    CUB_CALL(DeviceReduce::Max,
            mem_labels,
            mem_max,
            n);

    int l_min, l_max;
    CUDA_CALL(cudaMemcpyAsync, &l_min, mem_min, sizeof(int), cudaMemcpyDeviceToHost, stream);
    CUDA_CALL(cudaMemcpyAsync, &l_max, mem_max, sizeof(int), cudaMemcpyDeviceToHost, stream);
    CUDA_CALL(cudaStreamSynchronize, stream);

    num_comms = l_max + 1;

    if (l_min < 0 || l_max >= n) {
        int *mem_old_labels = allocator.allocate<int>(n);
        int *mem_sorted_labels = allocator.allocate<int>(n);
        int *mem_num_comms = allocator.allocate<int>(1);

        timer.event("compress labels");
        CUDA_CALL(cudaMemcpyAsync, mem_old_labels, mem_labels, n * sizeof(int), cudaMemcpyDeviceToDevice, stream);

        DoubleBuffer<int> buffer(mem_old_labels, mem_sorted_labels);
        CUB_CALL(DeviceRadixSort::SortKeys,
                buffer,
                n,
                0,
                sizeof(int) * 8);

        mem_sorted_labels = buffer.Current();
        int *mem_unique_labels = buffer.Alternate();

        CUB_CALL(DeviceRunLengthEncode::Encode,
                mem_sorted_labels,
                mem_unique_labels,
                mem_stats_sizes,
                mem_num_comms,
                n);

        CUDA_CALL(cudaMemcpyAsync, &num_comms, mem_num_comms, sizeof(int), cudaMemcpyDeviceToHost, stream);
        CUDA_CALL(cudaStreamSynchronize, stream);

        CUDA_LAUNCH(stream, n, GROUP_SIZE,
                ker_compress_labels,
                n,
                num_comms,
                mem_labels,
                mem_unique_labels);
    }

    timer.event("clear old data");
    CUDA_CLEAR(stream, mem_stats_sizes, n);
    CUDA_CLEAR(stream, mem_stats_ext, n);
    CUDA_CLEAR(stream, mem_stats_int, n);

    timer.event("collect vertex statistics");
    CUDA_LAUNCH(stream, n, GROUP_SIZE,
            ker_update_stats_sizes,
            n,
            mem_labels,
            mem_stats_sizes);

    timer.event("collect edge statistics");
    CUDA_LAUNCH(stream, num_endpoints, GROUP_SIZE,
            ker_update_stats_edges,
            num_endpoints,
            mem_labels,
            mem_edges,
            mem_stats_ext,
            mem_stats_int);

    CUDA_SYNC();
    timer.stop();

    return true;
}

__global__ void ker_select_random_vertices(
        const int n,
        curandState_t *__restrict__ rng,
        const unsigned int threshold,
        bool *__restrict__ is_active)
{
    int i = get_global_id();
    if (i >= n) return;

    is_active[i] = curand(&rng[i]) < threshold;
}

struct transform_active_and_degree_tuple: unary_function<tuple<bool, int>, int2> {
    __device__ __host__ int2 INLINE operator()(tuple<bool, int> active_and_degree) const {
        bool is_active = active_and_degree.left;
        int degree = active_and_degree.right;

        return make_int2(is_active ? 1      : 0,
                         is_active ? degree : 0);
    }
};


__global__ void ker_copy_vertex_label_pairs(
        const int e,
        const int2 *__restrict__ edges,
        const int *__restrict__ labels,
        const bool *__restrict__ is_active,
        int2 *__restrict__ pairs,
        int *__restrict__ num_pairs) {
    typedef cub::BlockScan<int, GROUP_SIZE> BlockScan;

    __shared__ BlockScan::TempStorage shared_scan_storage;
    __shared__ int shared_offset;

    int gid = get_global_id();
    int lid = get_local_id();

    int a, b;
    bool act = false;
    int la, lb;

    if (gid < e) {
        int2 ed = edges[gid];
        a = ed.x;
        b = ed.y;
        act = is_active[a];

        if (act) {
            la = labels[a];
            lb = labels[b];
        }
    }

    int index, sum;
    BlockScan(shared_scan_storage).ExclusiveSum(act, index, sum);

    if (lid == 0) {
        shared_offset = atomicAdd(num_pairs, sum);
    }

    __syncthreads();
    int offset = shared_offset;

    if (act) {
        pairs[offset + index] = make_int2(
                la == lb ? INVALID_LABEL : lb, a);
    }
}

__global__ void ker_unpack_int2(
        const int n,
        const int2 *__restrict__ src,
        int *__restrict__ dst_x,
        int *__restrict__ dst_y) {
    int i = get_global_id();
    if (i >= n) return;

    int2 p = src[i];
    dst_x[i] = p.x;
    dst_y[i] = p.y;
}

struct make_best_label: unary_function<tuple<int, float>, int3> {
    __device__ __host__ INLINE int3 operator()(tuple<int, float> label_score_packed) const {
        int label = label_score_packed.left;
        float insert_score = label_score_packed.right;
        float remove_score = label == INVALID_LABEL ? -insert_score : CUDA_NAN_FLOAT;

        return make_int3(label,
                interpret_float_as_int(insert_score),
                interpret_float_as_int(remove_score));
    }
};

struct reduce_best_label: binary_function<int3, int3, int3> {
    __device__ __host__ INLINE int3 operator()(int3 a, int3 b) {
        int label_a = a.x;
        int label_b = b.x;

        float insert_a = interpret_int_as_float(a.y);
        float insert_b = interpret_int_as_float(b.y);

        int best_label = insert_a > insert_b ? label_a : label_b;
        float best_insert = insert_a > insert_b ? insert_a : insert_b;

        float remove_a = interpret_int_as_float(a.z);
        float remove_b = interpret_int_as_float(b.z);
        float remove = isnan(remove_b) ? remove_a : remove_b;

        return make_int3(best_label,
                interpret_float_as_int(best_insert),
                interpret_float_as_int(remove));
    }
};

__global__ void ker_select_best_label(
        const int k,
        const int *__restrict__ vertices,
        int *__restrict__ new_labels,
        const int3 *__restrict__ packed_best_labels) {
    int i = get_global_id();
    if (i >= k) return;

    int v = vertices[i];
    int3 p = packed_best_labels[i];
    int insert_label = p.x;
    float insert_score = interpret_int_as_float(p.y);
    float remove_score = interpret_int_as_float(p.z);

    if (isnan(remove_score)) remove_score = 0.0;

    if (remove_score > 0.0 && insert_score <= 0.0) {
        new_labels[v] = -v;
    } else if (remove_score + insert_score > 0.0 && insert_label != INVALID_LABEL) {
        new_labels[v] = insert_label;
    }
}

bool cuda_propagation_step(double frac, CudaRule &rule) {
    const int n = num_vertices;
    CudaAllocator allocator;
    CudaTimer timer(stream, "propagation");

    // -------
    timer.start();
    timer.event("select vertices to update");

    int2 *mem_nactives_and_npairs = allocator.allocate<int2>(1);
    bool *mem_is_active = allocator.allocate<bool>(n);

    int *mem_nactives = &(mem_nactives_and_npairs->x);
    int *mem_npairs = &(mem_nactives_and_npairs->y);

    CUDA_LAUNCH(stream, n, GROUP_SIZE,
            ker_select_random_vertices,
            n,
            mem_rng,
            (unsigned int)(frac * 0xffffffff),
            mem_is_active);

    CUB_CALL(DeviceReduce::Sum,
            make_transform_iterator(
                make_zip_iterator(
                    mem_is_active,
                    mem_deg
                ),
                transform_active_and_degree_tuple()
            ),
            mem_nactives_and_npairs,
            n);

    timer.event("transfer device to host");
    int npairs, nactives;
    CUDA_CALL(cudaMemcpyAsync, &npairs, mem_npairs, sizeof(int), cudaMemcpyDeviceToHost, stream);
    CUDA_CALL(cudaMemcpyAsync, &nactives, mem_nactives, sizeof(int), cudaMemcpyDeviceToHost, stream);
    CUDA_CALL(cudaStreamSynchronize, stream);

    // ------

    int2 *mem_pairs = allocator.allocate<int2>(npairs);
    int2 *mem_pairs_sorted = allocator.allocate<int2>(npairs);
    int *mem_counts = allocator.allocate<int>(npairs);
    int *mem_dummy = allocator.allocate<int>(1);
    int *mem_npairs_unique = allocator.allocate<int>(1);

    CUDA_CLEAR(stream, mem_dummy, 1);

    timer.event("copy vertex-label pairs");
    CUDA_LAUNCH(stream, num_endpoints, GROUP_SIZE,
            ker_copy_vertex_label_pairs,
            num_endpoints,
            mem_edges,
            mem_labels,
            mem_is_active,
            mem_pairs,
            mem_dummy);

    DoubleBuffer<int2> buffer_pairs(mem_pairs, mem_pairs_sorted);

    timer.event("sort vertex-label pairs");
    assert(sizeof(int2) == sizeof(long long unsigned int));
    {
        mgpu::ContextPtr ctx = mgpu::CreateCudaDeviceAttachStream(device_id, stream);

        mgpu::LocalitySortKeys(
                (unsigned long long int*) mem_pairs,
                npairs,
                *ctx,
                true);
    }
    int2 *mem_pairs_unique = mem_pairs_sorted;
    mem_pairs_sorted = mem_pairs;

    //CUB_CALL(DeviceRadixSort::SortKeys,
    //        *(DoubleBuffer<long long unsigned int>*) (void*) &buffer_pairs,
    //        npairs,
    //        0,
    //        sizeof(int2) * 8);

    //mem_pairs_sorted = buffer_pairs.Current();
    //int2 *mem_pairs_unique = buffer_pairs.Alternate();

    timer.event("count distinct vertex-label pairs");
    CUB_CALL(DeviceReduce::ReduceByKey,
            mem_pairs_sorted,
            mem_pairs_unique,
            cub::ConstantInputIterator<int>(1),
            mem_counts,
            mem_npairs_unique,
            cub::Sum(),
            npairs);

    timer.event("transfer device to host");
    int npairs_unique;
    CUDA_CALL(cudaMemcpy, &npairs_unique, mem_npairs_unique, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CALL(cudaStreamSynchronize, stream);
    cout << npairs << " " << npairs_unique << " " << nactives << endl;

    allocator.deallocate(mem_pairs_sorted);

    // ------

    int *mem_active_vertices = allocator.allocate<int>(nactives);
    int3 *mem_best_labels = allocator.allocate<int3>(nactives);
    float *mem_scores = allocator.allocate<float>(npairs);
    int *mem_pairs_vertices = allocator.allocate<int>(npairs_unique);
    int *mem_pairs_labels = allocator.allocate<int>(npairs_unique);

    timer.event("calculate improvements");

    CUDA_LAUNCH(stream, npairs_unique, GROUP_SIZE,
            ker_unpack_int2,
            npairs_unique,
            mem_pairs_unique,
            mem_pairs_labels,
            mem_pairs_vertices);

    rule.score_join(stream, make_cuda_graph(), make_cuda_partition(),
            npairs_unique,
            mem_pairs_vertices,
            mem_pairs_labels,
            mem_counts,
            mem_scores);

    timer.event("find best labels");

    CUB_CALL(DeviceReduce::ReduceByKey,
            mem_pairs_vertices,
            mem_active_vertices,
            make_transform_iterator(
                make_zip_iterator(
                    mem_pairs_labels,
                    mem_scores
                ),
                make_best_label()
            ),
            mem_best_labels,
            mem_nactives,
            reduce_best_label(),
            npairs_unique);
    CUDA_SYNC();

    timer.event("apply labels");
    CUDA_SYNC();

    CUDA_CALL(cudaMemcpyAsync, mem_new_labels, mem_labels, n * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    CUDA_SYNC();

    int blaz; int blaz2;
    CUDA_CALL(cudaMemcpy, &blaz, mem_nactives, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_CALL(cudaMemcpy, &blaz2, mem_dummy, sizeof(int), cudaMemcpyDeviceToHost);
    cout << "blaz: " << blaz << " " << nactives << " " << blaz2 << endl;

    CUDA_LAUNCH(stream, nactives, GROUP_SIZE,
            ker_select_best_label,
            nactives,
            mem_active_vertices,
            mem_new_labels,
            mem_best_labels);

    CUDA_SYNC();
    timer.stop();
    CUDA_SYNC();

    swap(mem_labels, mem_new_labels);
    return true;
}

__global__ void ker_copy_label_label_pairs(
        const int e,
        const int2 *__restrict__ edges,
        const int *__restrict__ labels,
        int2 *__restrict__ pairs,
        int *__restrict__ num_pairs) {
    typedef cub::BlockScan<int, GROUP_SIZE> BlockScan;

    __shared__ BlockScan::TempStorage shared_scan_storage;
    __shared__ int shared_offset;

    int gid = get_global_id();
    int lid = get_local_id();
    int la, lb;

    if (gid < e) {
        int2 ed = edges[gid];
        la = labels[ed.x];
        lb = labels[ed.y];
    } else {
        la = lb = -1;
    }

    int index, sum;
    BlockScan(shared_scan_storage).ExclusiveSum(int(la == lb), index, sum);

    if (lid == 0) {
        shared_offset = atomicAdd(num_pairs, sum);
    }

    __syncthreads();
    int offset = shared_offset;

    if (la != lb) {
        pairs[offset + index] = make_int2(la, lb);
    }
}

struct select_int2_x: unary_function<int2, int> {
    __device__ __host__ INLINE int operator()(int2 t) const {
        return t.x;
    }
};

struct select_int2_y: unary_function<int2, int> {
    __device__ __host__ INLINE int operator()(int2 t) const {
        return t.y;
    }
};

struct make_best_partner: unary_function<tuple<int, float>, int2> {
    __device__ __host__ int2 operator()(tuple<int, float> x) const {
        return make_int2(x.left, interpret_float_as_int(x.right));
    }
};

struct reduce_best_partner: binary_function<int2, int2, int2 > {
    __device__ __host__ INLINE int2 operator()(int2 a, int2 b) {
        return interpret_int_as_float(a.x) > interpret_int_as_float(b.y) ? a : b;
    }
};

__global__ void ker_set_best_partner(
        const int k,
        const int *__restrict__ selected_labels,
        const int2 *__restrict__ selected_partner,
        int *__restrict__ partner) {
    int i = get_global_id();
    if (i >= k) return;

    int l = selected_labels[i];
    int2 p = selected_partner[i];

    if (interpret_int_as_float(p.y) > 0) {
        partner[l] = p.x;
    }
}

__global__ void ker_set_has_partner(
        const int c,
        const int *__restrict__ partner,
        bool *__restrict__ has_partner) {
    int i = get_global_id();
    if (i >= c) return;

    if (partner[i] != INVALID_LABEL && partner[partner[i]] == i) {
        has_partner[i] = true;
    }
}

__global__ void ker_relabel(
        const int n,
        const int *__restrict__ labels,
        const int *__restrict__ partner,
        const bool *__restrict__ has_partner,
        int *__restrict__ new_labels) {
    int i = get_global_id();
    if (i >= n) return;

    int label = labels[i];
    int new_label = label;

    if (has_partner[label]) {
        new_label = max(label, partner[label]);
    }

    new_labels[i] = new_label;
}

bool cuda_merging_step(int rounds, CudaRule &rule) {
    const int n = num_vertices;
    CudaAllocator allocator;
    CudaTimer timer(stream, "merging");

    // -------
    timer.start();
    timer.event("count number of pairs");

    int *mem_npairs = allocator.allocate<int>(1);

    CUB_CALL(DeviceReduce::Sum,
            mem_stats_ext,
            mem_npairs,
            n);

    timer.event("transfer device to host");
    int npairs;
    CUDA_CALL(cudaMemcpyAsync, &npairs, mem_npairs, sizeof(int), cudaMemcpyDeviceToHost, stream);
    CUDA_CALL(cudaStreamSynchronize, stream);

    // ------

    int2 *mem_pairs = allocator.allocate<int2>(npairs);
    int2 *mem_pairs_sorted = allocator.allocate<int2>(npairs);
    int *mem_counts = allocator.allocate<int>(npairs);
    int *mem_dummy = allocator.allocate<int>(1);
    int *mem_npairs_unique = allocator.allocate<int>(1);

    CUDA_CLEAR(stream, mem_dummy, 1);

    timer.event("copy label-label pairs");
    CUDA_LAUNCH(stream, num_endpoints, GROUP_SIZE,
            ker_copy_label_label_pairs,
            num_endpoints,
            mem_edges,
            mem_labels,
            mem_pairs,
            mem_dummy);

    DoubleBuffer<int2> buffer_pairs(mem_pairs, mem_pairs_sorted);

    timer.event("sort pairs");

    assert(sizeof(int2) == sizeof(long long unsigned int));
    CUB_CALL(DeviceRadixSort::SortKeys,
            *(DoubleBuffer<long long unsigned int>*) (void*) &buffer_pairs,
            npairs,
            0,
            sizeof(int2) * 8);

    mem_pairs_sorted = buffer_pairs.Current();
    int2 *mem_pairs_unique = buffer_pairs.Alternate();

    timer.event("count distinct pairs");

    CUB_CALL(DeviceReduce::ReduceByKey,
            mem_pairs_sorted,
            mem_pairs_unique,
            cub::ConstantInputIterator<int>(1),
            mem_counts,
            mem_npairs_unique,
            cub::Sum(),
            npairs);

    timer.event("transfer device to host");
    int npairs_unique;
    CUDA_CALL(cudaMemcpy, &npairs_unique, mem_npairs_unique, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SYNC_STREAM(stream);


    // ------

    float *mem_scores = allocator.allocate<float>(npairs_unique);
    int *mem_selected_labels = allocator.allocate<int>(npairs_unique);
    int2 *mem_selected_partners = allocator.allocate<int2>(npairs_unique);
    int *mem_nselected_labels = allocator.allocate<int>(1);

    timer.event("calculate improvements");

    rule.score_merge(stream, make_cuda_graph(), make_cuda_partition(),
            npairs_unique,
            mem_pairs_unique,
            mem_counts,
            mem_scores);

    timer.event("find best partner");

    CUB_CALL(DeviceReduce::ReduceByKey,
            make_transform_iterator(
                mem_pairs_unique,
                select_int2_y()
            ),
            mem_selected_labels,
            make_transform_iterator(
                make_zip_iterator(
                    make_transform_iterator(
                        mem_pairs_unique,
                        select_int2_x()
                    ),
                    mem_scores
                ),
                make_best_partner()
            ),
            mem_selected_partners,
            mem_nselected_labels,
            reduce_best_partner(),
            npairs_unique);


    timer.event("transfer device to host");
    int nselected_labels;
    CUDA_CALL(cudaMemcpy, &nselected_labels, mem_nselected_labels, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SYNC_STREAM(stream);

    // ------

    int *mem_partners = allocator.allocate<int>(num_comms);
    bool *mem_has_partner = allocator.allocate<bool>(num_comms);

    CUDA_FILL(stream, mem_has_partner, num_comms, false);
    CUDA_FILL(stream, mem_partners, num_comms, INVALID_LABEL);

    CUDA_LAUNCH(stream, nselected_labels, GROUP_SIZE,
            ker_set_best_partner,
            nselected_labels,
            mem_selected_labels,
            mem_selected_partners,
            mem_partners);

    CUDA_LAUNCH(stream, num_comms, GROUP_SIZE,
            ker_set_has_partner,
            num_comms,
            mem_partners,
            mem_has_partner);

    CUDA_LAUNCH(stream, n, GROUP_SIZE,
            ker_relabel,
            n,
            mem_labels,
            mem_partners,
            mem_has_partner,
            mem_new_labels);

    timer.stop();

    swap(mem_labels, mem_new_labels);
    return true;
}
