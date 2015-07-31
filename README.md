Par-CD: Flexible, Fast and Parallel Community Detection
===
This repository provides the implementation of the work done for Par-CD [1]. Par-CD is a flexible framework for parallel community detection in large network on multi-core CPUs (C++) and NVIDIA GPUs (CUDA).

Compilation
===
The following packages are required for compilation (in parentheses are the recommended versions):

* GNU Compiler Collection (`gcc` 4.8.2), other C++ compilers might work
* GNU Make (3.81)

Run `make main convert` in `src` to compile the source files

```
make main convert
```

For compiling the GPU implementation, the following additional packages are required:

* NVIDIA Cuda Compiler (`nvcc` 5.5)
* [Modern GPU Library](https://github.com/NVlabs/moderngpu) by NVlabs
* [CUB](http://nvlabs.github.io/cub/index.html) by NVIDIA Research (1.4.1)

Download Modern GPU and CUB from the provided websites and modify the paths in `src/config.mk` to refer to the root directory of both libraries.

```
MGPU_PATH=<path to Modern GPU>
CUB_PATH=<path to CUB>
```

Next, run `make main-cuda` in `src` to compile the GPU implementation.

```
make main-cuda
```


Usage
===
To use Par-CD, one should first convert a graph file to a binary format using the `convert` program.

```
./convert [text input file] [binary output file]
```

The text file should contain one edge per line where each edge consists of a pair of two numbers. The graph is assumed to be undirected so the order of the endpoints is irrelevant. Duplicated edges, loops, empty lines and lines starting with a `#` are ignored. An example of a valid file is shown below.

```
1 4
4 3
3 5
1 5
```

To run the C++ implementation, run the following command:

```
./main [binary file]
```

The metric to be optimized can be set with `-m`, the number of threads can be set with `-p` and the schedule of the refinement phase can be set with `-s`. For additional flags and options, see `-h`.


To run the CUDA implementation, run the following command:

```
./main-cuda [binary file]
```

Options and flags can be found by using the `-h` flags. Note that the CUDA implementation is a simplified version of the C++ implementation and not all options are currently supported.


License
===
This software is licensed under the GNU GPL v3.0.

Bibliography
====
[1] Stijn Heldens, Henri E. Ball, A. L. Varbanescu. (2015), "Par-CD: A Flexible Framework for Parallel Community Detection in Large Networks", MSc thesis, VU University Amsterdam, The Netherlands.

