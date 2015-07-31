.PHONY: all clean

include config.mk

CC=g++
CFLAGS=-Wall -Wextra -Werror-implicit-function-declaration -Wno-unused-parameter -Wno-unused-local-typedefs -ffast-math -Winline -fopenmp -lm

NVCC=nvcc
NVFLAGS=-Xcompiler "$(CFLAGS)" -lcurand -I$(MGPU_PATH) -I$(CUB_PATH)

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS += -O2 -g3 -pg -DDEBUG -UNDEBUG
	NVFLAGS += -O2 -g -pg -DDEBUG -UNDEBUG -Xcompiler "-g3"
else
	CFLAGS += -O3 -ffast-math -fomit-frame-pointer -march=native -UDEBUG -DNDEBUG
	NVFLAGS += -O3 -Xcompiler "-ffast-math -fomit-frame-pointer -march=native" -UDEBUG -DNDEBUG $(NVARCH)
endif

BINS=main \
	 main-cuda \
	 convert

all: $(BINS) Makefile config.mk

main: graph.o common.o rules.o framework.o refinement.o
main-cuda: graph.o common.o framework.o framework-cuda.o rules-cuda.o mgpucontext.o mgpuutil.o
convert:

main convert: %: %.cpp
	$(CC) -o $@ $^ $(CFLAGS)

main-cuda: %: %.cu
	$(NVCC) -o $@ $^ $(NVFLAGS)

mgpucontext.o: $(MGPU_PATH)/src/mgpucontext.cu
mgpuutil.o: $(MGPU_PATH)/src/mgpuutil.cpp

mgpucontext.o mgpuutil.o: %.o:
	$(NVCC) -c -o $@ $^ $(NVFLAGS)

%.o: %.cu
	$(NVCC) -c -o $@ $^ $(NVFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $^ $(CFLAGS)

clean:
	rm -rf $(BINS) *.o

