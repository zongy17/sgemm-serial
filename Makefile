# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = gcc
VEC = -O3 -fomit-frame-pointer -march=armv8-a -ffast-math #-march=armv8-a #-mtune=tsv110
MACRO = #-DSET_J_BLOCK_SIZE=$(JS) -DSET_I_BLOCK_SIZE=$(IS) -DSET_P_BLOCK_SIZE=$(PS)
OPT = $(VEC) $(MACRO)
CFLAGS = -DGETTIMEOFDAY -std=c99 $(OPT) -Wall
LDFLAGS = -Wall
# mkl is needed for blas implementation
LDLIBS = /storage/software/libs/hccSPC010B010/openblas/0.3.10/lib/libopenblas.a -lpthread -lm

targets = benchmark-test benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark-test.o benchmark.o sgemm-naive.o sgemm-blocked.o sgemm-blas.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-test : benchmark-test.o sgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-naive : benchmark.o sgemm-naive.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o sgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o sgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
