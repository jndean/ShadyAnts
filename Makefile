CUDA_PATH := /usr/local/cuda-9.0

HOST_COMPILER := g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

NVCCFLAGS   := -g -G
CCFLAGS     := -g


ALL_CCFLAGS = $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

INCLUDES  := -I/usr/local/cuda/samples/common/inc
LIBRARIES = -lGL -lGLU -lglut


simulation: main.o simulation.o
	$(NVCC) -o $@ $+ $(LIBRARIES)

main.o:main.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<

simulation.o:simulation.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) -o $@ -c $<

clean:
	rm -f simulation main.o simulation.o
