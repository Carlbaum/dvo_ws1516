all: cublas noncublas

cublas: main.cu helper.cu helper.h Makefile
	nvcc --std=c++11 -g -o ludicrous_cublas main.cu helper.cu -lcublas -I../third_party/include --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core -DENABLE_CUBLAS

noncublas: main.cu helper.cu helper.h Makefile
	nvcc --std=c++11 -g -o ludicrous_non_cublas main.cu helper.cu -I../third_party/include --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

clean:
	rm -rf ludicrous_cublas ludicrous_non_cublas
