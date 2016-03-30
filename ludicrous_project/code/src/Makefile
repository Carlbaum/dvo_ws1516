main: main.cu helper.cu helper.h Makefile
	nvcc --std=c++11 -o ludicrous main.cu helper.cu -I../third_party/include --ptxas-options=-v --use_fast_math --compiler-options -Wall -lopencv_highgui -lopencv_core

