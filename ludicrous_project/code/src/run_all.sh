#!/bin/sh

MYPATH="../data/rgbd_dataset_freiburg1_desk2" &&

make clean && make all &&

./ludicrous_non_cublas -path $MYPATH -tDistWeights 0 > run_all_out.txt &&

./ludicrous_non_cublas -path $MYPATH -tDistWeights 1  >> run_all_out.txt &&

./ludicrous_cublas -path $MYPATH -tDistWeights 0 >> run_all_out.txt &&

./ludicrous_cublas -path $MYPATH -tDistWeights 1  >> run_all_out.txt
