#!/bin/sh

make clean && make all &&

./ludicrous_non_cublas -path ../data/rgbd_dataset_freiburg1_desk2 -tDistWeights 0 > run_all_out.txt &&

./ludicrous_non_cublas -path ../data/rgbd_dataset_freiburg1_desk2 -tDistWeights 1  >> run_all_out.txt &&

./ludicrous_cublas -path ../data/rgbd_dataset_freiburg1_desk2 -tDistWeights 0 >> run_all_out.txt &&

./ludicrous_cublas -path ../data/rgbd_dataset_freiburg1_desk2 -tDistWeights 1  >> run_all_out.txt
