#!/bin/sh
# Script for running the most relevant variations
# of this implementation. This is non-/cuBLAS and Gaussian/T-Distribution weights.

#../data/rgbd_dataset_freiburg1_desk
#../data/rgbd_dataset_freiburg1_desk2

MYPATH="../data/rgbd_dataset_freiburg1_desk" &&

make clean && make all &&

./ludicrous_non_cublas -path $MYPATH -tDistWeights 0 > run_all_out.txt &&

python ../../benchmark_tools/evaluate_rpe.py --fixed_delta --plot non_cublas_no_weights_rpe.png $MYPATH"/groundtruth.txt" $MYPATH"_gdist_nocublas_trajectory.txt" >> run_all_out.txt &&
python ../../benchmark_tools/evaluate_ate.py --plot non_cublas_no_weights_ate.png $MYPATH"/groundtruth.txt" $MYPATH"_gdist_nocublas_trajectory.txt" >> run_all_out.txt &&

./ludicrous_non_cublas -path $MYPATH -tDistWeights 1  >> run_all_out.txt &&

python ../../benchmark_tools/evaluate_rpe.py --fixed_delta --plot non_cublas_td-weights_rpe.png $MYPATH"/groundtruth.txt" $MYPATH"_tdist_nocublas_trajectory.txt" >> run_all_out.txt &&
python ../../benchmark_tools/evaluate_ate.py --plot non_cublas_td-weights_ate.png $MYPATH"/groundtruth.txt" $MYPATH"_tdist_nocublas_trajectory.txt" >> run_all_out.txt &&

./ludicrous_cublas -path $MYPATH -tDistWeights 0 >> run_all_out.txt &&

python ../../benchmark_tools/evaluate_rpe.py --fixed_delta --plot cublas_no_weights_rpe.png $MYPATH"/groundtruth.txt" $MYPATH"_gdist_nocublas_trajectory.txt" >> run_all_out.txt &&
python ../../benchmark_tools/evaluate_ate.py --plot cublas_no_weights_ate.png $MYPATH"/groundtruth.txt" $MYPATH"_gdist_nocublas_trajectory.txt" >> run_all_out.txt &&

./ludicrous_cublas -path $MYPATH -tDistWeights 1  >> run_all_out.txt &&

python ../../benchmark_tools/evaluate_rpe.py --fixed_delta --plot cublas_td-weights_rpe.png $MYPATH"/groundtruth.txt" $MYPATH"_tdist_nocublas_trajectory.txt" >> run_all_out.txt &&
python ../../benchmark_tools/evaluate_ate.py --plot cublas_td-weights_ate.png $MYPATH"/groundtruth.txt" $MYPATH"_tdist_nocublas_trajectory.txt" >> run_all_out.txt
