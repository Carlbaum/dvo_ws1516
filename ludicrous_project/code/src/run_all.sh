#!/bin/sh
# Script for running the most relevant variations
# of this implementation. This is non-/cuBLAS and Gaussian/T-Distribution weights.
#
# Give in as first positional argument the name of the dataset to run
# e.g.:    ./run_all.sh rgbd_dataset_freiburg1_desk
#
# Before the corresponding dataset has to be downloaded and uncompressed in the ../data folder
# The corresponding K.txt with the intrinsic parameters has to be manually included too
# All of this can be found on the TUM Benchmark website 

DATAPATH="../data/"$1 &&
OUTPATH="../../results/"$1 &&
mkdir $OUTPATH

echo "Running non_cublas no weights" &&
./ludicrous_non_cublas -path $DATAPATH -tDistWeights 0 > $OUTPATH"/run_all_output.txt" &&
echo "Running non_cublas with weights" &&
./ludicrous_non_cublas -path $DATAPATH -tDistWeights 1  >> $OUTPATH"/run_all_output.txt" &&
echo "Running cublas no weights" &&
./ludicrous_cublas -path $DATAPATH -tDistWeights 0 >> $OUTPATH"/run_all_output.txt" &&
echo "Running cublas with weights" &&
./ludicrous_cublas -path $DATAPATH -tDistWeights 1  >> $OUTPATH"/run_all_output.txt" &&

cp "$DATAPATH"/*_trajectory.txt "$OUTPATH" &&

echo "*******************************************************" >> $OUTPATH"/run_all_output.txt" &&
echo "*********** PYTHON EVALUATION TOOL RESULTS ************" >> $OUTPATH"/run_all_output.txt" &&
echo "*******************************************************" >> $OUTPATH"/run_all_output.txt" &&
echo "" >> $OUTPATH"/run_all_output.txt" &&

echo "*******************************************************" >> $OUTPATH"/run_all_output.txt" &&
echo "no cublas no weigths RPE:" >> $OUTPATH"/run_all_output.txt" &&
python ../../benchmark_tools/evaluate_rpe.py --verbose  --fixed_delta --plot $OUTPATH"/non_cublas_no_weights_rpe.png" $DATAPATH"/groundtruth.txt" $OUTPATH"/gdist_nocublas_trajectory.txt" >> $OUTPATH"/run_all_output.txt" &&
echo "*******************************************************" >> $OUTPATH"/run_all_output.txt" &&
echo "no cublas no weigths ATE:" >> $OUTPATH"/run_all_output.txt" &&
python ../../benchmark_tools/evaluate_ate.py --verbose  --plot $OUTPATH"/non_cublas_no_weights_ate.png" $DATAPATH"/groundtruth.txt" $OUTPATH"/gdist_nocublas_trajectory.txt" >> $OUTPATH"/run_all_output.txt" &&

echo "*******************************************************" >> $OUTPATH"/run_all_output.txt" &&
echo "no cublas no weigths RPE:" >> $OUTPATH"/run_all_output.txt" &&
python ../../benchmark_tools/evaluate_rpe.py --verbose  --fixed_delta --plot $OUTPATH"/non_cublas_td-weights_rpe.png" $DATAPATH"/groundtruth.txt" $OUTPATH"/tdist_nocublas_trajectory.txt" >> $OUTPATH"/run_all_output.txt" &&
echo "*******************************************************" >> $OUTPATH"/run_all_output.txt" &&
echo "no cublas no weigths ATE:" >> $OUTPATH"/run_all_output.txt" &&
python ../../benchmark_tools/evaluate_ate.py --verbose  --plot $OUTPATH"/non_cublas_td-weights_ate.png" $DATAPATH"/groundtruth.txt" $OUTPATH"/tdist_nocublas_trajectory.txt" >> $OUTPATH"/run_all_output.txt" &&

echo "*******************************************************" >> $OUTPATH"/run_all_output.txt" &&
echo "no cublas no weigths RPE:" >> $OUTPATH"/run_all_output.txt" &&
python ../../benchmark_tools/evaluate_rpe.py --verbose  --fixed_delta --plot $OUTPATH"/cublas_no_weights_rpe.png" $DATAPATH"/groundtruth.txt" $OUTPATH"/gdist_cublas_trajectory.txt" >> $OUTPATH"/run_all_output.txt" &&
echo "*******************************************************" >> $OUTPATH"/run_all_output.txt" &&
echo "no cublas no weigths ATE:" >> $OUTPATH"/run_all_output.txt" &&
python ../../benchmark_tools/evaluate_ate.py --verbose  --plot $OUTPATH"/cublas_no_weights_ate.png" $DATAPATH"/groundtruth.txt" $OUTPATH"/gdist_cublas_trajectory.txt" >> $OUTPATH"/run_all_output.txt" &&

echo "*******************************************************" >> $OUTPATH"/run_all_output.txt" &&
echo "no cublas no weigths RPE:" >> $OUTPATH"/run_all_output.txt" &&
python ../../benchmark_tools/evaluate_rpe.py --verbose  --fixed_delta --plot $OUTPATH"/cublas_td-weights_rpe.png" $DATAPATH"/groundtruth.txt" $OUTPATH"/tdist_cublas_trajectory.txt" >> $OUTPATH"/run_all_output.txt" &&
echo "*******************************************************" >> $OUTPATH"/run_all_output.txt" &&
echo "no cublas no weigths ATE:" >> $OUTPATH"/run_all_output.txt" &&
python ../../benchmark_tools/evaluate_ate.py --verbose  --plot $OUTPATH"/cublas_td-weights_ate.png" $DATAPATH"/groundtruth.txt" $OUTPATH"/tdist_cublas_trajectory.txt" >> $OUTPATH"/run_all_output.txt"
