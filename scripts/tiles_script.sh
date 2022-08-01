#!/bin/bash
touch tiles_result.txt && echo 'Tiles;Total_time;Assemble_time;Cholesky_time;Triangular_time;Predict_time;Error;N_train;N_test;N_regressor;Algorithm' >> cores_result.txt
START=$1
END=$2
STEP=$3
N_TRAIN=$4
N_TEST=$5
N_REG=$6
N_CHOLESKY=$7
for (( i=$START; i<= $END; i=i+$STEP ))
do
    cd ../build && ./hpx_cholesky --n_train $N_TRAIN --n_test $N_TEST --n_regressors $N_REG --n_tiles $i --cholesky $N_CHOLESKY
done
