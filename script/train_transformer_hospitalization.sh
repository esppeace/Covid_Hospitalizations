#!/usr/bin/env bash
if [[ $# -ne 1 ]]; then
    GPUID=0
else
    GPUID=$1
fi
echo "Run on GPU $GPUID"

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
echo $PROJECT_ROOT
STATE_DATA_ROOT=$PROJECT_ROOT/dataset/CDC_2021_Jan/
COUNTY_DATA_ROOT=$PROJECT_ROOT/dataset/CDC_2021_Jan/
echo $DATA_ROOT

# model
MODEL_NAME=transformer
#MODEL_NAME=pop_biLSTM

# parameters
SEED=1
EPOCH=500
LR=7.5e-4
WD=1e-4
DROPOUT=0

#HIDDEN_DIM=64
BATCH_SIZE=512
TRAIN_SIZE=0.8
LOSS=huber

NUM_ENCODER_LAYER=1
MODEL_DIM=8
NUM_HEAD=8
FEEDFORWARD_DIM=16

NUM_WORKERS=4
HUBER_BETA=1.0
GAMMA=0.5
SIZE=7 # how many days for X

NUM_OF_WEEK=4
HALF_EPOCH=250
ACT=relu
LAMBDA1=3
RM=50

now=$(date +'%m_%d_%Y')

OUTPUT=$PROJECT_ROOT/outputs/${now}/CAT/${TRAIN_SIZE}/

echo $OUTPUT
[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
if [ -e $OUTPUT/tfboard ]; then
    rm -rf $OUTPUT/tfboard
fi
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'script' --exclude 'dataset' --exclude 'pretrained_model' --exclude 'backup' --exclude 'outputs' --exclude 'misc' $PROJECT_ROOT/ $OUTPUT/src

CUBLAS_WORKSPACE_CONFIG=:4096:2 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID \
python3 -u train_hospitalization.py \
    --state_data_dir $STATE_DATA_ROOT \
    --county_data_dir $COUNTY_DATA_ROOT \
    --state_dir 'states_economy_complete' \
    --counties_dir 'counties_economy_complete' \
    --use_hospitalization_smoothing \
    --use_hospitalization \
    --use_how_many_days ${SIZE} \
    --use_cases \
    --use_deaths \
    --use_economy \
    --num_of_week ${NUM_OF_WEEK} \
    --hospitalization_prediction_list 'hospitalization_total' \
    --use_smoothing_for_train \
    --seed ${SEED} \
    --model-dir ${OUTPUT} \
    --activation ${ACT} \
    --shuffle \
    --model ${MODEL_NAME} \
    --epochs ${EPOCH} \
    --huber_beta ${HUBER_BETA} \
    --standardize \
    --train_size ${TRAIN_SIZE} \
    --lambda1 ${LAMBDA1} \
    --gamma ${GAMMA} \
    --loss ${LOSS} \
    --county_remove_day ${RM} \
    --state_remove_day ${RM} \
    --split_train_val \
    --num_workers ${NUM_WORKERS} \
    --learning_rate ${LR} \
    --weight_decay ${WD} \
    --feedforward_dim ${FEEDFORWARD_DIM} \
    --num_encoder_layer ${NUM_ENCODER_LAYER} \
    --num_head ${NUM_HEAD} \
    --model_dim ${MODEL_DIM} \
    --dropout ${DROPOUT} \
    --seq_len ${SIZE} \
    --half_epoch ${HALF_EPOCH} \
    --batch_size ${BATCH_SIZE} | tee $OUTPUT/${MODEL_NAME}.out



