# County Augmented Transformer for COVID-19 State Hospitalizations Prediction
This repository contains the accompanying software for the paper "County Augmented Transformer for COVID-19 State Hospitalizations Prediction" by Siawpeng Er, Shihao Yang and Tuo Zhao.

### Usage
There are three models (CAT, WR, STATE model) with their corresponding setting used in this paper. To run each model, use their corresponding scripts in the script folder.

1. For CAT with default configurations
```shell
./script/train_transformer_hospitalization.sh [GPU location]
```
2. For WR with default configurations
```shell
./script/train_transformer_hospitalization_wr.sh [GPU location]
```
3. For STATE with default configurations
```shell
./script/train_transformer_hospitalization_wr_state.sh [GPU location]
```

