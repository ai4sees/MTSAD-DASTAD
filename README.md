# DASTAD: Dual Aspect Self-supervised Transformer based Anomaly Detection in Multivariate Time Series

This repository contains code of our paper "DASTAD: Dual Aspect Self-supervised Transformer based Anomaly Detection in Multivariate Time Series". Follow the below steps to replicate each cell in the results table. The code is provided as-is.

## Repository Structure
```bash
    root
    ├── main.py
    ├── requirements.txt
    ├── processed   
        ├── MSL
        ├── SMAP
        ├── SMD
        ├── SWaT
        └── WADI
    └── src
        ├── constants.py
        ├── diagnosis.py
        ├── dlutils.py
        ├── folderconstants.py
        ├── models.py
        ├── params.json
        ├── parser.py
        ├── pot.py
        ├── spot.py
        └── utils.py
```

## Installation
This code needs Python-3.7 or higher.
```bash
pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
```


## Result Reproduction
To run a model on a dataset, run the following command:
```bash
python3 main.py --model DASTAD --dataset <dataset> --retrain
```
where `<dataset>` can be one of 'SMAP', 'MSL', 'SWaT', 'WADI', 'SMD'. To train with 20% data, use the following command 
```bash
python3 main.py --model DASTAD --dataset <dataset> --retrain --less
```
You can use the parameters in `src/params.json` to set values in `src/constants.py` for each file. 


The output will provide anomaly detection and diagnosis scores and training time. For example:
```bash
$ python3 main.py --model DASTAD --dataset MSL --retrain 
Using backend: pytorch
Creating new model: DASTAD
Training TranAD on MSL
Epoch 0,        L1 = 0.09839354782306504
Epoch 1,        L1 = 0.039524692888342115
Epoch 2,        L1 = 0.022258711623482686
Epoch 3,        L1 = 0.01833707226553135
Epoch 4,        L1 = 0.016330517334598792
100%|███████████████████████████████████████████████████████████████████| 5/5 [00:03<00:00,  1.57it/s]
Training time:     3.1920 s
Testing TranAD on MSL
{'FN': 0,
 'FP': 11,
 'Hit@100%': 1.0,
 'Hit@150%': 1.0,
 'NDCG@100%': 1.0000000000000002,
 'NDCG@150%': 1.0000000000000002,
 'ROC/AUC': 0.9973249027237354,
 'TN': 2045,
 'TP': 100,
 'f1': 0.9478622223482878,
 'precision': 0.9009008197386649,
 'recall': 0.9999999000000099,
 'threshold': 0.018906979033661963}
```

All outputs can be run multiple times to ensure statistical significance. 



