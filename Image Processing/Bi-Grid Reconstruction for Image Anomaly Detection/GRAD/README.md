# GRAD: Bi-Grid Reconstruction for Image Anomaly Detection

This repository is the official implementation of **GRAD: Bi-Grid Reconstruction for Image Anomaly Detection**. 

## Get Started 
### Environment 

**Python3.8**

**Packages**:
- torch==1.12.1+cu116
- torchvision==0.13.1+cu116

## Requirements
To create conda environment and install requirements:
```setup
conda create -n GRAD python=3.8
conda activate GRAD
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## Data preparation
1. Download the [**MVTec AD dataset**]
2. Construct the data structure as follows:
```
|-- data
    |-- MVTec-AD
        |-- mvtec_anomaly_detection
            |--bottle
            |--cable
            |-- ...
        |-- train.json
        |-- test.json
```
3. Run make_josn.py to get train.json and test.json:
```setup
cd data/MVTec-AD
python make_json.py
```

## Training
To train the model(s) in the paper, run this command:
```train
cd experiments/
bash train_torch.sh
```

## Evaluation
To evaluate a trained model, run:
```eval
cd experiments/
bash eval_torch.sh
```

## Citation
If you use this work, please cite:
```bibtex
@inproceedings{grad2025,
  title={GRAD: Bi-Grid Reconstruction for Image Anomaly Detection},
  author={Huichuan Huang, Zhiqing Zhong, Guangyu Wei, Yonghao Wan, Wenlong Sun, Aimin Feng},
  booktitle={ICME 2025},
  year={2025},
  url={https://arxiv.org/abs/2504.00609}
}
```

**Accepted at [ICME 2025](https://www.icme2025.org)** | **Preprint: [arXiv:2504.00609](https://arxiv.org/abs/2504.00609)**
