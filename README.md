# SpikeDHS
This code is a demo of our NeurIPS 2022 paper (Spotlight) "Differentiable hierarchical and surrogate gradient search for spiking neural networks".

# Dataset
To proceed, please download the CIFAR10/100 dataset on your own.

# Environment
```
1. Python 3.8.*
2. CUDA 10.0
3. PyTorch 
4. TorchVision 
5. fitlog
```

# Install
Create a  virtual environment and activate it.
```shell
conda create -n SpikeDHS python=3.8
conda activate SpikeDHS
```
The code has been tested with PyTorch 1.6 and Cuda 10.2.
```shell
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2 -c pytorch
conda install matplotlib path.py tqdm
conda install tensorboard tensorboardX
conda install scipy scikit-image opencv
```

# Spikingjelly Installation 
Our project can also be developed by SpikingJelly. (ref: https://github.com/fangwei123456/spikingjelly)
```shell
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
python setup.py install
```

# Cofe for SpikeDHS
We provide search, decode and retrain code for CIFAR10/100.

## Search
For search procedure, execute: \
  `bash search.sh`

Once we have conducted a search, the next step is to decode the results in order to retrieve the searched architecture.

## Decode
For decode, execute: \
  `bash decode.sh`
  
Searched Architecture:
```bash
network_path_fea = [0,0,1,1,1,2,2,2] # default
cell_arch_fea = [[1, 1],
                    [0, 1],
                    [3, 2],
                    [2, 1],
                    [7, 1],
                    [8, 1]]
```
Replace the searched architecture in `LEAStereo.py`.

## Retrain
For retrain procedure, execute: \
  `bash train.sh`
  
# Paper Reference
```
@inproceedings{chedifferentiable,
  title={Differentiable hierarchical and surrogate gradient search for spiking neural networks},
  author={Che, Kaiwei and Leng, Luziwei and Zhang, Kaixuan and Zhang, Jianguo and Meng, Qinghu and Cheng, Jie and Guo, Qinghai and Liao, Jianxing},
  booktitle={Advances in Neural Information Processing Systems}
}
```

Our code is developed based on the code from papers "Hierarchical Neural Architecture Searchfor Deep Stereo Matching" and "Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation"  
code:  
https://github.com/XuelianCheng/LEAStereo  
https://github.com/NoamRosenberg/autodeeplab  
https://github.com/fangwei123456/spikingjelly


