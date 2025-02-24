# MolCLR Data Augmentation

## Getting Started

### Requirements

- conda v25.11

### Installation

Set up conda environment and clone the github repo

You need Anaconda installed to be able to do this project

```bash
# create a new environment
$ conda create --name molclr python=3.7
$ conda activate molclr

# install requirements
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
$ pip install PyYAML
$ conda install -c conda-forge rdkit
$ conda install -c conda-forge tensorboard
$ conda install -c conda-forge nvidia-apex # optional

# clone the source code of MolCLR
$ git clone https://github.com/PaulHo0501/MolCLR-data-augment.git
$ cd MolCLR-data-augment
```

### Dataset

You can download the pre-training data and benchmarks used in the paper [here](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view?usp=sharing) and extract the zip file under `./data` folder. The data for pre-training can be found in `pubchem-10m-clean.txt`. All the databases for fine-tuning are saved in the folder under the benchmark name. You can also find the benchmarks from [MoleculeNet](https://moleculenet.org/).

### Running

Run the following command for pre-training (step 1)

```bash
python molclr.py
```

Run the following command for fine-tuning (step 2):

```bash
python finetune.py
```

An example of configuration is in the file `config.example.yaml`

## Acknowledgement

- MolCLR: [https://github.com/yuyangw/MolCLR/](https://github.com/yuyangw/MolCLR/)
- PyTorch implementation of SimCLR: [https://github.com/sthalles/SimCLR](https://github.com/sthalles/SimCLR)
- Strategies for Pre-training Graph Neural Networks: [https://github.com/snap-stanford/pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns)
