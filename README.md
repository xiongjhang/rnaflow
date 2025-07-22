# RNA-Flow

Code for RNA Transcription Kinetics Analysis, including cell segmentaiton, cell tracking, site detection, single-cell time-series images registration, site linking and site instensity computation.

### Todo List

#### Cell Part

- [x] release pipeline code

    - [x] improve the code, test pipeline on dataset

- [x] support micro-sam 

- [x] support cellpose-sam

- [x] support more advanced track methods

    - [x] add cell-track-gnn, trackastra

- [ ] combine sam-based detection method with sam2 for instance tracking

- [ ] annotate an LLS cell segmentation and tracking dataset

- [ ] Fine-tune segmentation model

#### RNA Transcription Site Part

- [x] release pipeline code

- [ ] support 3d(time) site detection (for reduce FP)

      
## Installation

### env `rnaflow`

```bash
conda create --name rnaflow python=3.10
conda activate rnaflow
# for cellpose-sam
python -m pip install cellpose notebook matplotlib
# for trackastra
conda install -c conda-forge -c gurobi -c funkelab ilpy
pip install "trackastra[ilp]"
# ultralytics
pip install ultralytics

# for micro-sam
conda install -c conda-forge micro_sam

git clone https://github.com/xiongjhang/rnaflow.git
cd rnaflow
pip install -e .
```

**Notes/Troubleshooting**

1. `pip install "trackastra[ilp]` raise `src/pyscipopt/scip.c:1146:10: fatal error: scip/type_retcode.h: No such file or directory`

    这个错误是由于安装 pyscipopt 时缺少 SCIP 优化套件的依赖导致的
    
    **Solution:**  `conda install -c conda-forge scip`

### env `micro-sam`
Please refer to the [micro-sam](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#installation) for installation instructions.

```bash
conda create -c conda-forge -n micro-sam micro_sam
conda activate micro-sam
conda install -c conda-forge micro_sam "libtorch=*=cuda11*"  # base on your cuda version

git clone https://github.com/xiongjhang/rnaflow.git
cd rnaflow
pip install -e .
```

### env `cell-tracker-gnn`

Please refer to the [cell-tracker-gnn](https://github.com/talbenha/cell-tracker-gnn#set-up-conda-virtual-environment) for installation instructions.

```bash
# Enter to the code folder
cd cell-tracker-gnn

# create conda environment python=3.8 pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 faiss-gpu pytorch-lightning==1.4.9
conda create --name cell-tracker-gnn --file requirements-conda.txt
conda activate cell-tracker-gnn

# install other requirements
pip install -r requirements.txt

git clone https://github.com/xiongjhang/rnaflow.git
cd rnaflow
pip install -e .
```

## Usage/Examples

TODO


## License

MIT
