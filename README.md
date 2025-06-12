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

```bash
conda create --name rnaflow python=3.10
conda activate rnaflow
python -m pip install cellpose
conda install -c conda-forge -c gurobi -c funkelab ilpy
pip install "trackastra[ilp]"
conda install -c conda-forge micro_sam

git clone https://github.com/xiongjhang/rnaflow.git
cd rnaflow
pip install -e .
```

**Notes/Troubleshooting**

- `pip install "trackastra[ilp]` raise `src/pyscipopt/scip.c:1146:10: fatal error: scip/type_retcode.h: No such file or directory`

    这个错误是由于安装 pyscipopt 时缺少 SCIP 优化套件的依赖导致的
    
    **Solution:**  `conda install -c conda-forge scip`

## Usage/Examples

TODO


## License

MIT
