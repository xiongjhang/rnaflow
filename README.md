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
# for cellpose-sam
python -m pip install cellpose
python -m pip install notebook
python -m pip install matplotlib
# for trackastra
conda install -c conda-forge -c gurobi -c funkelab ilpy
pip install "trackastra[ilp]"
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

2. After install `micro-sam`, `import cv2` raise following error

    `ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version 'GLIBCXX_3.4.26' not found (required by /home/xiongjiahang/anaconda3/envs/rnaflow/lib/python3.10/site-packages/cv2/python-3.10/cv2.cpython-310-x86_64-linux-gnu.so)`

    **Solution:**  https://blog.csdn.net/YoJayC/article/details/120175550

3. `AssertionError: Torch not compiled with CUDA enabled`: `torch.cuda.is_available()` is `False`

**Tips**: `2.` and `3.` may be caused by `conda install -c conda-forge micro_sam`, there may be some conflicts.

Error Message:

```bash
/ By downloading and using the cuDNN conda packages, you accept the terms and conditions of the NVIDIA cuDNN EULA -
  https://docs.nvidia.com/deeplearning/cudnn/sla/index.html

| g_module_open() failed for /home/xiongjiahang/anaconda3/envs/rnaflow-test/lib/gdk-pixbuf-2.0/2.10.0/loaders/libpixbufloader-svg.so: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/xiongjiahang/anaconda3/envs/rnaflow-test/lib/gdk-pixbuf-2.0/2.10.0/loaders/../../.././libicuuc.so.75)
g_module_open() failed for /home/xiongjiahang/anaconda3/envs/rnaflow-test/lib/gdk-pixbuf-2.0/2.10.0/loaders/libpixbufloader-tiff.so: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/xiongjiahang/anaconda3/envs/rnaflow-test/lib/gdk-pixbuf-2.0/2.10.0/loaders/../../.././libLerc.so.4)

\ 
- g_module_open() failed for /home/xiongjiahang/anaconda3/envs/rnaflow-test/lib/gdk-pixbuf-2.0/2.10.0/loaders/libpixbufloader-svg.so: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/xiongjiahang/anaconda3/envs/rnaflow-test/lib/gdk-pixbuf-2.0/2.10.0/loaders/../../.././libicuuc.so.75)
g_module_open() failed for /home/xiongjiahang/anaconda3/envs/rnaflow-test/lib/gdk-pixbuf-2.0/2.10.0/loaders/libpixbufloader-tiff.so: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by /home/xiongjiahang/anaconda3/envs/rnaflow-test/lib/gdk-pixbuf-2.0/2.10.0/loaders/../../.././libLerc.so.4)
```

## Usage/Examples

TODO


## License

MIT
