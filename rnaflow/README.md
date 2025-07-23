# Blank Title

Here record different aspects of the RNAFlow project, including data visualization, model training, and evaluation.

## Cell Anaylysis Part

### Dataset

For dataset labeling, please refer to the [Cell Dataset README](./cell/dataset/README.md).

## RNA Transcription Site Analysis Part

This part contains the code for the RNA transcription site analysis pipeline.

Please make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install trackpy tifffile SimpleITK scikit-learn
```

`torch` is also required, please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific environment.

### Data Format

This pipeline expects the input data to be in the following format:

- A folder containing TIFF files of the cell sequence images.

TIFF files should be named in a way that indicates the cell id, such as `cellraw_0001.tif`, `cellraw_0002.tif`, etc. The images should be grayscale and have a consistent resolution. In this pipeline, we using the fixed resolution of `128*128` pixels for each cell image. 

### Usage

**Note**: We design different tracking methods for cell sequences with different transcription sites.

- For cell sequence which at most has one transcription site, we link sites in consecutive and constrained frames into *patch*, 
    and then link patches into *trajectory* based on the distance between the patches.

- For cell sequence which has 2 transcription sites, we use the cluster method to assign the transcription sites to different clusters.

- **IMPORTANT!!**  All site tracking are done in the registration space, which is the space after the registration of the cell sequence.

```python
import shutil
from os.path import join
from pathlib import Path
import torch

from rnaflow.site.pipeline.predictor import SitePredictor

cell_seq_path = Path('path/to/cell_sequence_folder/tiff_file')
cell_seq_data_dir = cell_seq_path.parent / cell_seq_path.stem
cell_seq_data_dir.mkdir(exist_ok=True)
shutil.copy(cell_seq_path, cell_seq_data_dir, follow_symlinks=True)

site_predictor = SitePredictor(
    cell_seq_data_dir, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

model_dir = 'path/to/pt'
# Run the site detection and registration
spotlearn_model_path = join(model_dir,'spotlearn/epoch40.pt')
site_predictor.site_detect(spotlearn_model_path)
site_predictor.registration_recursive()
# Get the coordinates of the transcription site 
rf_classifier_path = join(model_dir,'rf_classifier/random_forest_model.pkl')
nn_classifier_path = join(model_dir,'nn_classifier/tut1-model.pt')
site_predictor.get_mask_coor_reg(
    rf_classifier_path=rf_classifier_path,
    nn_classifier_path=nn_classifier_path,
)

# PAY ATTENTION:
# Choose the appropriate tracking method based on the transcription sites in the cell sequence.
# For cell sequence which at most has one transcription site
site_predictor.site_track()
# For cell sequence which has 2 transcription sites
site_predictor.site_cluster()

# Compute the intensity of the transcription site
have_two_sites = False  # Set to True if the cell sequence has 2 transcription sites
site_predictor.compute_intensity(site2=have_two_sites)
# Plot the raw stack with tracked sites coordinates
site_predictor.get_raw_stack_with_label()
```

## Data Visualization 

You can visualize the cell data and site data with the code under `visualize/` folder.

The default result moive format is `mp4`. However, it can not be opened by ImageJ. You can using `ffmpeg` to convert it to a format that ImageJ can read. For this issue, you can refer to [AVI unsupported compression error](https://forum.image.sc/t/avi-unsupported-compression-error/4008)

To check if your `ffmpeg` is installed, you can run the following command:

```bash
ffmpeg -version
```

To make it readble by ImageJ, you need to convert it to `avi` format with the following command:

```bash
ffmpeg -i myInputFile.mp4 -pix_fmt nv12 -f avi -vcodec rawvideo convertedFile.avi
```

- `myInputFile.mp4` is the input file you want to convert.

- `convertedFile.avi` is the output file you want to create.