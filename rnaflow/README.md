# Blank Title

Here record different aspects of the RNAFlow project, including data visualization, model training, and evaluation.

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