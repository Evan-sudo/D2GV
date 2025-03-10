# **Deformable 2D Gaussian Splatting for Video Representation at 400FPS**
Note: This repo is still under development.
## **Requirements**
Before running the code, install the required dependencies:

```
cd gsplat
pip install .[dev]
cd ../
pip install -r requirements.txt
```

The `gsplat` module is implemented based on [GaussianImage](https://github.com/Xinjie-Q/GaussianImage). Please ensure it is correctly installed before proceeding.

## **Dataset**
Before running the experiments, you need to download the **UVG** and **Davis** datasets.  
Ensure that the dataset is structured correctly:

```
dataset/
│── Bunny/
│   ├── bunny_1/
│   │   ├── f00001.png
│   │   ├── f00002.png
│   │   ├── f00003.png
│   │   ├── ...
│   ├── bunny_2/
│   │   ├── f00001.png
│   │   ├── f00002.png
│   │   ├── f00003.png
│── UVG/
```

- **UVG dataset**: Download from [UVG Dataset](https://ultravideo.fi/dataset.html)

After downloading, place the dataset inside the `datasets/` directory before proceeding with training or evaluation.

## **Startup**
To run **Deformed 2D Gaussian Splatting (2DGS)**, execute the following command:

```
sh ./scripts/gaussianimage_cholesky/bunny.sh /path/to/your/dataset
```

To run **Deformed 3D Gaussian Splatting (3DGS)**, use:

```
sh ./scripts/3dgs/bunny.sh /path/to/your/dataset
```

Make sure to replace `/path/to/your/dataset` with the actual dataset path.




