# **Deformable 2D Gaussian Splatting for Video Representation at 400FPS**
Note: This repo is still under development.

![Our method achieves high-quality rendering with fast convergence and rendering speed. It delivers competitive results within just 2 minutes of training for a single group of pictures (GoP). For a fair comparison, we aggregate the total training time across all GoPs, as shown on the right, which still offers faster convergence compared to INRs. The circle size indicates the decoding FPS.](./img/overview.png)

We propose a novel video representation based on deformable 2D Gaussian splatting, dubbed D2GV, which aims to achieve three key objectives: 1) improved efficiency while delivering superior quality; 2) enhanced scalability and interpretability; and 3) increased friendliness for downstream tasks. Specifically, we initially divide the video sequence into fixed-length Groups of Pictures (GoP) to allow parallel training and linear scalability with video length. For each GoP, D2GV represents video frames by applying differentiable rasterization to 2D Gaussians, which are deformed from a canonical space into their corresponding timestamps. Notably, leveraging efficient CUDA-based rasterization, D2GV converges fast and decodes at speeds exceeding 400 FPS, while delivering quality that matches or surpasses state-of-the-art INRs. 

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
sh ./scripts/D2GV/train.sh /path/to/your/dataset
```

To run **Deformed 2D Gaussian Splatting (2DGS)** with learnable pruning, execute the following command:

```
sh ./scripts/D2GV/Comp.sh /path/to/your/dataset
```

Make sure to replace `/path/to/your/dataset` with the actual dataset path.

e.g. :

```
./scripts/D2GV/train.sh ../D2GV/dataset/bunny/bunny_1
```

For bunny, we set the number of Gaussians to be 20000 while for Davis and UVG we set it as 35000.

**Visual Quality Comparison**

![Our method achieves best quality compared to competitive INR-based and GS-based methods.](./img/vis.png)

Some source code of ours is borrowed from [GaussianImage](https://github.com/Xinjie-Q/GaussianImage) and [D-3DGS](https://github.com/ingra14m/Deformable-3D-Gaussians). We sincerely appreciate the excellent works of these authors.
