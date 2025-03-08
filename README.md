# **Deformable 2D Gaussian Splatting for Video Representation at 400FPS**

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
Before running the experiments, you need to download the **Kodak** and **DIV2K-validation** datasets.  
Ensure that the dataset is structured correctly:

```
datasets/
│── kodak/
│   ├── image1.png
│   ├── image2.png
│   ├── ...
│── div2k_validation/
│   ├── 0001.png
│   ├── 0002.png
│   ├── ...
```

- **Kodak dataset**: Download from [Kodak Image Suite](http://r0k.us/graphics/kodak/)
- **DIV2K-validation dataset**: Download from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

After downloading, place the dataset inside the `datasets/` directory before proceeding with training or evaluation.

## **Startup**
To run **Deformed 2D Gaussian Splatting (2DGS)**, execute the following command:

```
sh ./scripts/gaussianimage_cholesky/bunny.sh /path/to/your/dataset
```

To run **Deformed 3D Gaussian Splatting (3DGS)**, use:

```
sh ./scripts/3dgs/kodak.sh /path/to/your/dataset
```

Make sure to replace `/path/to/your/dataset` with the actual dataset path.

## **Training**
To train the model, use the following command:

```
python train.py --config configs/train_config.yaml
```

Modify the configuration file (`configs/train_config.yaml`) to adjust training parameters as needed.

## **Evaluation**
To evaluate the trained model:

```
python evaluate.py --checkpoint path/to/checkpoint.pth --dataset datasets/kodak
```

Replace `path/to/checkpoint.pth` with the actual path to the trained checkpoint.

## **Inference**
To run inference on a single image:

```
python infer.py --input path/to/image.png --output path/to/output.png
```

## **Citation**
If you use this code, please cite our work:

```
@article{your_paper,
  title={Deformable 2D Gaussian Splatting for Video Representation at 400FPS},
  author={Your Name and Others},
  journal={Arxiv},
  year={2025}
}
```

## **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
