import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
import os
from deform_model import *
import math
import copy

class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image_path: Path,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        args = None,
    ):
        self.device = torch.device("cuda:0")
        self.image_list = self._load_images(image_path)  # Load all images
        self.original_image_list = self.image_list.copy()
        self.gt_image = self.image_list[0]

        self.num_points = num_points
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.save_imgs = args.save_imgs
        self.log_dir = Path(f"./checkpoints/{args.data_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}")
        
        if model_name == "GaussianImage_Cholesky":
            from gaussianimage_cholesky import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device)
            self.deform = DeformModel()
            self.deform.train_setting()
            
        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def _load_images(self, image_paths):
        gt_images = []
        for path in sorted(os.listdir(image_paths)):  # 按文件名排序
            path_full = os.path.join(image_paths, path)
            image_tensor = image_path_to_tensor(path_full).to(self.device)  # 转换为 Tensor 并移动到设备
            gt_images.append(image_tensor)
        return gt_images

    
    def _pop_random_image(self):
        if len(self.image_list) == 0:
            # print("Image list exhausted! Reloading...")
            self.image_list = list(self.original_image_list)
        img = self.image_list.pop(0)
        timestamp = 1.0 - len(self.image_list) / len(self.original_image_list)
        return img, timestamp  


    def train(self, stage):     
        psnr_list, iter_list = [], []
        if stage == "coarse":
            iterations = 10000
        elif stage == "fine":
            iterations = self.iterations
        progress_bar = tqdm(range(1, iterations+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        for iter in range(1, iterations+1):
            if stage == "coarse":
                image, timestamp = self._pop_random_image()
                gt_image = image
                # gt_image=self.image_list[0]
                render_pkg = self.gaussian_model.forward()
            elif stage == "fine":
                # if iter < 20000:
                #     self.gaussian_model._xyz.requires_grad = False
                image, timestamp = self._pop_random_image()
                gt_image = image
                if iter > 30000:
                    deformed_xyz, deformed_opacity, deformed_features = self.deform.step(self.gaussian_model.get_xyz,timestamp)
                    render_pkg = self.gaussian_model.forward_prune(deformed_xyz, deformed_opacity, deformed_features)
                else:
                    deformed_xyz, deformed_opacity, deformed_features = self.deform.step(self.gaussian_model.get_xyz,timestamp)
                    render_pkg = self.gaussian_model.forward_dynamic(deformed_xyz, deformed_opacity, deformed_features)
                
            image = render_pkg["render"]
            loss_n = torch.mean(torch.sigmoid(self.gaussian_model.m))
            loss = loss_fn(image, gt_image, self.gaussian_model.loss_type, lambda_value=0.8)
            if stage == "fine":
                if iter > 30000 and iter < 65000:
                    if iter % 15 == 0:
                        loss=loss+5e-3*loss_n
                # if iter == 60000:
                #     self.gaussian_model.apply_mask()
                     
            loss.backward()        
            with torch.no_grad():
                mse_loss = F.mse_loss(image, gt_image)
                psnr = 10 * math.log10(1.0 / mse_loss.item())
            self.gaussian_model.optimizer.step()
            self.gaussian_model.optimizer.zero_grad(set_to_none = True)
            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
            self.gaussian_model.scheduler.step()
            self.deform.update_learning_rate(iter)
        
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
                if iter % 5000 == 0:
                    psnr_value, ms_ssim_value = self.test()
                    self.save_model_parameters()
                    mask = torch.sigmoid(self.gaussian_model.m) > self.gaussian_model.epsilon  # 生成二值mask
                    count_mask_1 = mask.sum()  # 统计mask中值为1的元素数量
                    print(f"Number of elements with mask = 1: {count_mask_1.item()}")
        end_time = time.time() - start_time
        progress_bar.close()
        # psnr_value, ms_ssim_value = self.test()
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model()
            test_end_time = (time.time() - test_start_time)/100
            
        return psnr_value, ms_ssim_value

    def test(self):
        """Evaluate the model on all video frames."""
        self.gaussian_model.eval()
        total_psnr, total_ms_ssim = 0, 0
        num_frames = len(self.original_image_list)

        # Reset image list to ensure full video evaluation
        self.image_list = list(self.original_image_list)

        with torch.no_grad():
            for i in range(num_frames):
                # Pop the first image and calculate timestamp
                img = self.image_list.pop(0)
                timestamp = 1.0 - len(self.image_list) / len(self.original_image_list)
                deformed_xyz, deformed_opacity, deformed_features = self.deform.step(
                    self.gaussian_model.get_xyz, torch.tensor([[timestamp]], device=self.device)
                )
                render_pkg = self.gaussian_model.forward_prune_test(
                    deformed_xyz, deformed_opacity, deformed_features
                )
                rendered_image = render_pkg["render"]
                mse_loss = F.mse_loss(rendered_image.float(), img.float())
                psnr = 10 * math.log10(1.0 / mse_loss.item())
                ms_ssim_value = ms_ssim(
                    rendered_image.float(), img.float(), data_range=1, size_average=True
                ).item()

                total_psnr += psnr
                total_ms_ssim += ms_ssim_value

                # Save rendered images if required
                if self.save_imgs:
                    transform = transforms.ToPILImage()
                    img = transform(rendered_image.float().squeeze(0))
                    name = f"test_frame_{i:04d}.png"
                    img.save("./test/" + name)

        avg_psnr = total_psnr / num_frames
        avg_ms_ssim = total_ms_ssim / num_frames

        # Log test results
        self.logwriter.write(
            "Test Average PSNR:{:.4f}, MS_SSIM:{:.6f}".format(avg_psnr, avg_ms_ssim)
        )
        return avg_psnr, avg_ms_ssim
    

    def test_prune(self, prune_ratio=0.1):
        """Evaluate the model on all video frames with pruning based on a single ratio."""
        self.gaussian_model.eval()
        total_psnr, total_ms_ssim = 0, 0
        num_frames = len(self.original_image_list)

        # Reset image list to ensure full video evaluation
        self.image_list = list(self.original_image_list)

        with torch.no_grad():
            for i in range(num_frames):
                # Pop the first image and calculate timestamp
                img = self.image_list.pop(0)
                timestamp = 1.0 - len(self.image_list) / len(self.original_image_list)
                # Get features and opacity from the original model
                features = self.gaussian_model.get_features  # (N, 3)
                opacity = self.gaussian_model.get_opacity    # (N, 1)
                combined = features * opacity  # (N, 3) since opacity is (N, 1)
                combined_norm = torch.norm(combined, dim=1)  # L2 norm across the second dimension (3)
                sorted_norms, indices = torch.sort(combined_norm)
                threshold_index = int(len(sorted_norms) * prune_ratio)
                prune_mask = torch.zeros_like(combined_norm, dtype=torch.bool)
                prune_mask[combined_norm >= sorted_norms[threshold_index]] = 1

                deformed_xyz, deformed_opacity, deformed_features = self.deform.step(
                    self.gaussian_model.get_xyz[prune_mask], torch.tensor([[timestamp]], device=self.device)
                )
                render_pkg = self.gaussian_model.forward_prune(
                    deformed_xyz, deformed_opacity, deformed_features
                )
                rendered_image = render_pkg["render"]
                mse_loss = F.mse_loss(rendered_image.float(), img.float())
                psnr = 10 * math.log10(1.0 / mse_loss.item())
                ms_ssim_value = ms_ssim(
                    rendered_image.float(), img.float(), data_range=1, size_average=True
                ).item()

                total_psnr += psnr
                total_ms_ssim += ms_ssim_value

                # Save rendered images if required
                if self.save_imgs:
                    transform = transforms.ToPILImage()
                    img = transform(rendered_image.float().squeeze(0))
                    name = f"test_frame_{i:04d}_pruned_{prune_ratio}.png"
                    img.save(f"./test/{name}")
        avg_psnr = total_psnr / num_frames
        avg_ms_ssim = total_ms_ssim / num_frames
        self.logwriter.write(
            "Test Average PSNR:{:.4f}, MS_SSIM:{:.6f}".format(avg_psnr, avg_ms_ssim)
        )
        return avg_psnr, avg_ms_ssim



    def test_fps(self):
        """Render all video frames and calculate decode FPS."""
        self.gaussian_model.eval()
        num_frames = len(self.original_image_list)
        self.image_list = list(self.original_image_list)
        start_time = time.time()  # Start timer
        with torch.no_grad():
            for i in range(num_frames):
                self.image_list.pop(0)
                timestamp = 1.0 - len(self.image_list) / len(self.original_image_list)
                deformed_xyz, deformed_opacity, deformed_features = self.deform.step(
                    self.gaussian_model._xyz, torch.tensor([[timestamp]], device=self.device)
                )
                render_pkg = self.gaussian_model.forward_dynamic(
                    deformed_xyz, deformed_opacity, deformed_features
                )

        total_time = time.time() - start_time
        decode_fps = num_frames / total_time
        print(f"Decode FPS: {decode_fps:.2f}")

        return decode_fps    
    
    
    def save_model_parameters(self):
        save_path = f"./model3/saved_parameters_{self.image_name}.npy"
        xyz = self.gaussian_model._xyz  # Shape: (N, C)
        cholesky = self.gaussian_model._cholesky  # Shape: (N, C)
        features_dc = self.gaussian_model._features_dc  # Shape: (N, C)
        opacity = self.gaussian_model._opacity  # Shape: (N, C)
        concatenated_tensor = torch.cat((xyz, cholesky, features_dc, opacity), dim=1)  # Shape: (N, 4*C)
        concatenated_array = concatenated_tensor.cpu().detach().numpy()
        np.save(save_path, concatenated_array)
        print(f"Parameters saved to {save_path}")    

def image_path_to_tensor(image_path: Path):
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/bunny/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='bunny', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="Learning rate (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./checkpoints/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    # psnrs, ms_ssims, training_times, eval_times, eval_fpses = [], [], [], [], []
    image_path = Path(args.dataset)

    trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
        iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path)
    print("Start coarse training!")
    trainer.train(stage="coarse")
    print("Start fine training!")
    psnr,ssim = trainer.train(stage="fine")    
    print(f"Segment: {args.data_name}, PSNR: {psnr:.4f}, SSIM: {ssim:.6f}") 
    return psnr, ssim   



  

if __name__ == "__main__":
    main(sys.argv[1:])
