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

# def add_gaussian_noise(image, mean=0, std=0.045):
#     """Add Gaussian noise to an image."""
#     noise = torch.randn_like(image) * std + mean
#     noisy_image = image + noise
#     return noisy_image.clamp(0, 1)

def add_black_noise(image, prob=0.015):
    """Add black noise (set pixels to 0) to an image with a specified probability."""
    noisy_image = image.clone()

    # Black noise (set random pixels to 0)
    black_mask = torch.rand_like(image) < prob
    noisy_image[black_mask] = 0

    return noisy_image.clamp(0, 1)  # Ensure the values stay within valid image range [0, 1]



# def add_salt_and_pepper_noise(image, prob=0.015):
#     """Add salt and pepper noise to an image with a specified probability."""
#     noisy_image = image.clone()

#     # Salt noise (set random pixels to 1)
#     salt_mask = torch.rand_like(image) < prob / 2  # Half the probability for salt
#     noisy_image[salt_mask] = 1

#     # Pepper noise (set random pixels to 0)
#     pepper_mask = torch.rand_like(image) < prob / 2  # Half the probability for pepper
#     noisy_image[pepper_mask] = 0

#     return noisy_image.clamp(0, 1)  # Ensure the values stay within the valid image range [0, 1]



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
        self.image_list, self.image_n_list = self._load_images(image_path)  # Load all images
        self.original_image_n_list = self.image_n_list.copy()
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

        elif model_name == "GaussianImage_RS":
            from gaussianimage_rs import GaussianImage_RS
            self.gaussian_model = GaussianImage_RS(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=False).to(self.device) 
            self.deform = DeformModel()
            self.deform.train_setting()
            
        elif model_name == "3DGS":
            from gaussiansplatting_3d import Gaussian3D
            self.gaussian_model = Gaussian3D(loss_type="Fusion2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, sh_degree=args.sh_degree, lr=args.lr).to(self.device)
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

    # def _load_images(self, image_paths):
    #     gt_images = []
    #     for path in sorted(os.listdir(image_paths)):  # 按文件名排序
    #         path_full = os.path.join(image_paths, path)
    #         image_tensor = image_path_to_tensor(path_full).to(self.device)  # 转换为 Tensor 并移动到设备
    #         gt_images.append(image_tensor)
    #     return gt_images
    
    def _load_images(self, image_paths):
        gt_images = []
        noisy_gt_images = []  # To store noisy versions of the ground truth images

        for path in sorted(os.listdir(image_paths)):  # Sort files by name
            path_full = os.path.join(image_paths, path)
            image_tensor = image_path_to_tensor(path_full).to(self.device)  # Convert to Tensor and move to device
            
            # Generate the noisy version of the ground truth image
            noisy_gt_image = add_black_noise(image_tensor, prob=0.005)
            #noisy_gt_image = add_gaussian_noise(image_tensor, mean=0, std=0.045)  # You can adjust the std and mean
            
            gt_images.append(image_tensor)  # Store the original ground truth image
            noisy_gt_images.append(noisy_gt_image)  # Store the noisy ground truth image
        
        return gt_images, noisy_gt_images    
    

    # def _pop_random_image(self):
    #     if len(self.image_list) == 0:
    #         print("Image poped out!")
    #         self.image_list = list(self.original_image_list)
    #     index = random.randint(0, len(self.image_list) - 1)
    #     img = self.image_list.pop(index)
    #     return img, index/len(self.original_image_list)  # Return (image, timestamp)
    
    def _pop_random_image(self):
        if len(self.image_list) == 0:
            # print("Image list exhausted! Reloading...")
            self.image_list = list(self.original_image_list)
            self.image_n_list = list(self.original_image_n_list)
        img = self.image_list.pop(0)
        img_n = self.image_n_list.pop(0)
        timestamp = 1.0 - len(self.image_list) / len(self.original_image_list)
        return img, timestamp, img_n  


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
                image, timestamp, img_n = self._pop_random_image()           
                gt_image_0 = image
                
                gt_image = img_n
                render_pkg = self.gaussian_model.forward()
            elif stage == "fine":
                # self.gaussian_model._xyz.requires_grad = False
                image, timestamp, img_n = self._pop_random_image()
                gt_image_0 = image
                
                gt_image = img_n
                deformed_xyz, deformed_opacity, deformed_features = self.deform.step(self.gaussian_model.get_xyz,timestamp)

                # deformed_xyz = torch.zeros_like(self.gaussian_model._xyz)
                # deformed_cholesky = torch.zeros_like(self.gaussian_model._cholesky)
                # deformed_features = torch.zeros_like(self.gaussian_model._features_dc)
                render_pkg = self.gaussian_model.forward_dynamic(deformed_xyz, deformed_opacity, deformed_features)
                
            image = render_pkg["render"]
            loss = loss_fn(image, gt_image, self.gaussian_model.loss_type, lambda_value=0.7)  #0.8
            # if iter % 100 == 0:
            #     loss_f = compute_grid_loss(self.gaussian_model.get_xyz)
            #     loss+=1e-4*loss_f
            loss.backward()        
            with torch.no_grad():
                mse_loss = F.mse_loss(image, gt_image)
                psnr = 10 * math.log10(1.0 / mse_loss.item())
                mse_loss_n = F.mse_loss(gt_image, gt_image_0)
                psnr_n = 10 * math.log10(1.0 / mse_loss_n.item())
            # print("PSNR is:",psnr_n)

            self.gaussian_model.optimizer.step()
            self.gaussian_model.optimizer.zero_grad(set_to_none = True)
            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
            self.gaussian_model.scheduler.step()
            self.deform.update_learning_rate(iter)
            
            
            # if stage == "coarse":
            #     loss, psnr = self.gaussian_model.train_iter(gt_image=self.image_list[0],timestamp=None,stage="coarse")
            # elif stage == "fine":
            #     # Randomly select an image and timestamp
            #     self.gaussian_model._xyz.requires_grad = False
            #     image, timestamp = self._pop_random_image()
            #     loss, psnr = self.gaussian_model.train_iter(gt_image=image,timestamp=timestamp,stage="fine")
                
            psnr_list.append(psnr)
            iter_list.append(iter)
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f},"})
                    progress_bar.update(10)
                if iter % 5000 == 0:
                    psnr_value, ms_ssim_value = self.test()
                    #self.save_model_parameters()
                    decode_fps = self.test_fps()
                    print("Decoding fps:", decode_fps)
                    # self.interpolation_test()
                # if iter % 10000 == 0:
                #     self.dump_images()
                    
                    
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
        # self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        # torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        # np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        # "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time})
        # return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time
    
    

    # def test(self):
    #     self.gaussian_model.eval()
    #     with torch.no_grad():
    #         out = self.gaussian_model()
    #     mse_loss = F.mse_loss(out["render"].float(), self.gt_image.float())
    #     psnr = 10 * math.log10(1.0 / mse_loss.item())
    #     ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
    #     self.logwriter.write("Test PSNR:{:.4f}, MS_SSIM:{:.6f}".format(psnr, ms_ssim_value))
    #     if self.save_imgs:
    #         transform = transforms.ToPILImage()
    #         img = transform(out["render"].float().squeeze(0))
    #         name = self.image_name + "_fitting.png" 
    #         img.save(str(self.log_dir / name))
    #     return psnr, ms_ssim_value
    
    
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

                # Perform deformation using the deformation network
                deformed_xyz, deformed_opacity, deformed_features = self.deform.step(
                    self.gaussian_model.get_xyz, torch.tensor([[timestamp]], device=self.device)
                )

                # Forward pass through Gaussian model
                # render_pkg = self.gaussian_model.forward(
                    
                # )
                render_pkg = self.gaussian_model.forward_dynamic(
                    deformed_xyz, deformed_opacity, deformed_features
                )

                # Get the rendered image
                rendered_image = render_pkg["render"]

                # Compute metrics
                mse_loss = F.mse_loss(rendered_image.float(), img.float())
                psnr = 10 * math.log10(1.0 / mse_loss.item())
                ms_ssim_value = ms_ssim(
                    rendered_image.float(), img.float(), data_range=1, size_average=True
                ).item()

                total_psnr += psnr
                total_ms_ssim += ms_ssim_value
                
                transform = transforms.ToPILImage()
                img = transform(rendered_image.float().squeeze(0))
                name = f"test_frame_{i:04d}.png"
                img.save("./test/" + name)
                
        # Compute average PSNR and MS-SSIM
        avg_psnr = total_psnr / num_frames
        avg_ms_ssim = total_ms_ssim / num_frames
        
        # save canonical frame
        render_pkg = self.gaussian_model.forward()
        rendered_image = render_pkg["render"]
        transform = transforms.ToPILImage()
        img = transform(rendered_image.float().squeeze(0))
        name = f"test_frame_canonical.png"
        img.save("./test/" + name)
        # Log test results
        self.logwriter.write(
            "Test Average PSNR:{:.4f}, MS_SSIM:{:.6f}".format(avg_psnr, avg_ms_ssim)
        )
        return avg_psnr, avg_ms_ssim
    
    
    def dump_images(self):
        """Generate and save each frame's rendered image."""
        self.gaussian_model.eval()  # Set the model to evaluation mode
        num_frames = len(self.original_image_list)
        image_list = list(self.original_image_list)
        # Folder for saving images
        output_folder = "./test_it"
        os.makedirs(output_folder, exist_ok=True)

        with torch.no_grad():
            for i in range(num_frames):
                # Pop the first image and calculate timestamp
                img = image_list.pop(0)
                timestamp = 1.0 - len(image_list) / len(self.original_image_list)
                # Perform deformation using the deformation network
                deformed_xyz, deformed_opacity, deformed_features = self.deform.step(
                    self.gaussian_model.get_xyz, torch.tensor([[timestamp]], device=self.device)
                )

                # Forward pass through Gaussian model
                render_pkg = self.gaussian_model.forward_dynamic(
                    deformed_xyz, deformed_opacity, deformed_features
                )

                # Get the rendered image
                rendered_image = render_pkg["render"]

                # Convert the tensor to PIL image for saving
                rendered_image = transforms.ToPILImage()(rendered_image.float().squeeze(0))

                # Save the rendered image as PNG
                frame_name = f"frame_{i:04d}.png"
                rendered_image.save(os.path.join(output_folder, frame_name))

        print(f"Images saved in '{output_folder}/'.")       
    

    def test_fps(self):
        """Render all video frames and calculate decode FPS."""
        self.gaussian_model.eval()
        num_frames = len(self.original_image_list)

        # Reset image list to ensure full video evaluation
        self.image_list = list(self.original_image_list)

        start_time = time.time()  # Start timer

        with torch.no_grad():
            for i in range(num_frames):
                self.image_list.pop(0)
                timestamp = 1.0 - len(self.image_list) / len(self.original_image_list)
                deformed_xyz, deformed_opacity, deformed_features = self.deform.step(
                    self.gaussian_model.get_xyz, torch.tensor([[timestamp]], device=self.device)
                )
                render_pkg = self.gaussian_model.forward_dynamic(
                    deformed_xyz, deformed_opacity, deformed_features
                )

        total_time = time.time() - start_time


        decode_fps = num_frames / total_time
        print(f"Decode FPS: {decode_fps:.2f}")

        return decode_fps    
    
    def interpolation_test(self):
        """Generate and save only interpolated frames using midpoint timestamps."""
        self.gaussian_model.eval()
        num_frames = len(self.original_image_list)

        # Folder for saving interpolated images
        os.makedirs("./test_it", exist_ok=True)

        with torch.no_grad():
            for i in range(num_frames - 1):
                # Compute timestamps for the current and next frames
                timestamp1 = 1 - (num_frames - i) / num_frames
                timestamp2 = 1 - (num_frames - (i + 1)) / num_frames
                mid_timestamp = (timestamp1 + timestamp2) / 2.0

                # Save the interpolated image for the mid timestamp
                deformed_xyz_mid, deformed_opacity_mid, deformed_features_mid = self.deform.step(
                    self.gaussian_model.get_xyz, torch.tensor([[mid_timestamp]], device=self.device)
                )
                render_pkg_mid = self.gaussian_model.forward_dynamic(
                    deformed_xyz_mid, deformed_opacity_mid, deformed_features_mid
                )
                rendered_image_mid = render_pkg_mid["render"]
                rendered_image_mid = transforms.ToPILImage()(rendered_image_mid.float().squeeze(0))
                
                # Save with fractional index
                rendered_image_mid.save(f"./test_it/test_frame_{i + 0.5:.1f}.png")

        print("Interpolation test completed. Images saved in './test_it/'.")     
    
    
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



def compute_grid_loss(xyz, grid_size=10, region=(-1, 1, -1, 1), sigma=0.1):

    # Unpack region bounds
    xmin, xmax, ymin, ymax = region
    cell_size_x = (xmax - xmin) / grid_size
    cell_size_y = (ymax - ymin) / grid_size

    # Compute grid centers
    x_centers = torch.linspace(xmin + cell_size_x / 2, xmax - cell_size_x / 2, grid_size, device=xyz.device)
    y_centers = torch.linspace(ymin + cell_size_y / 2, ymax - cell_size_y / 2, grid_size, device=xyz.device)
    grid_centers = torch.stack(torch.meshgrid(x_centers, y_centers, indexing='ij'), dim=-1)  # Shape: (grid_size, grid_size, 2)
    grid_centers = grid_centers.view(-1, 2)  # Shape: (M, 2)

    # Compute pairwise distances between points and grid centers
    pairwise_dist = torch.cdist(xyz, grid_centers)  # Shape: (N, M)

    # Compute soft assignment weights
    weights = torch.exp(-pairwise_dist ** 2 / (2 * sigma ** 2))  # Shape: (N, M)

    # Total weight per grid cell
    cell_weights = torch.sum(weights, dim=0)  # Shape: (M,)

    # Compute mean and variance of grid cell weights
    mean_weight = cell_weights.mean()
    loss = torch.mean((cell_weights - mean_weight) ** 2)

    return loss


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./datasets/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
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
        # default=1e-2,
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
