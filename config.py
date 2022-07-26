import os
import torch

history_length = 5
upsampling_factor = 4
batch_size_train = 8
batch_size_test = 1
learning_rate = 1e-4  # Xiao et al. 2020
ssim_window_size = 9
n_epochs = 100
log_interval = 1
save_interval = 10
enable_amp = False
enable_anomaly_detection = False
perceptual_loss_weight = 0.1
num_workers = os.cpu_count() // 2
weight_decay = 1e-5
fps = 30
data_root = "/content/renderings"
tensorboard_root = os.environ["TENSORBOARD_PATH"]
color_file_extension = "png"
depth_file_extension = "exr"
motion_file_extension = "exr"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inference_dtype = torch.float32
source_resolution = (480, 270)
target_resolution = (1920, 1080)
train_random_crop_size = (64, 64)
train_scenes = [
    "agent327",
    "coffeerun",
    "caminandesllamigos",
]
test_scenes = [
    "dweebs",
]
blender_executable = os.environ["BLENDER_PATH"]