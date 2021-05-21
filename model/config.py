import os
import torch

history_length = 1
upsampling_factor = 4
batch_size = 1
ssim_window_size = 9
n_epochs = 10
log_interval = 1
save_interval = 10
enable_amp = False
enable_anomaly_detection = False
# perceptual_loss_weight = 0.1
perceptual_loss_weight = 0
# num_workers = os.cpu_count() // 2
num_workers = 1
weight_decay = 1e-5
max_depth = 25
# data_root = os.path.join(os.path.expanduser("~"), "gdrive", "neural-supersampling", "output")
data_root = os.path.join(os.path.expanduser("~"), "Downloads", "neural-supersampling-output-test")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
inference_dtype = torch.float32
source_resolution = (480, 270)
target_resolution = (1920, 1080)