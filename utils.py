import numpy as np
import torch
import torch.nn.functional as F


def noop(arg):
    return arg


class noop_context():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False


def warp(image: torch.Tensor, flow: torch.Tensor):
    B, _, H, W = image.size()
    xx = torch.arange(0, W, dtype=image.dtype, device=image.device).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H, dtype=image.dtype, device=image.device).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx, yy),1)
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1
    vgrid[:, 1, :, :] = 2 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1
    return F.grid_sample(image, vgrid.permute(0, 2, 3, 1).to(image.dtype), mode="bilinear", align_corners=True)


# TODO: fix
# adapted from: https://zenn.dev/pinto0309/scraps/7d4032067d0160
def bilinear_sample_noloop(image, grid):
    """
    :param image: sampling source of shape [N, C, H, W]
    :param grid: integer sampling pixel coordinates of shape [N, grid_H, grid_W, 2]
    :return: sampling result of shape [N, C, grid_H, grid_W]
    """
    Nt, C, H, W = image.shape
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]
    xgrid, ygrid = grid.split([1, 1], dim=-1)
    mask = ((xgrid >= 0) & (ygrid >= 0) & (xgrid < W - 1) & (ygrid < H - 1)).float()
    x0 = torch.floor(xgrid)
    x1 = x0 + 1
    y0 = torch.floor(ygrid)
    y1 = y0 + 1
    wa = ((x1 - xgrid) * (y1 - ygrid)).permute(3, 0, 1, 2)
    wb = ((x1 - xgrid) * (ygrid - y0)).permute(3, 0, 1, 2)
    wc = ((xgrid - x0) * (y1 - ygrid)).permute(3, 0, 1, 2)
    wd = ((xgrid - x0) * (ygrid - y0)).permute(3, 0, 1, 2)
    x0 = (x0 * mask).view(Nt, grid_H, grid_W).long()
    y0 = (y0 * mask).view(Nt, grid_H, grid_W).long()
    x1 = (x1 * mask).view(Nt, grid_H, grid_W).long()
    y1 = (y1 * mask).view(Nt, grid_H, grid_W).long()
    ind = torch.arange(Nt, device=image.device) #torch.linspace(0, Nt - 1, Nt, device=image.device)
    ind = ind.view(Nt, 1).expand(-1, grid_H).view(Nt, grid_H, 1).expand(-1, -1, grid_W).long()
    image = image.permute(1, 0, 2, 3)
    output_tensor = (image[:, ind, y0, x0] * wa + image[:, ind, y1, x0] * wb + image[:, ind, y0, x1] * wc + \
                 image[:, ind, y1, x1] * wd).permute(1, 0, 2, 3)
    output_tensor *= mask.permute(0, 3, 1, 2).expand(-1, C, -1, -1)
    image = image.permute(1, 0, 2, 3)
    return output_tensor, mask


# TODO: fix
# adapted from: https://zenn.dev/pinto0309/scraps/7d4032067d0160
def warp_tensorrt(image: torch.Tensor, flow: torch.Tensor):
    B, _, H, W = image.size()
    yy = torch.arange(0, H, dtype=image.dtype, device=image.device).view(-1,1).repeat(1,W)
    xx = torch.arange(0, W, dtype=image.dtype, device=image.device).view(1,-1).repeat(H,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((yy, xx), 1)
    vgrid = grid + flow
    output, _ = bilinear_sample_noloop(image, vgrid.permute(0, 2, 3, 1).to(image.dtype))
    return output


def transform_rgb(rgb):
    return rgb / 255.


def transform_depth(depth):
    max_disp = max(depth.shape[0], depth.shape[1])
    return 1 / depth.clip(1 / max_disp, np.inf)


def torch_to_numpy(tensor):
    return (tensor * 255).clamp(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().detach().numpy()