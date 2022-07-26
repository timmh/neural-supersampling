import argparse
import os
import glob
import multiprocessing
from tqdm.auto import tqdm
import numpy as np
import OpenEXR
import Imath
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from config import source_resolution, target_resolution


def process_image(args):
    args, f = args
    try:
        scene, engine, w, h, t = os.path.splitext(os.path.basename(f))[0].split("_")
        w, h, t = int(w), int(h), int(t)
        source = OpenEXR.InputFile(f)

        main_layer = "RenderLayer"
        if os.path.basename(f).startswith("coffeerun"):
            main_layer = "View Layer"
        try:
            depth_channel = source.channel(f"{main_layer}.Depth.Z", Imath.PixelType(Imath.PixelType.FLOAT))
        except TypeError:
            depth_channel = np.full((h, w), np.nan, dtype=np.float32)
        depth = np.dstack([np.frombuffer(depth_channel, dtype=np.float32).reshape(h, w)])
        motion = np.dstack([
            np.frombuffer(source.channel(f"{main_layer}.Vector.{c}", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(h, w) for c in ["Y", "X"]
        ] + [
            np.zeros((h, w), dtype=np.float32)  # OpenCV will only save 1, 3 or 4 channel OpenEXR images
        ])
        cv2.imwrite(os.path.join(args.out_dir_depth, os.path.basename(f)), depth)
        cv2.imwrite(os.path.join(args.out_dir_motion, os.path.basename(f)), motion)
    except Exception as e:
        tqdm.write(f"Error occured while processing file {f}: {str(e)}")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("in_dir", type=str, help="the directory containing raw multilayer files produced by blender")
    argparser.add_argument("out_dir_depth", type=str, help="the directory to output compressed depth files to")
    argparser.add_argument("out_dir_motion", type=str, help="the directory to output compressed motion files to")
    args = argparser.parse_args()

    os.makedirs(args.out_dir_depth, exist_ok=True)
    os.makedirs(args.out_dir_motion, exist_ok=True)

    # find all rendered image files
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.exr")))

    # filter for low-resolution files which have a corresponding high-resolution file
    # TODO: fix caminandesllamigos-01 depth and motion rendering and re-enable here
    filtered_files = []
    for f in files:
        scene, engine, w, h, t = os.path.splitext(os.path.basename(f))[0].split("_")
        if (int(w), int(h)) == source_resolution and os.path.exists(os.path.join(args.in_dir, f"{scene}_{engine}_{target_resolution[0]}_{target_resolution[1]}_{t}.exr")) and not os.path.basename(f).startswith("caminandesllamigos-01"):
            filtered_files.append(f)

    with multiprocessing.Pool(os.cpu_count() // 2) as p:
        with tqdm(total=len(filtered_files)) as pbar:
            for _ in p.imap_unordered(process_image, [(args, f) for f in filtered_files]):
                pbar.update()
        

if __name__ == "__main__":
    main()