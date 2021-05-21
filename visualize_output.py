import os
import glob
import numpy as np
import cv2
import av
import OpenEXR
import Imath
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def main():
    FPS = 24
    MIN_DEPTH = 0
    MAX_DEPTH = 50
    SHOW_PREVIEW = True

    container = av.open("output.mp4", mode='w')
    stream = None

    for motion_image_path in tqdm(
        sorted(glob.glob(os.path.join("output", "dweebs_cycles_480_270_*.exr"))),
    ):
        color_image = OpenEXR.InputFile(motion_image_path)
        color_dw = color_image.header()["dataWindow"]
        color_size = (color_dw.max.x - color_dw.min.x + 1, color_dw.max.y - color_dw.min.y + 1)
        color = np.dstack([np.frombuffer(color_image.channel(f"RenderLayer.Combined.{c}", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(color_size[1], color_size[0]) for c in ["R", "G", "B"]])
        depth = np.frombuffer(color_image.channel(f"RenderLayer.Depth.Z", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(color_size[1], color_size[0])
        depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)

        motion_image = OpenEXR.InputFile(motion_image_path)
        motion_dw = motion_image.header()["dataWindow"]
        motion_size = (motion_dw.max.x - motion_dw.min.x + 1, motion_dw.max.y - motion_dw.min.y + 1)
        motion = np.dstack([np.frombuffer(motion_image.channel(f"RenderLayer.Vector.{c}", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(motion_size[1], motion_size[0]) for c in ["X", "Y"]])

        color = np.clip(color * 255, 0, 255).astype(np.uint8)
        depth = (plt.cm.jet((depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH))[..., 0:3] * 255).astype(np.uint8)
        flow = np.zeros_like(color)
        flow[..., 1] = 255
        mag, ang = cv2.cartToPolar(motion[..., 0], motion[..., 1])
        flow[..., 0] = ang * 180 / np.pi / 2
        flow[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow = cv2.cvtColor(flow, cv2.COLOR_HSV2RGB)
        frame = np.hstack([color, depth, flow])
        if frame.shape[0] % 2 != 0:
            frame = frame[:-1, :, :]
        if frame.shape[1] % 2 != 0:
            frame = frame[:, :-1, :]

        if SHOW_PREVIEW:
            cv2.imshow("Preview", frame[..., ::-1])
            cv2.waitKey(1000 // FPS)

        if stream is None:
            stream = container.add_stream("libx264", rate=FPS)
            stream.width = frame.shape[1]
            stream.height = frame.shape[0]
            stream.pix_fmt = "yuv420p"

        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()

if __name__ == "__main__":
    main()