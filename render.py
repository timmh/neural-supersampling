import sys
import os
from enum import Enum, auto
import bpy


# print to stderr by default
sys.stdout = sys.stderr


class RenderingEngine(Enum):
    Eevee = auto()
    Cycles = auto()


def render(output_directory_path: str, file_prefix: str):

    assert os.path.isdir(output_directory_path), "output path must be a directory"
    assert len(file_prefix) > 0, "file prefix must not be empty"

    for rendering_engine in [
        RenderingEngine.Cycles,
    ]:
        for render_resolution in [
            (480, 270),
            (1920, 1080),
        ]:

            context = bpy.context
            scene = context.scene
            preferences = context.preferences
            assert len(scene.view_layers) == 1, "number of view layers in scene is ambiguous"
            view_layer = scene.view_layers[0]
            
            if rendering_engine == RenderingEngine.Eevee:
                scene.render.engine = "BLENDER_EEVEE"
            elif rendering_engine == RenderingEngine.Cycles:
                scene.render.engine = "CYCLES"
                scene.cycles.samples = 256
                preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
                scene.cycles.device = "GPU"
                for devices in bpy.context.preferences.addons["cycles"].preferences.get_devices():
                    for device in devices:
                        device.use = device.type != "CPU"
            else:
                raise RuntimeError("unknown render pass")

            scene.render.resolution_x = render_resolution[0]
            scene.render.resolution_y = render_resolution[1]
            scene.render.resolution_percentage = 100
            scene.render.tile_x = 480
            scene.render.tile_y = 270

            scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
            scene.render.image_settings.use_preview = False
            scene.render.use_placeholder = False
            scene.render.use_overwrite = True
            scene.render.use_motion_blur = False

            if rendering_engine == RenderingEngine.Eevee:
                scene.render.image_settings.color_mode = "RGB"
            elif rendering_engine == RenderingEngine.Cycles:
                view_layer.use_pass_combined = True
                view_layer.use_pass_vector = True
                view_layer.use_pass_z = True
            else:
                raise RuntimeError("unknown render pass")

            scene.render.filepath = os.path.join(output_directory_path, f"{file_prefix}_{rendering_engine.name.lower()}_{render_resolution[0]}_{render_resolution[1]}_")
            bpy.ops.render.render(animation=True)


render(
    file_prefix=os.environ["FILE_PREFIX"],
    output_directory_path=os.environ["OUTPUT_DIRECTORY_PATH"],
)