import os

blender_executable = "/tmp/haucke-blender/blender-2.92.0-linux64/blender"

def main():
    for blender_file, file_prefix, output_directory_path in [
        # ("scenes/TheDailyDweebsProductionFile/01.lighting/01.lighting.blend", "dweebs", "output"),
        # ("scenes/CoffeeRunProductionRepository/scenes/assembly/coffeerun.assembly.blend", "coffeerun", "output"),
        ("scenes/agent327_lighting/11_01_A.lighting.flamenco.blend", "agent327", "output"),
    ]:
        if os.system(f'FILE_PREFIX="{file_prefix}" OUTPUT_DIRECTORY_PATH="{output_directory_path}" "{blender_executable}" "{blender_file}" --background --python render.py') != 0:
            raise RuntimeError("failed running blender")

if __name__ == "__main__":
    main()