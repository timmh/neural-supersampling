import os

from config import blender_executable


def main():
    for blender_file, file_prefix, output_directory_path in [

        # the daily dweebs: https://cloud.blender.org/films/dailydweebs
        ("scenes/TheDailyDweebsProductionFile/01.lighting/01.lighting.blend", "dweebs", "output", -1, -1),

        # coffee run: https://cloud.blender.org/films/coffee-run
        ("scenes/CoffeeRunProductionRepository/scenes/assembly/coffeerun.assembly.blend", "coffeerun", "output", -1, -1),

        # agent 327: https://cloud.blender.org/films/agent-327
        ("scenes/agent327/01_01_B-CityTilt/01_01_B-city_tilt/01_01_B.lighting.blend", "agent327-citytilt", "output", -1, -1),
        ("scenes/agent327/02_01_A-CarEnter/02_01_A-car_enter/02_01_A.lighting.blend", "agent327-carenter", "output", -1, -1),
        ("scenes/agent327/03_01_B-Enter/03_01_B-enter/03_01_B.lighting.blend", "agent327-enter", "output", -1, -1),
        ("scenes/agent327/04_01_E-Sitting/04_01_H-sitting/04_01_H.lighting.blend", "agent327-sitting", "output", -1, -1),
        ("scenes/agent327/07_04_F-WallSlam/07_04_F-wall_slam/07_04_F.lighting.blend", "agent327-wallslam", "output", -1, -1),
        ("scenes/agent327/08_05_A-Headbutt/08_05_A-headbutt/08_05_A.lighting.blend", "agent327-headbutt", "output", -1, -1),
        ("scenes/agent327/10_03_B-Agentdodges/10_03_B-agent_dodges/10_03_B.lighting.blend", "agent327-agentdodges", "output", -1, -1),
        ("scenes/agent327/11_02_A-Pinned/11_02_A-pinned/11_02_A.lighting.blend", "agent327-pinned", "output", -1, -1),
        ("scenes/agent327/13_04_C-ReturnoftheBarber/13_04_C-return_of_the_barber/13_04_C.lighting.blend", "agent327-returnofthebarber", "output", -1, -1),
        ("scenes/agent327/14_01_C-Descent/14_01_C-descent/14_01_C.lighting.blend", "agent327-descent", "output", -1, -1),
        ("scenes/agent327/14_06_G-WuManchu/14_06_G-wu_manchu/14_06_G.lighting.blend", "agent327-wumanchu", "output", -1, -1),
        ("scenes/agent327/14_09_G-Goonsapproach/14_09_G-goons_approach/14_09_G.lighting.blend", "agent327-goonsapproach", "output", -1, -1),

        # caminandes llamigos: https://cloud.blender.org/films/caminandes-3
        ("scenes/CaminandesLlamigos/01_01_A/01_01_A.lighting.blend", "caminandesllamigos-01", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_02_A/01_02_A.lighting.blend", "caminandesllamigos-02", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_02_B/01_02_B.lighting.blend", "caminandesllamigos-03", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_02_C/01_02_C.lighting.blend", "caminandesllamigos-04", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_02_D/01_02_D.lighting.blend", "caminandesllamigos-05", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_03_A/01_03_A.lighting.blend", "caminandesllamigos-06", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_04_A/01_04_A.lighting.blend", "caminandesllamigos-07", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_04_B/01_04_B.lighting.blend", "caminandesllamigos-08", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_05_A/01_05_A.lighting.blend", "caminandesllamigos-09", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_05_B/01_05_B.lighting.blend", "caminandesllamigos-10", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_06_A/01_06_A.lighting.blend", "caminandesllamigos-11", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_08_A/01_08_A.lighting.blend", "caminandesllamigos-12", "output", -1, -1),
        ("scenes/CaminandesLlamigos/01_09_A/01_09_A.lighting.blend", "caminandesllamigos-13", "output", -1, -1),
        ("scenes/CaminandesLlamigos/02_01_A/02_01_A.lighting.blend", "caminandesllamigos-14", "output", -1, -1),
        ("scenes/CaminandesLlamigos/02_02_A/02_02_A.lighting.blend", "caminandesllamigos-15", "output", -1, -1),
        ("scenes/CaminandesLlamigos/02_03_A/02_03_A.lighting.blend", "caminandesllamigos-16", "output", -1, -1),
        ("scenes/CaminandesLlamigos/02_04_A/02_04_A.lighting.blend", "caminandesllamigos-17", "output", -1, -1),
        ("scenes/CaminandesLlamigos/02_05_A/02_05_A.lighting.blend", "caminandesllamigos-18", "output", -1, -1),
        ("scenes/CaminandesLlamigos/03_01_A/03_01_A.lighting.blend", "caminandesllamigos-19", "output", -1, -1),
        ("scenes/CaminandesLlamigos/03_01_B/03_01_B.lighting.blend", "caminandesllamigos-20", "output", -1, -1),
        ("scenes/CaminandesLlamigos/03_02_A/03_02_A.lighting.blend", "caminandesllamigos-21", "output", -1, -1),
        ("scenes/CaminandesLlamigos/06_01_A/06_01_A.lighting.blend", "caminandesllamigos-22", "output", -1, -1),
        ("scenes/CaminandesLlamigos/06_02_A/06_02_A.lighting.blend", "caminandesllamigos-23", "output", -1, -1),
        ("scenes/CaminandesLlamigos/06_03_A/06_03_A.lighting.blend", "caminandesllamigos-24", "output", -1, -1),
        ("scenes/CaminandesLlamigos/06_04_A/06_04_A.lighting.blend", "caminandesllamigos-25", "output", -1, -1),
        ("scenes/CaminandesLlamigos/06_05_A/06_05_A.lighting.blend", "caminandesllamigos-26", "output", -1, -1),
        ("scenes/CaminandesLlamigos/07_01_A/07_01_A.lighting.blend", "caminandesllamigos-27", "output", -1, -1),
        ("scenes/CaminandesLlamigos/08_01_A/08_01_A.lighting.blend", "caminandesllamigos-28", "output", -1, -1),
        ("scenes/CaminandesLlamigos/08_02_A/08_02_A.lighting.blend", "caminandesllamigos-29", "output", -1, -1),
        ("scenes/CaminandesLlamigos/09_01_A.lighting/09_01_A.lighting.blend", "caminandesllamigos-30", "output", -1, -1),
        ("scenes/CaminandesLlamigos/09_02_A.lighting/09_02_A.lighting.blend", "caminandesllamigos-31", "output", -1, -1),
        ("scenes/CaminandesLlamigos/09_03_A.lighting/09_03_A.lighting.blend", "caminandesllamigos-32", "output", -1, -1),
        ("scenes/CaminandesLlamigos/09_04_A.lighting/09_04_A.lighting.blend", "caminandesllamigos-33", "output", -1, -1),
        ("scenes/CaminandesLlamigos/10_01_A.lighting/10_01_A.lighting.blend", "caminandesllamigos-34", "output", -1, -1),
    ]:
        if os.system(f'FILE_PREFIX="{file_prefix}" OUTPUT_DIRECTORY_PATH="{output_directory_path}" "{blender_executable}" "{blender_file}" --background --python render.py') != 0:
            raise RuntimeError("failed running blender")

if __name__ == "__main__":
    main()