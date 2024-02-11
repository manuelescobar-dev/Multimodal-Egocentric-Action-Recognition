import subprocess
from multiprocessing import Pool, Process
from utils.logger import logger

FRAMES = [5, 10, 25]
SPLITS = ["test", "train"]
SAMPLING = [True, False]


def run_command(args):
    command = [
        "python3",
        "save_feat.py",
        f"name={args['name']}",
        f"config={args['config']}",
        f"dataset.shift={args['shift']}",
        f"dataset.RGB.data_path={args['data_path']}",
        f"split={args['split']}",
        f"save.num_frames_per_clip.RGB={args['num_frames_per_clip']}",
        f"save.dense_sampling.RGB={args['sampling']}",
    ]
    subprocess.run(command)


# Maximum number of parallel executions
max_processes = 2  # Adjust as needed


if __name__ == "__main__":
    logger.info("Starting feature extraction")
    arguments_list = []
    for i in FRAMES:
        for j in SPLITS:
            for k in SAMPLING:
                arguments_list.append(
                    {
                        "name": f"feat_{i}_{j}_{k}",
                        "config": "configs/I3D_save_feat.yaml",
                        "shift": "D1-D1",
                        "data_path": "data/EK",
                        "split": j,
                        "num_frames_per_clip": i,
                        "sampling": k,
                    }
                )

    # Execute commands sequentially
    for args in arguments_list:
        run_command(args)
