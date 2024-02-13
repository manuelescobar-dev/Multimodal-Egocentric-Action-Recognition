import subprocess
from multiprocessing import Pool, Process
from utils.logger import logger

FRAMES = [5, 10, 25]
ACTIONS = ["train", "test"]
SAMPLING = [True, False]
CONFIG = "configs/training.yaml"
MODELS = ["OriginalClassifier"]


def run_command(args):
    command = [
        "python3",
        "train_classifier.py",
        f"action={args['action']}",
        f"name={args['name']}",
        f"config={CONFIG}",
        f"train.num_frames_per_clip.RGB={args['num_frames_per_clip']}",
        f"train.dense_sampling.RGB={args['sampling']}",
        f"test.num_frames_per_clip.RGB={args['num_frames_per_clip']}",
        f"test.dense_sampling.RGB={args['sampling']}",
        f"dataset.RGB.features_name={args['name']}",
        f"models.RGB.model={args['model']}",
    ]
    subprocess.run(command)


# Maximum number of parallel executions
max_processes = 2  # Adjust as needed


if __name__ == "__main__":
    logger.info("Starting training")
    arguments_list = []
    for i in MODELS:
        for j in FRAMES:
            for k in SAMPLING:
                for l in ACTIONS:
                    arguments_list.append(
                        {
                            "action": l,
                            "name": f"EAR_{i}_{j}_{k}_{l}",
                            "num_frames_per_clip": j,
                            "sampling": k,
                            "model": i,
                            "name": f"feat_{j}_{k}",
                        }
                    )

    # Execute commands sequentially
    for args in arguments_list:
        run_command(args)
