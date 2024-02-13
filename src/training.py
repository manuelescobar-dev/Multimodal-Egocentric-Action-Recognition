import subprocess
from multiprocessing import Pool, Process
from utils.logger import logger
import os
import torch

FRAMES = [5, 10, 25]
ACTIONS = ["train"]
SAMPLING = [True, False]
CONFIG = "configs/training.yaml"
MODELS = ["LSTM_Classifier"]


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
        f"dataset.RGB.features_name={args['feats_name']}",
        f"models.RGB.model={args['model']}",
    ]
    if args["resume_from"]:
        command.append(f"resume_from={os.path.join('saved_models', args['name'])}")
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
                    if os.path.exists(os.path.join("saved_models", f"{i}_{j}_{k}")):
                        arguments_list.append(
                            {
                                "action": l,
                                "name": f"{i}_{j}_{k}",
                                "num_frames_per_clip": j,
                                "sampling": k,
                                "model": i,
                                "feats_name": f"feat_{j}_{k}",
                                "resume_from": True,
                            }
                        )
                    else:
                        arguments_list.append(
                            {
                                "action": l,
                                "name": f"{i}_{j}_{k}",
                                "num_frames_per_clip": j,
                                "sampling": k,
                                "model": i,
                                "feats_name": f"feat_{j}_{k}",
                                "resume_from": False,
                            }
                        )

    # Execute commands sequentially
    for args in arguments_list:
        run_command(args)
