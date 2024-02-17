import subprocess
from multiprocessing import Pool, Process
from utils.logger import logger
import os
import torch

SAMPLING = [True, False]
COMPLETED = {""}
MODALITY = "EMG"
CONFIG = f"configs/{MODALITY}_train.yaml"
NAME = "emg_train"


def run_command(args):
    command = [
        "python3",
        "train_classifier.py",
        f"name={args['name']}",
        f"config={CONFIG}",
        f"train.dense_sampling.{MODALITY}={args['sampling']}",
        f"test.dense_sampling.{MODALITY}={args['sampling']}",
        f"train.stride.{MODALITY}={args['stride']}",
        f"test.stride.{MODALITY}={args['stride']}",
    ]
    if args["resume_from"]:
        r = {"EMG": os.path.join("saved_models", args["name"])}
        command.append(f"resume_from={r}")
    subprocess.run(command)


# Maximum number of parallel executions
max_processes = 2  # Adjust as needed


if __name__ == "__main__":
    logger.info("Starting training")
    arguments_list = []
    for s in SAMPLING:
        if s == True:
            max_stride = 2
        for i in range(1, max_stride + 1):
            name = NAME + f"_{s}_{i}"
            if name not in COMPLETED:
                a = {
                    "name": name,
                    "sampling": s,
                    "stride": i,
                }
                if os.path.exists(os.path.join("saved_models", name)):
                    a["resume_from"] = True
                    arguments_list.append(a)
                else:
                    a["resume_from"] = False
                    arguments_list.append(a)

    # Execute commands sequentially
    for args in arguments_list:
        run_command(args)
