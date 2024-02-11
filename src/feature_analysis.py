import pickle
from utils import args
import os

PATH = "saved_features"
FRAMES = 5
NAME = "feat"
SPLIT = "test"
DENSE = True
SHIFT = "D1"


# Load features (pkl)
filename = os.path.join(
    PATH,
    NAME + "_" + str(FRAMES) + "_" + str(DENSE) + "_" + SHIFT + "_" + SPLIT + ".pkl",
)

with open(filename, "rb") as f:
    features = pickle.load(f)
print(features["features"][0]["features_RGB"].shape)
