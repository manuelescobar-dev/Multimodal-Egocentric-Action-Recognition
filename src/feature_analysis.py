import pickle
from utils import args
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


PATH = "saved_features"
FRAMES = 10
NAME = "feat"
SPLIT = "train"
DENSE = True
SHIFT = "D1"


def load_features():
    # Load features (pkl)
    filename = os.path.join(
        PATH,
        NAME
        + "_"
        + str(FRAMES)
        + "_"
        + SPLIT
        + "_"
        + str(DENSE)
        + "_"
        + SHIFT
        + "_"
        + SPLIT
        + ".pkl",
    )

    with open(filename, "rb") as f:
        features = pickle.load(f)
    return features


def load_labels():
    # Load labels
    filename = os.path.join(
        "train_val",
        SHIFT + "_" + SPLIT + ".pkl",
    )

    with open(filename, "rb") as f:
        labels = pickle.load(f)
    return labels["verb_class"].values


def transform_features():
    features = load_features()
    df = pd.DataFrame(features["features"])
    df = video_level_features(df)
    return df


def plot_features(features):
    # Plot features
    plt.figure()
    for i in range(5):
        plt.scatter(
            np.arange(len(features["features"][0]["features_RGB"][i])),
            features["features"][0]["features_RGB"][i],
            label=f"Clip {i}",
            s=0.2,
        )
    plt.legend()
    plt.show()


def video_level_features(df):
    # Define a function to calculate mean along axis 0
    def calculate_mean(array):
        return np.mean(array, axis=0)

    # Apply the function to each element of the array_column and create a new column with means
    df["mean_feats"] = df["features_RGB"].apply(calculate_mean)
    return df


def clustering(samples):
    kmeans = KMeans(n_clusters=8)  # Change the number of clusters as needed
    kmeans.fit(samples)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    return cluster_centers, labels


def accuracy(labels, true_labels):
    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)

    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)


def visualize_clusters(samples, cluster_centers, labels):
    pca = PCA(n_components=2)
    transformed_samples = pca.fit_transform(samples)

    plt.figure(figsize=(10, 6))
    for i in range(len(cluster_centers)):
        cluster_samples = transformed_samples[labels == i]
        plt.scatter(
            cluster_samples[:, 0], cluster_samples[:, 1], label=f"Cluster {i+1}", s=10
        )

    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        marker="x",
        color="black",
        label="Centroids",
    )
    plt.title("K-means Clustering")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    load_labels()
    df = transform_features()
    # Convert the 'mean_column' to a numpy array
    samples = df["mean_feats"].to_numpy()

    # Stack the arrays along axis 0 to create a single numpy array
    samples = np.vstack(samples)
    centers, predictions = clustering(samples)
    visualize_clusters(samples, centers, predictions)
    labels = load_labels()
    accuracy(predictions, labels)
