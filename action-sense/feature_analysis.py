import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

PATH = "saved_features"
NAME = "saved_feat_I3D_RGB"
SPLIT = ["train", "test"]
Model="MULTIMODAL"

def load_features():
    filename = os.path.join(PATH, f"{NAME}_{SPLIT[0]}.pkl")
    with open(filename, "rb") as f:
        ini_dictionary1 = pickle.load(f)
    filename = os.path.join(PATH, f"{NAME}_{SPLIT[1]}.pkl")
    with open(filename, "rb") as f:
        ini_dictionary2 = pickle.load(f)
    final_dictionary = {}
    for key in ini_dictionary1:
        final_dictionary[key] = ini_dictionary1[key] + ini_dictionary2.get(key, 0)
    for key in ini_dictionary2:
        if key not in final_dictionary:
            final_dictionary[key] = ini_dictionary2[key]

    return final_dictionary


def load_labels():
    filename = os.path.join("data/final_10x20", f"{SPLIT[0]}_{Model}.pkl")
    with open(filename, "rb") as f:
        ini_dictionary1 = pickle.load(f)
    filename = os.path.join("data/final_10x20", f"{SPLIT[1]}_{Model}.pkl")
    with open(filename, "rb") as f:
        ini_dictionary2 = pickle.load(f)
    labels1 = ini_dictionary1["label"].values
    labels2 = ini_dictionary2["label"].values
    return np.concatenate((labels1, labels2))



def transform_features():
    features = load_features()
    df = pd.DataFrame(features["features"])
    df = video_level_features(df)
    return df



def video_level_features(df):
    def calculate_mean(array):
        return np.mean(array, axis=0)

    df["mean_feats"] = df["features_RGB"].apply(calculate_mean)
    return df


def kmeans_clustering(samples):
    kmeans = KMeans(n_clusters=19)
    kmeans.fit(samples)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    return cluster_centers, labels


def hierarchical_clustering(samples):
    agg_clustering = AgglomerativeClustering(n_clusters=19, linkage='ward')
    labels = agg_clustering.fit_predict(samples)
    return labels


def accuracy(labels, true_labels):
    ari = adjusted_rand_score(true_labels, labels)
    nmi = normalized_mutual_info_score(true_labels, labels)

    print("Adjusted Rand Index (ARI):", ari)
    print("Normalized Mutual Information (NMI):", nmi)

def visualize_clusters(ax, samples, cluster_centers, labels, title):
    pca = PCA(n_components=2)
    transformed_samples = pca.fit_transform(samples)

    if cluster_centers is not None:
        for i in range(len(cluster_centers)):
            cluster_samples = transformed_samples[labels == i]
            ax.scatter(
                cluster_samples[:, 0], cluster_samples[:, 1], label=f"Cluster {i+1}", s=10
            )

        ax.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],
            marker="x",
            color="black",
            label="Centroids",
        )
    else:
        unique_labels = np.unique(labels)
        for i in unique_labels:
            cluster_samples = transformed_samples[labels == i]
            ax.scatter(
                cluster_samples[:, 0], cluster_samples[:, 1], label=f"Cluster {i+1}", s=10
            )

    ax.set_title(title)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()



def plot_dendrogram(samples):
    Z = linkage(samples, method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()


def plot_cluster_heatmap(samples, labels):
    df = pd.DataFrame(samples)
    df['cluster'] = labels
    df = df.sort_values(by='cluster')
    df = df.drop(columns='cluster')
    sns.clustermap(df, cmap='viridis', figsize=(10, 8))
    plt.title('Cluster Heatmap')
    plt.show()


def plot_tree_diagram(samples, labels):
    Z = linkage(samples, method='ward')
    plt.figure(figsize=(10, 6))
    dendrogram(Z, orientation='right')
    plt.title('Hierarchical Clustering Tree Diagram')
    plt.xlabel('Distance')
    plt.ylabel('Sample Index')
    plt.show()



if __name__ == "__main__":
    features = load_features()
    load_labels()
    df = transform_features()
    samples = np.vstack(df["mean_feats"].to_numpy())

    # K-means clustering
    kmeans_centers, kmeans_predictions = kmeans_clustering(samples)
    visualize_clusters(plt.gca(), samples, kmeans_centers, kmeans_predictions, "K-means Clustering")
    accuracy(kmeans_predictions, load_labels())
    plt.show()

    # Hierarchical clustering
    hierarchical_predictions = hierarchical_clustering(samples)
    visualize_clusters(plt.gca(), samples, None, hierarchical_predictions, "Hierarchical Clustering")
    accuracy(hierarchical_predictions, load_labels())
    plt.show()

    # Dendrogram
    plot_dendrogram(samples)
    plt.show()

    # Cluster Heatmap
    plot_cluster_heatmap(samples, hierarchical_predictions)
    plt.show()

    # Tree Diagram
    plot_tree_diagram(samples, hierarchical_predictions)
    plt.show()

   

    




