from sklearn import datasets
import numpy as np

# Import helper functions
from ml_fs.common_utils import Plot
from ml_fs.unsupervised_learning_models import PAM

def main():
    # Load the dataset
    X, y = datasets.make_blobs()

    # Cluster the data using K-Medoids
    clf = PAM(k=3)
    y_pred = clf.predict(X)

    # Project the data onto the 2 primary principal components
    p = Plot()
    p.plot_in_2d(X, y_pred, title="PAM Clustering")
    p.plot_in_2d(X, y, title="Actual Clustering")


if __name__ == "__main__":
    main()
