"""
Module for plotting graphs. Plots from csv file
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_epochs(path):
    df = pd.read_csv(path)
    data_dict = df.to_dict()
    epochs = len(data_dict["Train Epoch Loss"])
    X = [i for i in range(0, epochs)]
    plt.scatter(X, data_dict["Train Epoch Loss"], label="Train")
    plt.scatter(X, data_dict["Val Epoch Loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.scatter(X, data_dict["Train Epoch Acc"], label="Train")
    plt.scatter(X, data_dict["Val Epoch Acc"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()







