import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from seaborn import kdeplot, scatterplot

def histograms(z, p, factor):
    fig = plt.figure(figsize=(9, 6))
    clase = np.unique(p)
    q = len(clase)
    fig.suptitle("Histograms for factor: " + factor)
    axe = fig.subplots(1, q, sharey=True)
    for i in range(q):
        print(z[p == clase[i]])
        axe[i].set_xlabel(clase[i])
        axe[i].hist(x=z[p == clase[i]], range=(min(z), max(z)), rwidth=0.9)

def distributionPlot(z,p,factors):
    fig = plt.figure(figsize=(9,6))
    for i in range(len(factors)):
        ax = fig.add_subplot(len(factors), 1, i + 1)
        ax.set_title("Distributions for factor: "+factors[i])
        kdeplot(x=z[:,i],hue=p,ax=ax)

def hierarchyPlot(h, labels, threshold, title):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Dendogram - Hierarchy Plot for "+title)
    dendrogram(h, labels=labels, color_threshold=threshold, ax=ax)

def partitionPlot(z, p, title, labels=None):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Partition plot - "+title)
    scatterplot(x=z[:, 0], y=z[:, 1], hue=p, hue_order=np.unique(p), ax=ax)
    if labels is not None:
        for i in range(len(labels)):
            ax.text(z[i, 0], z[i, 1], labels[i])

def show():
    plt.show()
