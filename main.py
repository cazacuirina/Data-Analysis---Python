import pandas as pd
from hclust import hclust
from functii import *
from sklearn.decomposition import PCA

#Citire date
dataTable = pd.read_csv("WineQualities.csv", index_col=0)
factors = list(dataTable)[1:]
print(factors)

# Partitia optimala
model_hclust = hclust(dataTable, factors)
optimalPart = model_hclust.calcPartition()
print("Optimal partition ", optimalPart)
unique, counts = np.unique(optimalPart, return_counts=True)
print("c1: ", counts[0], "c2: ",counts[1])

#Stabilire ierarhie (Dendograma) - partitia optimala
hierarchyPlot(model_hclust.h, dataTable.index, model_hclust.threshold, "Optimal Partition")

#Axe principale - reducere dimensionalitate
pca = PCA(2)
pca.fit(model_hclust.x)
z = pca.transform(model_hclust.x)

#Plot partitie - partitia optimala
partitionPlot(z, optimalPart, "Optimal", dataTable.index)

#Plot distributii - partitia optimala
distributionPlot(model_hclust.x,optimalPart,factors)

#Histograme partitie optimala
for i in range(len(factors)):
    histograms(model_hclust.x[:, i], optimalPart, factors[i])
show()
# Partitia din 3 clusteri
p3 = model_hclust.calcPartition(3)
print("Partition with 3 clusters ", p3)
unique2, counts2 = np.unique(p3, return_counts=True)
print("c1: ", counts2[0], "c2: ",counts2[1], "c3: ", counts2[2])

#Stabilire ierarhie (Dendograma) - partitia cu 3 clusteri
hierarchyPlot(model_hclust.h, dataTable.index, model_hclust.threshold, "Partition with 3 clusters")

#Plot partitie - partitia cu 3 clusteri
partitionPlot(z, p3,  "3 clusters", dataTable.index)

#Plot distributii - partitia cu 3 clusteri
distributionPlot(model_hclust.x,p3,factors)

#Histograme partitie in 3 clusteri
for i in range(len(factors)):
    histograms(model_hclust.x[:, i], p3, factors[i])

#Afisarea tuturor graficelor
# show()

#Salvarea componenetelor partitiilor intr-un fisier csv
partitionTable = pd.DataFrame(data={
    "OptimalPartition": optimalPart,
    "Partition3Clusters": p3
}, index=dataTable.index)
partitionTable.to_csv("Partitions.csv")
