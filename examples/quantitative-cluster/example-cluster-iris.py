import numpy as np
import torch
import copy
import random
import json

from cobweb.cobweb_torch import CobwebTorchTree
from cobweb.visualize import visualize
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cluster import cluster

seed = 100
random.seed(seed)

""" Data loading and preprocessing """
with open('iris.json', 'r') as file:
	instances = json.load(file)
random.shuffle(instances)
# Remove the classifications they have
instance_no_class = [{k: instance[k] for k in instance if k != 'class'} for instance in instances]
classes = [instance[c] for instance in instances for c in instance if c == 'class']
# Transform the dictionaries into tensors
instance_tensors = [torch.tensor([instance[k] for k in instance]) for instance in instance_no_class]
# print(instance_tensors[0].shape)

""" Train the Cobweb tree """
tree = CobwebTorchTree(instance_tensors[0].shape)
for instance in instance_tensors:
	tree.ifit(instance)
visualize(tree)

clusters = next(cluster(tree, instance_tensors, instance_tensors[0].shape))

ari = adjusted_rand_score(clusters, classes)

dv = DictVectorizer(sparse=False)
iris_X = dv.fit_transform(instance_no_class)
pca = PCA(n_components=2)
iris_2d_x = pca.fit_transform(iris_X)

# print(clusters)

colors = ['b', 'g', 'r', 'y', 'k', 'c', 'm']
shapes = ['o', '^', '+']
clust_set = {v: i for i, v in enumerate(list(set(clusters)))}
class_set = {v: i for i, v in enumerate(list(set(classes)))}

for class_idx, class_label in enumerate(class_set):
    x = [v[0] for i, v in enumerate(iris_2d_x) if classes[i] == class_label]
    y = [v[1] for i, v in enumerate(iris_2d_x) if classes[i] == class_label]
    c = [colors[clust_set[clusters[i]]] for i, v in enumerate(iris_2d_x) if
         classes[i] == class_label]
    plt.scatter(x, y, color=c, marker=shapes[class_idx], label=class_label)

plt.title("COBWEB/4V Iris Clustering (ARI = %0.2f)" % (ari))
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend(loc=4)
plt.show()



