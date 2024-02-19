Run the example script simply by entering 

	python3 example-vision.py

in the terminal/commmand line here. Make sure you have installed PyTorch first to implement the example, since we require Tensor representations of the instances (images).

In this example, we trained Cobweb/4V with sampled 1000 (out of 60,000) training examples in the [MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST) dataset, and then test it with the entire MNIST test set (10,000 test examples in total). You can try different settings by changing the global variables defined in the example script. Lastly, we can derive the visualization of the trained Cobweb/4V tree.

Each MNIST image is represented by a tensor of pixel values like the following:

	tensor([[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000],
         ...
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.4627, 0.9922, 0.9922, 0.9922, 0.1490,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000],
         ...
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 0.0000]]])

If an image is used for training, its ground-truth label (digit) is used along with the image tensor in the training process.

The predictions on all test data may takes a couple of minutes, and with the default global variances defined in the script, you should expect to obtain a test accuracy of 80.86% in the end.

## Vision Example Walkthrough:

	import numpy as np
	import torch
	from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
	from torchvision import datasets, transforms
	from cobweb.visualize import visualize
	import copy
	import random
	from tqdm import tqdm
	from cobweb.cobweb_torch import CobwebTorchTree

 First, we set up the configurations as well as the random seed for fixed shuffling:

	# Configurations:
	size_tr = 1000  # the size of the example training set
	size_te = 10000  # the size of the example test set
	normalize = False  # If true, normalize the MNIST training/test set
	download = True  # Download a copy of MNIST dataset
	seed = 123  # random seed used for shuffling
	cuda = False  # We may just keep cuda=False since the training/predicting process will not be boosted with GPUs
	verbose = True
	random.seed(seed)

 Then, we load MNIST dataset:

	# Load the original MNIST dataset:
	dataset_class = datasets.MNIST
	transform = [transforms.ToTensor()]
	if normalize:
		transform.append(transforms.Normalize((0.1307,), (0.3081,)))
	dataset_transform = transforms.Compose(transform)
	dataset_tr = dataset_class('./datasets/MNIST', train=True, download=download, transform=dataset_transform)
	dataset_te = dataset_class('./datasets/MNIST', train=False, download=download, transform=dataset_transform)
	
	# Return <DataLoader> object for the provided dataset object.
	def get_data_loader(dataset, batch_size, cuda=cuda, drop_last=False, shuffle=False):
		return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
						  **({'num_workers': 0, 'pin_memory': True} if cuda else {}))
	
And generate example training/test DataLoaders:

	# Generate an example training DataLoader.
	dataset_indices_tr = list(range(len(dataset_tr)))
	random.shuffle(dataset_indices_tr)
	dataset_indices_tr = dataset_indices_tr[:size_tr]
	loader_tr = get_data_loader(Subset(dataset_tr, dataset_indices_tr), 
		batch_size=size_tr, cuda=cuda, drop_last=False, shuffle=True)
	
	# Generate an example test DataLoader.
	dataset_indices_te = list(range(len(dataset_te)))
	random.shuffle(dataset_indices_te)
	dataset_indices_te = dataset_indices_te[:size_te]
	loader_te = get_data_loader(Subset(dataset_te, dataset_indices_te), 
		batch_size=size_te, cuda=cuda, drop_last=False, shuffle=True)

### Train the Cobweb Tree

First, initialize the tree:

	imgs_tr, labels_tr = next(iter(loader_tr))
	tree = CobwebTorchTree(imgs_tr.shape[1:])

Then, train the instances:

	if verbose:
		print("Start Training.")
	for i in tqdm(range(imgs_tr.shape[0])):
		tree.ifit(imgs_tr[i], labels_tr[i].item())

You can visualize the concepts generated in the trained tree:

	#Visualize Cobweb:
	visualize(tree)

To see how Cobweb/4V is implemented, please direct to the `README.md` [here](https://github.com/Teachable-AI-Lab/cobweb).
