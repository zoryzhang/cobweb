Here we present an implementation example based on the [Disease Symptom Prediction dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset?select=dataset.csv) on Kaggle to make diagnosis based on the symptoms people might have.

	from random import shuffle, seed, sample
	import time
	import csv
	import numpy as np
	import pandas as pd
	from copy import copy, deepcopy
	from tqdm import tqdm
	from cobweb.cobweb import CobwebTree
	from cobweb.visualize import visualize

Suppose we have the global variable values:

	size_tr = 100
	random_seed = 32
	verbose = True

`size_tr`: the size of the training set (the remaining data is then used for prediction)
`random_seed`: the random seed

### Data Overview and Preprocessing

A snippet of the dataset is as the following:

| Disease | Symptom_1 | Symptom_2 | Symptom_3 | Symptom_4 | Symptom_5 | Symptom_6 | Symptom_7 | Symptom_8 | Symptom_9 | Symptom_10 | Symptom_11 | Symptom_12 | Symptom_13 | Symptom_14 | Symptom_15 | Symptom_16 | Symptom_17 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fungal infection | itching | skin_rash | nodal_skin_eruptions | dischromic _patches | | | | | | | | | | | | |
| Fungal infection | skin_rash | nodal_skin_eruptions | dischromic _patches | | | | | | | | | | | | | |
| Fungal infection | itching | nodal_skin_eruptions | dischromic _patches | | | | | | | | | | | | | |
| Fungal infection | itching | skin_rash | dischromic _patche | | | | | | | | | | | | | |
| Fungal infection | itching | skin_rash | nodal_skin_eruptions | | | | | | | | | | | | | |




