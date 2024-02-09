Run the example script simply by entering

	python3 example-tabular.py

in the terminal/command line here. Here we employ the example [Disease Symptom Prediction dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset?select=dataset.csv) on Kaggle to make diagnosis based on the symptoms people might have. One example of the data point representation after preprocessing:

	{'disease': {'hepatitis-a': 1}, 
	 'symptom': {'joint_pain': 1, 'vomiting': 1, 'yellowish_skin': 1, 'dark_urine': 1, 'nausea': 1, 'loss_of_appetite': 1, 'abdominal_pain': 1, 'diarrhoea': 1, 'mild_fever': 1, 'yellowing_of_eyes': 1, 'muscle_pain': 1}}

In the example script, we first transform the tabular data into corresponding dictionary representations, then train Cobweb with a small portion of data, and test it with the remaining data available. With the default global variables defined, you are expected to obtain a test accuracy of 97.51%.

To see how Cobweb is implemented, please direct to the `README.md` [here](https://github.com/Teachable-AI-Lab/cobweb).

