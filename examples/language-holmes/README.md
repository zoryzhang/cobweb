In this example, we implement the [Microsoft Research Sentence Completion Challenge](https://www.microsoft.com/en-us/research/project/msr-sentence-completion-challenge/overview/). In particular, we preprocess and train Cobweb with only 2 Holmes stories (in `./data`. You can incorporate more text files from the raw data, though the corresponding training process will take more time), and the stories will be preprocessed by filtering words with less frequencies and then make them into n-grams (anchor + context words) for training and predictions. All the preprocessed grams (instances) will be integrated in the output `holmes_stories.json` as you run the example script. 

Start by installing the necessary spacy language package:

    python -m spacy download en_core_web_sm

If you don't find the compiled Python library of Cobweb from `cobweb.cpp` with `pybind11`, please refer to the `README.md` [here](https://github.com/Teachable-AI-Lab/cobweb/tree/main). 

    import os
    import json
    import random as rd
    from random import random, shuffle
    from collections import Counter
    from tqdm import tqdm
    from multiprocessing import Pool
    from utility import load_texts, story2instances
    from cobweb.cobweb import CobwebTree
    from cobweb.visualize import visualize

Here we import the `utility` module to preprocess and load the text data.

Configuration:

    json_name = "holmes_stories.json"  # the name of the integrated JSON file
    limit = None
    window = 10  # the size of a "gram"
    least_frequency = 100  # used to filter out the words having frequency less than some specified one.
    seed = 123  # random seed for shuffling instances


## Load and preprocess the text data

We first create the integrated text from the text files (stories) we use (in `./data`. In the example we only use two stories `1ADAM10.txt` and `1ARGN10.txt`. If you are interested in training with more stories, feel free to download the stories directly in the corresponding site).

    print("Start the loading and preprocessing process.")
    # Create the integrated text from several text file (stories):
    if not os.path.isfile(json_name):
        print("\nReading and preprocessing...")
        stories_dir = "./data"
        stories = list(load_texts(stories_dir, limit=limit))
        with open(json_name, "w") as fout:
            json.dump(stories, fout, indent=4)
    else:
        # If the collection of tokens is produced previously, just load it instead of re-generate one
        print("\nLoading the preprocessed stories...")
        with open(json_name, "r") as fin:
            stories = json.load(fin)

You can then see the collection of sparsed tokens of the processed stories in `./holmes_stories.json` - it is  indeed the two collections (lists) of the tokens of the two stories we have.

Next, lets look at the top 100 most frequent word:

    print("\npreview of the 100 most frequent words frequency:")
	overall_freq = Counter([word for story in stories for word in story])
	print(overall_freq.most_common(100))

To generate the instances learned by the Cobweb tree, we may filter out the words (tokens) that are not frequent (not emerge more than `least_frequency=100` times in a story) for each story:

    stories = [[word for word in story if overall_freq[word] >= least_frequency] for story in stories]

Finally, we can generate the instances for Cobweb learning, which are indeed the grams each with an anchor word and its context words:

    instances = []

    # We use Pool to preprocess the data simultaneously among the stories
    with Pool() as pool:
        processed_stories = pool.starmap(story2instances, [(story, window) for story in stories])
        for story_idx, story_instances in enumerate(processed_stories):
            for anchor_idx, instance in story_instances:
                instances.append(instance)
    rd.seed(seed)
    shuffle(instances)

We use the function `story2instances` in `utility` to transfer the tokens of each stories into corresponding instances. One instance example:

    {
        'context': {'hero': 0.1, 'son': 0.5, 'go': 0.1, 'return': 0.14285714285714285, 'near': 0.16666666666666666, 'like': 0.2, 'fall': 0.25, 'time': 0.3333333333333333, 'round': 1.0, 'hand': 1.0, 'know': 0.5, 'ship': 0.3333333333333333, 'tell': 0.25, 'heart': 0.2, 'll': 0.16666666666666666, 'shall': 0.14285714285714285, 'bring': 0.125, 'thee': 0.1111111111111111}, 
        
        'anchor': {'man': 1}
    }

which corresponds to the story snippet below:

    hero son go return near like fall time son round _<man>_ hand know ship tell heart ll shall bring thee go

Instead of storing the counts of each context word in an instance, we weight these words based on their proximity to the anchor word and their frequency in the snippet, so that the instance can store the propositional information and the co-occurance between words.


## Train Cobweb with Sparsed Instances

First intialize the tree object:

    tree = CobwebTree(0.000001, False, 0, True, False)

Then use all the instances incrementally:

    for instance in instances:
        tree.ifit(instance)

You can have a visualization of the trained tree with the following:

    visualize(tree)

<figure>
    <img src="./viz-example.png"
         alt="Visualization of concept formation">
    <figcaption>The visualization interface of the trained Cobweb tree. You can select the attribute you want to focus on with the `Focus Attribute` tab, and select (zoom in/out) the learned concept by directly clicking the concept/cluster circle. The corresponding attribute-value table (i.e. the stored information of a concept node) is shown on the lower right. </figcaption>
</figure>

# Test Cobweb

[TBC]

# Incremental Learning in Language Tasks

Since Cobweb learns instances incrementally instead of in batches, we can see how Cobweb's performance evolves as it learns more instances.

We first generate the collection of instances like what we did previously, but we include the additional story and instance indices:

    instances = []
    with Pool() as pool:
        processed_stories = pool.starmap(story2instances, [(story, window) for story in stories])
        for story_idx, story_instances in enumerate(processed_stories):
            for anchor_idx, instance in story_instances:
                instances.append((instance, story_idx, anchor_idx))
    rd.seed(seed)
    shuffle(instances)

Then do the iterative train-and-test process: After training an instance, test Cobweb with its anchor prediction of the next upcoming instance, and then train the instance that is just made prediction to.

    tree = CobwebTree(0.000001, False, 0, True, False)
    n_training_words = 0
    occurances = Counter()
    training_queue = []
    outfile = "cobweb_{}_holmes_out".format(window)

    # Build up a CSV file summarizing the information of each instance and the performance of Cobweb after learning each instance
    with open(outfile + ".csv", 'w') as fout:
            fout.write("n_training_words,n_training_stories,model,word,word_freq,word_obs_count,vocab_size,pred_word,prob_word,correct,story\n")

    for idx, (instance, story_idx, anchor_idx) in enumerate(tqdm(instances)):
        anchor = list(instance['anchor'].keys())[0]

        # Story as a sentence completion form
        story = stories[story_idx]
        text = " ".join([w for w in story[max(0, anchor_idx-window):anchor_idx]])
        text += " _ "
        text += " ".join([w for w in story[max(0, anchor_idx+1):anchor_idx+window+1]])

        instance_no_anchor = {'context': instance['context']}
        probs_pred = tree.predict_probs(instance_no_anchor, 100, False, False, 1)
        prob_pred = 0
        anchor_pred = "NONE"
        if "anchor" in probs_pred and anchor in probs_pred['anchor']:
            prob_pred = probs_pred['anchor'][anchor]
        if "anchor" in probs_pred and len(probs_pred["anchor"]) > 0:
            anchor_pred = sorted([(probs_pred["anchor"][word], random(), word) for word in probs_pred['anchor']], reverse=True)[0][2]

        training_queue.append(instance)
        with open(outfile + ".csv", "a") as fout:
            fout.write("{},{},cobweb,{},{},{},{},{},{},{},{}\n".format(
                    n_training_words,  # n_training_words
                    story_idx,  # n_training_stories
                    anchor,  # word
                    overall_freq[anchor],  # word_freq
                    occurances[anchor],  # word_obs_count
                    len(occurances),  # vocab_size
                    anchor_pred,  # pred_word
                    prob_pred,  # prob_word
                    1 if anchor_pred == anchor else 0,  # correct
                    text  # story
                    ))

        if len(training_queue) > 0:
                old_inst = training_queue.pop(0)
                tree.ifit(old_inst)
                old_anchor = list(old_inst['anchor'].keys())[0]
                occurances[old_anchor] += 1
                n_training_words += 1

We can then derive a summary result table `cobweb_10_holmes_out.csv`. The first rows are as follows:

| n_training_words  |  n_training_stories | model |  word  |  word_freq  | word_obs_count | vocab_size | pred_word |  prob_word  | correct | story |
| -----  |  ----- | ----- |  -----  |  -----  | ----- | ----- | ----- |  -----  | ----- | -------------------------------------------- |
| 0  | 1  | cobweb  | rock  |  102 | 0 |  0  | NONE  |  0  | 0  | far like hero ll ship like sea rock ship hand _ ship like rock rock god man ship hero time sea |
| 1 |  1  | cobweb | thou  |  184| 0  | 1   |rock  |  0  | 0  | hero ship land son hand ll bring come let ship _ thy god thou take far ll spake heart son fall|
| 2  | 0   |cobweb | take  |  142| 0  | 2  | thou |   0  | 0  | adam word god come day adam god fall night adam _ eve return cave see fall adam eve adam say eve|

Here are the descriptions of the result table columns:

| Column | Description |
| --- | ----------- |
| `n_training_words` | The number of instances used to trained in the iteration, or simply the iteration number |
| `n_training_stories` | The index of the story that the instance is from |
| `model` | Model used in the iteration. The value for this example should be `cobweb` |
| `word` | The ground-truth anchor word of the tested instance |
| `word_freq` | The frequency of the anchor word in the stories processed |
| `word_obs_count` | The encountering times of the anchor word so far |
| `vocab_size` | The number of anchor words learned by Cobweb so far |
| `pred_word` | The anchor word prediction for the instance |
| `prob_word` | The predicted probability of the anchor word prediction made |
| `correct` | Whether the prediction is correct. 0 if incorrect, 1 if correct |
| `story` | The corresponding n-gram of the tested instance |


Further, you can derive the evolved learning curve of Cobweb based on the result table by entering the following in the terminal/command line in the current directory:

    python3 plot_curve.py cobweb_10_holmes_out.csv

Like the figure in the following:

<figure>
    <img src="./example-learning-curve.png"
         alt="Example Learning Curve Visualization">
    <figcaption>Example of the output learning curve figure. Top: Number of instances learned vs. Predicted probability of the anchor word of the most recent instance. Bottom: Number of instances learned vs. Accuracy of the batch of instances </figcaption>
</figure>

Which are the anchor word predicted probability evolutions and the accuracy evolutions summarized every `window_size=100` instances.

----------------------

To see how Cobweb is implemented, please direct to the `README.md` [here](https://github.com/Teachable-AI-Lab/cobweb).



