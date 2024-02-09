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

To see how Cobweb/4V is implemented, please direct to the `README.md` [here](https://github.com/Teachable-AI-Lab/cobweb).
