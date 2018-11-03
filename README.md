# Dog-Breed-Classification

A Jupyter notebook created to complete Dog Breed Identification task on Kaggle ([link here](https://www.kaggle.com/c/dog-breed-identification/)).

The project is supported with free Tesla K80 GPU provided by [Google Colaboratory](https://colab.research.google.com/) with a runtime of 12 hours.

The link to the Colab notebook is present in the repository's notebook.

### About the dataset

The dataset is strict canine subset of the ImageNet dataset given to perform image classification task. It contains 120 classes of dogs with seperate train and test dataset.

### Approach

Convolutional Neural Networks (CNNs) have proven to be quite effective in computer vision tasks like image classification, detection and segmentation. Hence, a deep learning method involving CNNs has been used to deal with this task.

ResNet CNN architecture ([paper link](https://arxiv.org/abs/1512.03385)) has been used to train and evaluate the approach. In this project, both ResNet18 and ResNet50 have been used under various settings to analyse the performance of the model on image classification task.

Apart from CNN, other standard settings included use of Stochastic Gradient Descent with momentum as model optimizer and Cross-Entropy loss.

### Data Augmentation

Data augmentation techniques like resize and normalize were used. For training dataset, horizontal flips were used to make the model more robust to variations.

During one of the experiments, Random Crop was also used for pure performance analysis.

### Settings

The common settings included:

  1. Learning Rate: 0.001
  2. Momentum: 0.9
  3. Input Size: 224 x 224
  
For experiments using only resize, only 20 epochs were performed for training. With Random Crop, 100 epochs were performed.

### Modules

The following modules are needed to execute the code (the Python notebook handles installation of all modules):

  1. PyTorch (Deep Learning Framework)
  2. Numpy
  3. Matplotlib
  4. Kaggle (for downloading dataset with _kaggle.json_ file.)
  5. TensorboardX (to obtain graphs)
  6. Torchvision (for pre-trained models)
  
### Observations

The following observations were made during the experiments:
  
  1. ResNet18 architecture when used **without random crop** achieved _training_ accuracy of **99.55** percent and loss of **0.033** within 20 epochs.
  2. ResNet50 architecture with the above settings achieved _training_ accuracy of **99.38** percent and loss of **0.039**.
  3. ResNet18 when used **with random crop** images showed _training_ accuracy of **87.00** percent and loss of **0.518** respectively
  
### Graphs

The below graphs show a comparative analysis of training performance under different settings and architecture.

1. Training Accuracy

![Training Accuracy](https://github.com/radonys/Dog-Breed-Classification/blob/master/graphs/Training_Accuracies.png "Training Accuracy")

2. Training Loss

![Training Loss](https://github.com/radonys/Dog-Breed-Classification/blob/master/graphs/Training_Loss.png "Training Loss")

### Evaluation Results

The best evaluation results are reported below (as obtained on Kaggle):

  1. ResNet18 (11th epoch, no random crop): 1.05815
  2. ResNet50 (20th epoch, no random crop): 1.07014
  3. ResNet18 (41st epoch, random crop): 1.63381
