# Dog-Breed-Classification

A Jupyter notebook created to complete Dog Breed Identification task on Kaggle ([link here](https://www.kaggle.com/c/dog-breed-identification/)).

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

### Observations

The following observations were concluded 
