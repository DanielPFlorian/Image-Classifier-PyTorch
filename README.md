# Image-Classifier-PyTorch
[PyTorch](https://pytorch.org/) implementation of a Deep Neural Network image classifier to predict 102 species of flowers.

Check out the model demo [here](https://huggingface.co/spaces/DanielPFlorian/Flower-Image-Classifier) on Hugging Face.

Just upload an image of a flower and hit "Submit".

The trained model will try to classify what type of flower it is.

### Pytorch Model Overview

The dataset the model is trained on comes from the
[oxford_flowers102 dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102).
To efficiently train, validate and test the Pytorch model, a custom Pytorch class is
created and training functions are implemented to test out various hyperparameters in order to achieve the best accuracy.

Image data is transformed, augmented,and normalized, then batch fed to the model. To increase the accuracy of predictions and extract features from the images, the pretrained
[MAXVIT_T](https://pytorch.org/vision/main/models/generated/torchvision.models.maxvit_t.html)
neural network is utilized after which a classifier of fully connected linear layers with relu activation units and a final softmax layer are used. Dropout is also used to help prevent overfitting.

The [Image_Classifier_Pytorch_Demo.ipynb](https://github.com/DanielPFlorian/Image-Classifier-PyTorch/blob/main/Image_Classifier_Pytorch_Demo.ipynb) jupyter notebook file contains the code and steps used for model creation and
evaluation.

A technique to achieve even higher accuracy than just training the model's classifier layers alone is also shown in the notebook. Deeper blocks of the pretrained model's layers are trained at increasingly lower learning rates in order to only slightly nudge the pretrained weights during training with the flower dataset. A learning rate multiplier function is defined to easily change the learning rate of the deeper layers. With this technique the model achieves an accuracy of 98.59% on the test dataset.

### Techniques Used

- Pytorch
- Image normalization/pre-processing
- Image Augmentation
- Training, Validation and Testing Functions created
- Custom Pytorch Class for Model Classifier Implementation
- Pre-Trained Neural Networks
- Hyperparameter Optimization
- Overfitting prevention using Dropout
- Image Recognition Inference using a deep learning neural network trained Model
