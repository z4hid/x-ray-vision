# pretrainedmodel.py
import torch.nn as nn
from torchvision import models

def get_pretrained_model():
    """
    This function returns a ResNet34 model with a custom fully connected layer.
    """
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Adjust the final layer for binary classification
    return model
