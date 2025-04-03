"""
NAME: data_loader.py
DESCRIPTION: python file meant to load dataset of images for sign language recognition
PROGRAMMER: Caidan Gray
CREATION DATE: 3/3/2025
LAST EDITED: 4/3/2025   (please update each time the script is changed)
"""

from os import walk
import torch
import torchvision
import torch.utils.data as data
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import import_data


class Data_Loader():
    def __init__(self):
        """initialize the variables"""
        super(Data_Loader, self).__init__()
        self.import_data = import_data.Import_Data(""" Matteo fill this in with the database prams (input is multiple strings)""")
        self.training_data = self.import_data.get_training_data('training')
        self.testing_data = self.import_data.get_testing_data('testing')

    def train_loader(self):
        """load the Training data"""

        return 0

    def test_loader(self):
        """Load the testing data"""

        return 0