import os
from pathlib import Path
from tensorflow.keras.datasets import mnist
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np

class TrainingProgram:
    
    def __init__(self):
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.height = 0
        self.width = 0

    def createTrainingDirectoryPrompt(self):
        (self.trainX, self.trainY), (self.testX, self.testY) = mnist.load_data()
        self.height = self.trainX.shape[1]
        self.width = self.trainX.shape[2]
        print(f'Training images shape: {self.trainX.shape}')
        print(f'Training labels shape: {self.trainY.shape}')
        print(f'Test images shape: {self.testX.shape}')
        print(f'Test labels shape: {self.testY.shape}') 

if __name__ == "__main__":
    tp = TrainingProgram()
    tp.createTrainingDirectoryPrompt()