import os
from pathlib import Path
from tensorflow.keras.datasets import mnist
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np

# Custom dataset class for DataLoader object (for training input)
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    # allows image and label access by index + transform
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

class TrainingProgram:
    
    def __init__(self):
        # initialize class vars
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.height = 0
        self.width = 0
        self.model = None
        self.device = None
        self.transform = None

    def createTrainingDirectoryPrompt(self):
        # load mnist dataset from keras
        (self.trainX, self.trainY), (self.testX, self.testY) = mnist.load_data()
        # take height and width of images from shape property
        self.height = self.trainX.shape[1]
        self.width = self.trainX.shape[2]
        # print shape values of each set
        print(f'Training images shape: {self.trainX.shape}')
        print(f'Training labels shape: {self.trainY.shape}')
        print(f'Test images shape: {self.testX.shape}')
        print(f'Test labels shape: {self.testY.shape}') 

    def loadMachine(self):
        # load model with set number of classifications for the last layer, set loss function, and type optimization
        self.model = models.resnet18()
        numFeatures = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(numFeatures, 10)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(self.height),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # mean and stdev for RGB channels
        ])
        # use CPU to allow general usability and Metal Performance Shader if user has Apple Silicon
        self.device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
        self.model = self.model.to(self.device)
        # proof of loaded machine
        print(f'Final Layer Biases: {self.model.fc.bias.data}')
    
    def trainMachine(self, numEpochs):
        self.model.train()
        # define loss function, optimization function, and image transformation
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        dataset = CustomDataset(self.trainX, self.trainY, transform=self.transform)
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=True) #lower batch size to prevent memory overload
        for epoch in range(numEpochs):
            runningLoss = 0.0
            for inputs, labels in dataLoader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Clear previous gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backpropagation pass
                loss.backward()
                optimizer.step()

                runningLoss += loss.item()
            print(f"Epoch {epoch+1}/{numEpochs}, Loss: {runningLoss/len(dataLoader):.4f}")
    
    def testMachine(self):
        # load testing dataset
        dataset = CustomDataset(self.testX, self.testY, transform=self.transform)
        dataLoader = DataLoader(dataset, batch_size=32, shuffle=False)
        # evaluate testing machine
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataLoader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy: {100 * correct / total:.2f}%")

    def saveWeights(self, filename):
        weights = []

        # check if file exists
        file_obj = os.path.exists(filename)
        if not file_obj:
            with open(filename, "w") as file:
                file.write('')
            print(f"File, {filename}, created.")
            
        # Iterate layers and save weights
        for name, param in self.model.named_parameters():
            weights.append({
                "parameter_name": name,
                "values": param.detach().cpu().numpy().flatten().tolist()
            })
        
        # store into pandas dataframe to store in csv
        df = pd.DataFrame(weights)
        df.to_csv(filename, index=False)
        print(f"Model weights saved to {filename}")
        

if __name__ == "__main__":
    tp = TrainingProgram()
    tp.createTrainingDirectoryPrompt()
    tp.loadMachine()
    tp.trainMachine(5)
    tp.testMachine()
    filename = input("Please enter a file to save weights to (csv format): ")
    tp.saveWeights(filename)