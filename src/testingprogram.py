from torchvision import models, transforms
import torch

class TestingProgram:
    
    def __init__(self, height):
        self.testImage = None
        self.height = height

    def takeInput(self):
        """
        Takes user's terminal input of a file path and opens the image if it exists for processing

        Returns: True if the file exists, false if not
        """
        filePath = input("Please input a file path for an image to process: ")

        try:
            f = open(filePath, 'r')
            self.testImage = f
            return True

        except:
            print("File not found")
            return False
        
    def transformImage(self):
        """
        Transforms input image to match other inputs using transformation defined by training algorithm

        Returns: transformed image
        """

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(self.height),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # mean and stdev for RGB channels
        ])

        transformedImage = transform(self.testImage)

        return transformedImage

    def classifyImage(self):
        """
        Runs image through testing program and returns results

        Returns: (Letter identified, Percent accuracy)
        """

        return self.classificationStub(self.testImage)

    def classificationStub(self, image):
        """
        Temporary method placeholder that will be replaced by actual classification functionality.
        Serves only for testing purposes
        """
        pass
