from torchvision import models, transforms
import torch
from PIL import Image
import time

class TestingProgram:
    
    def __init__(self, height = None):
        """Initialize class variables. User can specify height parameter for model."""
        self.testImage = None
        self.height = height
        
        # Initialize a fresh model with weights = None, so there is no weights
        self.model = models.resnet18(weights=None)
        numFeatures = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(numFeatures, 10)
        # use CPU to allow general usability and Metal Performance Shader if user has Apple Silicon

        self.device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
        self.model = self.model.to(self.device)

    def takeInput(self):
        """
        Takes user's terminal input of a file path and opens the image if it exists for processing

        Returns: True if the file exists, false if not
        """
        filePath = input("Please input a file path for an image to process: ")

        try:
            f = Image.open(filePath).convert('RGB')
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
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(self.height),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]), # mean and stdev for RGB channels
        ])

        transformedImage = transform(self.testImage)

        # Add batch dimension (from [C, H, W] to [1, C, H, W])
        transformedImage = transformedImage.unsqueeze(0)

        return transformedImage

    def classifyImage(self):
        """
        Runs image through testing program and returns results

        Returns: (Letter identified, Percent accuracy)
        """
        
        return self.classificationStub()

    def classificationStub(self):
        """
        Preforms an inference on self.image using the loaded trained model

        Returns: (Letter identified, Percent accuracy)
        """

        # set the model to evaluation mode
        self.model.eval()

        # Process the self.image object 
        processedImage = self.transformImage()
        processedImage = processedImage.to(self.device)

        # Use torch.no_grad() to avoid gradient computation for inference
        with torch.no_grad():
            output = self.model(processedImage)

        # Get the predicted class and confidence score
        _, predictedIndex = torch.max(output, 1)
        confidenceScore = torch.nn.functional.softmax(output, dim=1)[0][predictedIndex].item()
        predictedCharacter = predictedIndex.item()

        return predictedCharacter, confidenceScore
    

    def loadModelWeights(self, weightsFilePath = None, heightFilePath = None):
        """
        Loads the specified model with weights and height stored in each file specified by filename.

        Returns: None
        """
        if weightsFilePath == None:
            weightsFilePath = input("Please input the file path of the saved weights for the trained model: ")
        if heightFilePath == None:
            heightFilePath = input("Please input the file path of the saved height for the image input: ")
        # Load weights from pth into the model
        try:
            heightFile = open(heightFilePath, 'r')
            self.height = int(heightFile.readline().strip())
            self.model.load_state_dict(torch.load(weightsFilePath, weights_only=True))
        except FileNotFoundError:
            print("Model Weights File Does Not Exist. Run Testing Program")
            return


if __name__ == "__main__":
    testingProgram = TestingProgram()
    print("\n\n------------   Welcome to Digit Identifier!   ------------\n\n")

    testingProgram.loadModelWeights()

    while(True):
        testingProgram.takeInput()
        startTime = time.time()
        predictedCharacter, confidenceScore = testingProgram.classifyImage()
        endTime = time.time()
        executionTime = round((endTime - startTime) * 1000, 2)
        confidenceScore = round(confidenceScore * 100, 2)
        print(f"\nWe identified the image to be the digit: {predictedCharacter}")
        print(f"We have a confidence of {confidenceScore}\n")
        print(f"Digit Identified in {executionTime} ms\n")
        try:
            while True:
                userInput = input("Enter 1 if you would like to identify another image: \nEnter 0 if you are done: ").strip()
        
                if userInput.isdigit():
                    again = int(userInput)

                    if again == 1:
                        break
                    elif again == 0:
                        print("\nThank You, Goodbye.")
                        exit()
                    else:
                        print("\nCharacter entered was not a 1 or 0, please try again.")
                else:
                    print("\nInvalid input. Please enter a valid number (1 or 0).")
        except ValueError:
            print("\nInvalid input. Please enter a valid number (1 or 0).")