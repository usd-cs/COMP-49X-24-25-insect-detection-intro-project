from torchvision import models, transforms
import torch
import pandas as pd
import ast
from PIL import Image

class TestingProgram:
    
    def __init__(self, height = None):
        self.testImage = None
        if height == None:
            self.height = 240 # set this to whatever default value we want
        self.height = height
        
        # Initialize a fresh model with weights = None, so there is no weights
        self.model = models.resnet18(weights=None)
        numFeatures = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(numFeatures, 10)
        # use CPU to allow general usability and Metal Performance Shader if user has Apple Silicon
        self.device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
        self.model = self.model.to(self.device)

    def takeInput(self):
        """
        Takes user's terminal input of a file path and opens the image if it exists for processing

        Returns: True if the file exists, false if not
        """
        filePath = input("Please input a file path for an image to process: ")

        try:
            f = Image.open(filePath)
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
            # transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(self.height),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # mean and stdev for RGB channels
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

        self.model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
        self.model.eval()

        # Process the self.image object 
        processedImage = self.transformImage()

        # Use torch.no_grad() to avoid gradient computation for inference
        with torch.no_grad():
            output = self.model(processedImage)

        # Get the predicted class and confidence score
        _, predictedIndex = torch.max(output, 1) # ignore maximum value, but get its index
        confidenceScore = torch.nn.functional.softmax(output, dim=1)[0][predictedIndex].item()
        # predictedCharacter = chr(predictedIndex.item())
        predictedCharacter = predictedIndex.item()

        # Return the predicted class and confidence score
        return predictedCharacter, confidenceScore
    

    def loadModelWeights(self, weightsFilePath = None):
        """
        Loads the specified model with weights stored in the file specified by filename.

        Returns: None
        """
        if weightsFilePath == None:
            weightsFilePath = input("Please input the file path of the saved weights for the trained model: ")

        # Load weights from CSV into a DataFrame
        try:
            weightFile = open(weightsFilePath, 'r')
            self.height = int(weightFile.readline().strip())
            weights_df = pd.read_csv(weightsFilePath, skiprows=1)
        except FileNotFoundError:
            print("Model Weights File Does Not Exist. Run Testing Program")
            return
            
        # Iterate through each parameter in the model
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Find the corresponding row in the DataFrame
                weight_row = weights_df[weights_df['parameter_name'] == name]
                
                if not weight_row.empty:
                    # Convert the 'values' string to a list of floats
                    weight_array = ast.literal_eval(weight_row['values'].values[0])

                    if len(weight_array) != param.numel():
                        print(f"Error: The number of elements does not match for {name}. Skipping this parameter.")
                        continue
                    # Reshape if necessary
                    weight_array = [float(x) for x in weight_array]  # make sure all values are floats
                    reshaped_weight = torch.tensor(weight_array).view(param.shape)

                    param.data = reshaped_weight


        torch.save(self.model.state_dict(), 'model_weights.pth')


if __name__ == "__main__":
    testingProgram = TestingProgram()
    print("\n\n------------   Welcome to Character Identifier!   ------------\n\n")

    # pass the fresh model with the the trained weights into the loadModelWeights method
    testingProgram.loadModelWeights()

    while(True):
        testingProgram.takeInput()
        predictedCharacter, confidenceScore = testingProgram.classifyImage()
        # predictedCharacter = chr(predictedCharacter)
        confidenceScore = round(confidenceScore * 100, 2)
        print(f"\nWe identified the image to be the character: {predictedCharacter}")
        print(f"We have confidence of {confidenceScore}\n")
        try:
            while True:
                user_input = input("Enter 1 if you would like to identify another image: \nEnter 0 if you are done: ").strip()
        
                if user_input.isdigit():
                    again = int(user_input)

                    if again == 1:
                        break
                    elif again == 0:
                        print("Thank You, Goodbye.")
                        exit()
                    else:
                        print("Character entered was not a 1 or 0, please try again.")
                else:
                    print("Invalid input. Please enter a valid number (1 or 0).")
        except ValueError:
            print("Invalid input. Please enter a valid number (1 or 0).")