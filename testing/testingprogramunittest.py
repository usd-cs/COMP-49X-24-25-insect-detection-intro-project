import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import pandas as pd
import torch
import torch.nn.functional as F
import time
import ast

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from testingprogram import TestingProgram

class TestUserChoice(unittest.TestCase):
    
    @patch('builtins.input', return_value = "B.png")
    def testValidUserInput(self, mock_input):
        tp = TestingProgram(10)                 #height of 10 is a placeholder, has no consequences
        result = tp.takeInput()
        self.assertEqual(result, True)

    @patch('builtins.input', return_value = "DoesnotExist")
    def testInvalidUserInput(self, mock_input):
        tp = TestingProgram(10)
        result = tp.takeInput()
        self.assertEqual(result, False)

    @patch.object(TestingProgram, 'classificationStub', return_value = ('A', .96))
    def testCorrectLetterOutput(self, mock_method):
        tp = TestingProgram(10)
        (result, accuracy) = tp.classifyImage()
        self.assertEqual(result, 'A')

    @patch.object(TestingProgram, 'classificationStub', return_value = ('A', .96))
    def testCorrectAccuracyOutput(self, mock_method):
        tp = TestingProgram(10)
        (result, accuracy) = tp.classifyImage()
        self.assertEqual(accuracy, .96)

    def testLoadModelWeights(self):
        # Create TestingProgram object and call loadModelWeights method
        my_instance = TestingProgram()

        my_instance.loadModelWeights('weights.csv') # with trained weights file

        weights_df = pd.read_csv('weights.csv', skiprows=1)

        weight_row_conv1 = weights_df[weights_df['parameter_name'] == "conv1.weight"]
        expected_conv1 = ast.literal_eval(weight_row_conv1['values'].values[0])[0]

        actual_conv1 = my_instance.model.conv1.weight.data[0, 0, 0, 0].item()

        # Compare loaded weights to expected values
        self.assertAlmostEqual(
            actual_conv1,
            expected_conv1,
            places=5,
            msg=f"Expected conv1.weight[0,0,0,0] to be {expected_conv1}, but got {actual_conv1}"
        )

        
    @patch('torchvision.models.resnet18')
    @patch.object(TestingProgram, 'loadModelWeights')
    @patch.object(TestingProgram, 'transformImage')
    def testClassificationStub(self, mock_transformImage, mock_loadModelWeights, mock_resnet):
        # Mock the model returned by resnet18
        mock_model = MagicMock()
        mock_resnet.return_value = mock_model

        # Mock the output of the model
        mock_output = torch.tensor([[0.1, 0.2, 0.7]])
        mock_model.return_value = mock_output

        # Mock the transformImage method to return a mock tensor (processed image)
        mock_transformImage.return_value = torch.tensor([[0.7]])

        my_instance = TestingProgram(10)

        predicted_class, confidence = my_instance.classificationStub()

        # Check that the classification results are correct
        self.assertEqual(predicted_class, 2)

        # Apply softmax manually to calculate the expected confidence and assertAlmostEqual
        softmax_output = F.softmax(mock_output, dim=1)
        expected_confidence = softmax_output[0][2].item()

        self.assertAlmostEqual(confidence, expected_confidence, places=2)

    @patch('builtins.input', return_value="k.png")
    @patch.object(TestingProgram, 'classifyImage', return_value=("A", 0.95))
    def testMainCompletesInTime(self, mock_classify, mock_input):
        
        # Simulate running the main code
        testingProgram = TestingProgram(28)

        testingProgram.takeInput()  # Simulate user input with mocked input

        # pass the trained weights(weights.csv) into the loadModelWeights method
        weightsFile = "weights.csv"
        testingProgram.loadModelWeights(weightsFile)

        start_time = time.time()
        
        predictedCharacter, confidenceScore = testingProgram.classifyImage()
        print(f"We identified the image to be the character: {predictedCharacter}")
        print(f"We have confidence of {confidenceScore}")

        end_time = time.time()
        
        # Check that the execution time is less than or equal to 10ms
        execution_time = (end_time - start_time) * 1000
        self.assertLessEqual(execution_time, 10, f"Main function took too long: {execution_time}ms")

if __name__ == "__main__":
    unittest.main()