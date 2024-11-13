import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import torch
from io import StringIO
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from testingprogram import TestingProgram

class TestUserChoice(unittest.TestCase):

    @patch('builtins.input', return_value="dummy_path.png")
    @patch("PIL.Image.open", return_value=mock_open())  # Mock Image.open
    def testValidUserInput(self, mockOpenFile, mockInput):
        tp = TestingProgram(10)  # Height of 10 is a placeholder, has no consequences
        
        # return mock image object when Image.open is called
        mockOpenFile.return_value = Image.new('RGB', (100, 100))
        
        result = tp.takeInput()

        # Assert that the mock open was called with dummy_path.png and assert takeInput() returns true
        mockOpenFile.assert_called_with("dummy_path.png")
        self.assertEqual(result, True)

    @patch('builtins.input', return_value = "DoesnotExist")
    def testInvalidUserInput(self, mockInput):
        tp = TestingProgram(10)
        result = tp.takeInput()
        self.assertEqual(result, False)

    @patch.object(TestingProgram, 'classificationStub', return_value = ('A', .96))
    def testCorrectLetterOutput(self, mockMethod):
        tp = TestingProgram(10)
        (result, accuracy) = tp.classifyImage()
        self.assertEqual(result, 'A')

    @patch.object(TestingProgram, 'classificationStub', return_value = ('A', .96))
    def testCorrectAccuracyOutput(self, mockMethod):
        tp = TestingProgram(10)
        (result, accuracy) = tp.classifyImage()
        self.assertEqual(accuracy, .96)

    @patch("builtins.open", new_callable=mock_open, read_data="224\n")
    @patch("torch.load")  # Mock torch.load for simulating loading model weights
    def testLoadModelWeights(self, mockTorchLoad, mockOpenFile):
        
        testingInstance = TestingProgram()
        testingInstance.model = MagicMock()
        heightFilePath = "mock_height.txt"
        weightsFilePath = "mock_weights.pth"
        
        # Mock the return value of torch.load to a mocked weight dictionary
        mockTorchLoad.return_value = {"mock_key": torch.tensor([1.0])}
        
        testingInstance.loadModelWeights(weightsFilePath, heightFilePath)
        
        # Check that the height was set correctly, the torch.load was called with the weightsFilePath, 
        # and the load_state_dict was called on the model
        self.assertEqual(testingInstance.height, 224)
        mockTorchLoad.assert_called_once_with(weightsFilePath, weights_only=True)
        testingInstance.model.load_state_dict.assert_called_once_with(mockTorchLoad.return_value)


    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('sys.stdout', new_callable=StringIO)
    def testLoadModelWeightsFileNotFoundMessage(self, mockStdout, mockOpen):
        testingInstance = TestingProgram()
        
        # Call loadModelWeights with false file paths
        testingInstance.loadModelWeights("nonexistent_weights.pth", "nonexistent_height.txt")
        
        # Assert that the correct error message was printed
        self.assertIn("Model Weights File Does Not Exist. Run Testing Program", mockStdout.getvalue())

if __name__ == "__main__":
    unittest.main()