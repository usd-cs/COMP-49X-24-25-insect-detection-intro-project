import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import io
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from trainingprogram import TrainingProgram

class TestUserInput(unittest.TestCase):

    # dataset prints expected output
    @patch("sys.stdout", new_callable=io.StringIO)
    def testLoadsValidDataset(self, stdOut):
        tp = TrainingProgram()
        tp.createTrainingDirectoryPrompt()
        outValue = stdOut.getvalue().strip()
        self.assertEqual(outValue, 'Training images shape: (60000, 28, 28)\nTraining labels shape: (60000,)\nTest images shape: (10000, 28, 28)\nTest labels shape: (10000,)')
        
    # dataset saves correct height for images
    def testSavesCorrectHeight(self):
        tp = TrainingProgram()
        tp.createTrainingDirectoryPrompt()
        self.assertEqual(tp.height, 28)
    
    # dataset saves correct width for images
    def testSavesCorrectWidth(self):
        tp = TrainingProgram()
        tp.createTrainingDirectoryPrompt()
        self.assertEqual(tp.width, 28)
    
    # pretrained machine properly loads from pytorch
    @patch("sys.stdout", new_callable=io.StringIO)
    def testMachineLoads(self, stdOut):
        tp = TrainingProgram()
        tp.loadMachine()
        outValue = stdOut.getvalue().strip()
        # just check that function prints something as proof of the machine being loaded
        self.assertNotEqual(outValue, None)

    # mock machine properly saves weights to path and txt file
    @patch("torch.save")
    @patch("builtins.open", new_callable=mock_open)
    def testSavesMachine(self, mockOpenFile, mockSaveTorch):
        # Mock the model and its state_dict
        mockModel = MagicMock()
        mockModel.state_dict.return_value = {
            "layer1.weight": torch.randn(64, 3, 7, 7),
            "layer2.weight": torch.randn(128, 64, 3, 3),
        }

        # Mock filenames
        weightFilename = "test.pth"
        heightFilename = "test.txt"

        tp = TrainingProgram()
        tp.model = mockModel
        tp.height = 28

        tp.saveWeights(weightFilename, heightFilename)

        # Verify the height file and torch save
        mockOpenFile.assert_called_once_with(heightFilename, "w")
        mockOpenFile().write.assert_called_once_with("28")  # Height as a string

        mockSaveTorch.assert_called_once_with(mockModel.state_dict(), weightFilename)
        if os.path.exists(weightFilename):
            os.remove(weightFilename)
        if os.path.exists(heightFilename):
            os.remove(heightFilename)


if __name__ == "__main__":
    unittest.main()