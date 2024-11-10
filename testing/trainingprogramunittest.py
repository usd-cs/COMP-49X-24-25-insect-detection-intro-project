import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import io
import sys
import os

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

    # mock machine properly saves weights to csv file
    @patch("sys.stdout", new_callable=io.StringIO)
    def testSavesMachine(self, stdOut):
        # create mock machine model
        mockModel = MagicMock()
        mockModel.named_parameters.return_value = [
            ("layer1.weight", MagicMock(data=np.random.rand(64, 3, 7, 7))),
            ("layer2.weight", MagicMock(data=np.random.rand(128, 64, 3, 3))),
        ]
        tp = TrainingProgram()
        tp.model = mockModel
        tp.saveWeights('test.csv')
        outValue = stdOut.getvalue().strip()
        self.assertEqual(outValue, "File, test.csv, created.\nModel weights saved to test.csv")
        # remove csv file after testing
        if os.path.exists('test.csv'):
            os.remove('test.csv')


if __name__ == "__main__":
    unittest.main()