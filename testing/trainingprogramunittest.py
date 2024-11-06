import unittest
import sys
import os
from unittest.mock import patch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from trainingprogram import TrainingProgram

class TestUserInput(unittest.TestCase):
    @patch("builtins.input", return_value="trainingfiles")
    def test_valid_file(self, mock_input):
        tp = TrainingProgram()
        self.assertEqual(tp.trainingDirectoryPrompt(), "Directory exists.")
    
    @patch("builtins.input", return_value="invalid")
    def test_invalid_file(self, mock_input):
        tp = TrainingProgram()
        self.assertEqual(tp.trainingDirectoryPrompt(), "Directory does not exist.")
    

if __name__ == "__main__":
    unittest.main()