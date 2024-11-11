import unittest
from unittest.mock import patch
import io
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from testingprogram import TestingProgram

class TestUserChoice(unittest.TestCase):
    
    @patch('builtins.input', return_value = "README.md")
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

        
if __name__ == "__main__":
    unittest.main()