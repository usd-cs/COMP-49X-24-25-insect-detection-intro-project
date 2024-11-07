import unittest
from unittest.mock import patch
import io
import sys
import os

from testingprogram import TestingProgram

class TestUserChoice:
    
    @patch('builtins.input', return_value = "README.md")
    def testValidUserInput(self, mock_input):
        tp = TestingProgram()
        (result, fileObject) = tp.takeInput()
        self.assertEqual(result, 1)

    @patch('builtins.input', return_value = "DoesnotExist")
    def testInvalidUserInput(self, mock_input):
        tp = TestingProgram()
        (result, fileObject) = tp.takeInput()
        self.assertEqual(result, 0)

    
        