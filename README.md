# Insect Detection - Introduction Project: Character Recognition
## By: Joe Cox, Daniel Daugbjerg, and Thomas McKeown***

### Description:
Our application is a character recognizer. There are two main python files with this application, trainingprogram.py and testingprogram.py. The trainingprogram.py is used to train a model and save the height and weights of the model to designated files. After this, the testingprogram.py uses these files to load a trained model for classification. Then the application asks for the users input of the path to an image they would like to be classified. This image is then evaluated by the model and it outputs its prediction and confidence score.

### Required Software:
In order to run this application you must download numpy, torchvision, torch, and tensorflow

### Running the application:
1) ***First run trainingprogram.py*** This will begin to train the model with the keras mnist character dataset
2) After the this program has completed training it will promt the user for the file to save the wieghts to(.pth format)
3) Similarly it will ask the user to enter the file they would like to save the image hieght to. Then this program is complete.
4) ***Next run testingprogram.py***
5) The user will be promted to input the file path of the saved weights for the trained model, so enter the path that was used for saving the weights from the trainingprogram.py.
6) Next the user will be promted to input the file path of the saved height for the image input, so enter the path that was used for saving the height from the trainingprogram.py.
7) The user will be promted to input a file path for an image to be classified by the model.
8) The program will then output the predicted digit, the confidence, and the time it took for the model to identify the image.
9) Lastly, the user will be promted to either identify another image by inputting 1, or input 0 to stop the application.