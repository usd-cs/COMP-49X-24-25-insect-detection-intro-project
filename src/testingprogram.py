

class TestingProgram:
    
    def __init__(self):
        testImage = None


    def takeInput():
        """
        Takes user's terminal input of a file path and opens the image if it exists for processing

        Returns: True if the file exists, false if not
        """
        filePath = input("Please input a file path for an image to process: ")

        try:
            f = open(filePath, 'r')
            testImage = f
            return True

        except:
            print("File not found")
            return False
        
    def transformImage(image):
        """
        Transforms input image to match other inputs based on guidelines established by training program

        Returns: transformed image
        """
        pass

    def classifyImage():
        """
        Runs image through testing program and returns results

        Returns: (Letter identified, Percent accuracy)
        """
        pass

    def classificationStub(image):
        """
        Temporary method placeholder that will be replaced by actual classification functionality.
        Serves only for testing purposes
        """
        pass


if __name__ == "__main__":
    tp = TestingProgram()
    tp.takeInput