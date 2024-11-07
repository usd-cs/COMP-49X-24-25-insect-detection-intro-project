

class TestingProgram:
    
    def __init__(self):
        pass

    
    def takeInput():
        filePath = input("Please input a file path for an image to process: ")

        try:
            f = open(filePath, 'r')

        except:
            print("File not found, ")


if __name__ == "__main__":
    tp = TestingProgram()
    tp.takeInput