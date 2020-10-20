
import torch
import torchvision
import numpy as np
from PIL import Image

from .config import Config

class Predict:
    """ predicts the image to be toy or product """
    def __init__(self, image_path):
        """  initilize the preprocess path """
        self.image_path = image_path
        self.image = None


    def preprocess_image(self):
        # Load Image
        img = Image.open(self.image_path)
        
        # Get the dimensions of the image
        width, height = img.size
        
        # Resize by keeping the aspect ratio, but changing the dimension
        # so the shortest size is 255px
        img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
        
        # Get the dimensions of the new image size
        width, height = img.size
        
        # Set the coordinates to do a center crop of 224 x 224
        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2
        img = img.crop((left, top, right, bottom))
        
        # Turn image into numpy array
        img = np.array(img)
        
        # Make the color channel dimension first instead of last
        img = img.transpose((2, 0, 1))
        
        # Make all values between 0 and 1
        img = img/255
        
        # Normalize based on the preset mean and standard deviation
        img[0] = (img[0] - 0.485)/0.229
        img[1] = (img[1] - 0.456)/0.224
        img[2] = (img[2] - 0.406)/0.225
        
        # Add a fourth dimension to the beginning to indicate batch size
        img = img[np.newaxis,:]
        
        # Turn into a torch tensor
        image = torch.from_numpy(img)
        image = image.float()
        self.image = image
        return image

    def predict(self, image, model):
            """ predicts the passed to the model """
            # Pass the image through our model
            output = model.forward(image)
            
            # Reverse the log function in our output
            output = torch.exp(output)
            
            # Get the top predicted class, and the output percentage for
            # that class
            probs, classes = output.topk(1, dim=1)
            return probs.item(), classes.item()   


if __name__ == "__main__":
    """ class 0: Product | class 1: toys """
    # Give image to model to predict output
    
    pred = Predict("datasets/images/0a1f3af3af.jpg")
    image = pred.preprocess_image()
    model = torch.load("/mnt/d/kaggle/models/toy_clf_densenet161_v3.1.0.pth")
    top_prob, top_class = pred.predict(image, model)

    # Print the results
    if top_class == 0:
        print(f"The model is {top_prob*100} % certain that the image has a predicted class of: consumer_products")
    else:
        print(f"The model is {top_prob*100} % certain that the image has a predicted class of: toys")