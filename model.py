
import  Image_loader

import torch
from torchvision.transforms import transforms
import torch.nn as nn
from torch.optim import SGD
import  torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
print("building model...")

class Model(nn.Module):
    """ This class trains the model for image classification """

    def __init__(self ):
        """ initiize model parameter """
        super().__init__()
        # self.epoch = epoch
        # self.learning_rate = learning_rate
        # self.optim =  optim
        
    
        """ builds the convolution neural network """
        self.conv1 = nn.Conv2d(3, out_channels=32, kernel_size=3 )
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding_mode="zeros")
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(in_features = 32 * 5 * 5, out_features = 150)
        self.fc2 = nn.Linear(in_features = 150,out_features =  90)
        self.fc3 = nn.Linear(in_features = 90,out_features = 10)

    def forward(self, x):
        """ create feed forward network """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2 )
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        """ calculate number of parameters """
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

    

if __name__ == "__main__":
    cnn = Model()
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
     )
    img = Image_loader.ImageDataLoader(
                            Image_loader.data_dir, 
                            transform=transform
                    )
    # definr optimizer
    optimizer = Adam(cnn.parameters(), lr=0.07)
    # defining the loss function
    criterion = CrossEntropyLoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        cnn = cnn.cuda()
    criterion = criterion.cuda() 
    print(cnn)
    imdata = img.image_loader()
    for batch_idx, (image, label) in enumerate(imdata):
        print(batch_idx)