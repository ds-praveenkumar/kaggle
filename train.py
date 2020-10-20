from  model import Model

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

print("training model...")
class Train:
    """ Train the cnn model """

    def __init__(self, model, epoch, optimizer, criterion):
        """ initilize training parameters """
        # defining the model
        self.model = model
        # defining the optimizer
        self.optimizer = optimizer
        # defining the loss function
        self.criterion = criterion
        # no of iterations 
        self.epoch = epoch
        # checking if GPU is available
        if torch.cuda.is_available():
            model = model.cuda()
        criterion = criterion.cuda()
        print("model initialized")


    def train(self, train_x, train_y, val_x, val_y):
        """ Trains the CNN Model """
        self.model.train()
        tr_loss = 0
        # getting the training set
        x_train, y_train = Variable(train_x), Variable(train_y)
        # getting the validation set
        x_val, y_val = Variable(val_x), Variable(val_y)
        # converting the data into GPU format
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        # clearing the Gradients of the model parameters
        self.optimizer.zero_grad()
        
        # prediction for training and validation set
        output_train = self.model(x_train)
        output_val = self.model(x_val)

       
        # empty list to store training losses
        train_losses = []
        # empty list to store validation losses
        val_losses = []
       
        # computing the training and validation loss
        loss_train = self.criterion(output_train, y_train)
        loss_val = self.criterion(output_val, y_val)
        train_losses.append(loss_train)
        val_losses.append(loss_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        self.optimizer.step()
        tr_loss = loss_train.item()
        if self.epoch % 2 == 0:
            # printing the validation loss
            print('Epoch : ', self.epoch+1, '\t', 'loss :', loss_val)

if __name__ == "__main__":
    tr = Train(
        model=Model(),
        epoch=25,
        optimizer=SGD(
                    params= Model().parameters(),
                    lr = 0.01,
                    momentum=0.9,
                    weight_decay=0.0005
                 ),
        criterion=CrossEntropyLoss()
    )