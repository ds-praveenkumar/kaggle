from .create_dataset import ImageDataset
from .config import Config
from .model import DenseNet161

import torch
import torchvision

class Train:
    """ Trains the convnets """
    def  __init__(self, epochs, model, train_loader, val_loader, optimizer, criterion, device):
        """ initilize training parameters """
        self.epochs = epochs
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.val_loader = val_loader
        self.device = device

    def train(self):
        """ trains the model """
        print("*"*50) 
        for epoch in range(self.epochs):
            train_loss = 0
            val_loss = 0
            accuracy = 0
            
            # Training the model
            counter = 0
            for inputs, labels in self.train_loader:
                # Move to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Clear optimizers
                self.optimizer.zero_grad()
                # Forward pass
                output = self.model.forward(inputs)
                # Loss
                loss = self.criterion(output, labels)
                # Calculate gradients (backpropogation)
                loss.backward()
                # Adjust parameters based on gradients
                self.optimizer.step()
                # Add the loss to the training set's rnning loss
                train_loss += loss.item()*inputs.size(0)
                
                # Print the progress of our training
                counter += 1
                print(counter, "/", len(self.train_loader))
                
            # Evaluating the model
            self.model.eval()
            counter = 0
            # Tell torch not to calculate gradients
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    # Move to device
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # Forward pass
                    output = self.model.forward(inputs)
                    # Calculate Loss
                    valloss = self.criterion(output, labels)
                    # Add loss to the validation set's running loss
                    val_loss += valloss.item()*inputs.size(0)
                    
                    # Since our model outputs a LogSoftmax, find the real 
                    # percentages by reversing the log function
                    output = torch.exp(output)
                    # Get the top class of the output
                    top_p, top_class = output.topk(1, dim=1)
                    # See how many of the classes were correct?
                    equals = top_class == labels.view(*top_class.shape)
                    # Calculate the mean (get the accuracy for this batch)
                    # and add it to the running accuracy for this epoch
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    # Print the progress of our evaluation
                    counter += 1
                    print(counter, "/", len(self.val_loader))
            
            # Get the average loss for the entire epoch
            train_loss = train_loss/len(self.train_loader.dataset)
            valid_loss = val_loss/len(self.val_loader.dataset)
            # Print out the information
            print('Accuracy: ', accuracy/len(self.val_loader))
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
            print("*"*50)

if __name__ == "__main__":
    model = DenseNet161(num_class=2, epochs=2)
    tr_ds = ImageDataset(Config.TRAIN_DATASET)
    train_dl = tr_ds.create_data_loader()
    val_ds = ImageDataset(Config.VALIDATION_DATASET)
    validation_dl = val_ds.create_data_loader()
    tr = Train(
            model = model,
            epochs = 2,
            train_loader  = train_dl,
            val_loader = validation_dl,
            optimizer = model.optimizer,
            criterion = torch.nn.CrossEntropyLoss(),
            device = model.device
    )
    tr.train()