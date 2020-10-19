import torch
import torchvision
import  torch.nn.functional as F
from torch.autograd import Variable

from create_dataset import ImageDataset
from config import Config

class DenseNet161:
    """ create CNN Model """
    def __init__( self, num_class, epochs):
        """ initiize model parameter """
        print("building model...")
        self.model = torchvision.models.densenet161(pretrained=True)
        self.num_class = num_class
        self.device = None
        self.epochs = epochs
        # Turn off training for their parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace default classifier with new classifier
        print("*"*100)
        print(self.model)
        print("*"*100)
        self.optimizer = None

    def set_device(self):
        """ selects device for training model """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        # Move model to the device specified above    
        return self.model
    
    def train(self, train_loader, val_loader):
        """ trains the model """
        print("*"*100) 
        print("training started...")
        model = self.model
        # Turn off training for their parameters
        for param in model.parameters():
            param.requires_grad = False
        classifier_input = model.classifier.in_features
        num_labels = self.num_class
        classifier = torch.nn.Sequential(
                                torch.nn.Linear(classifier_input, 1024),
                                torch.nn.ReLU(),
                                torch.nn.Linear(1024, 512),
                                torch.nn.ReLU(),
                                torch.nn.Linear(512, num_labels),
                                torch.nn.LogSoftmax(dim=1))
        # Replace default classifier with new classifier
        model.classifier = classifier
        criterion = torch.nn.CrossEntropyLoss()
        # Set the optimizer function using torch.optim as optim library
        optimizer = torch.optim.Adam(model.classifier.parameters())
        self.set_device()
        for epoch in range(self.epochs):
            train_loss = 0
            val_loss = 0
            accuracy = 0
            self.model.train()
            # Training the model
            counter = 0
            for inputs, labels in train_loader:
                # Move to device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Clear optimizers
                optimizer.zero_grad()
                # Forward pass
                output = self.model.forward(inputs)
                # Loss
                loss = criterion(output, labels)
                # Calculate gradients (backpropogation)
                loss.backward()
                # Adjust parameters based on gradients
                optimizer.step()
                # Add the loss to the training set's rnning loss
                train_loss += loss.item()*inputs.size(0)
                
                # Print the progress of our training
                counter += 1
                print(counter, "/", len(train_loader))
                
            # Evaluating the model
            model.eval()
            counter = 0
            # Tell torch not to calculate gradients
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Move to device
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # Forward pass
                    output = model.forward(inputs)
                    # Calculate Loss
                    valloss = criterion(output, labels)
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
                    print(counter, "/", len(val_loader))
            
            # Get the average loss for the entire epoch
            train_loss = train_loss/len(train_loader.dataset)
            valid_loss = val_loss/len(val_loader.dataset)
            # Print out the information
            print('Accuracy: ', accuracy/len(val_loader))
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))
            print("saving model...")
            model_path = str(Config.MODELS) + f"acc_{round(accuracy/len(val_loader), 3)}" + f"_v3.1.{epoch}.pth"
            torch.save(model, model_path)
            print("model saved at", model_path)
            print("*"*50)


if __name__ == "__main__":
    model = DenseNet161(num_class=2, epochs=2)
    tr_ds = ImageDataset(Config.TRAIN_DATASET)
    train_dl = tr_ds.create_data_loader()
    val_ds = ImageDataset(Config.VALIDATION_DATASET)
    validation_dl = val_ds.create_data_loader()
    model.train(
        train_loader= train_dl,
        val_loader=validation_dl
    )

    