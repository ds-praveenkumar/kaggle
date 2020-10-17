from config import Config

import torch
import torchvision

class ImageDataset():
    """ Implements the custom data loader """

    def __init__(self, root_dir):
        """ initilize root folder """
        self.root_dir = root_dir
        print("data path:", root_dir)
        self.transformations = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize(255),
                                    torchvision.transforms.CenterCrop(224),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    def create_data_set(self):
        """ data set """
        dataset = torchvision.datasets.ImageFolder(
                                            root=self.root_dir,
                                            transform=self.transformations
                                    )
        return dataset
    
    def create_data_loader(self):
        """ Creates data loader for iterating over training samples """
        dl = torch.utils.data.DataLoader( 
                                dataset=self.create_data_set(),
                                batch_size=32,
                                shuffle=True,
                                num_workers=3
        )
        return dl


if __name__ == "__main__":
    tr_ds = ImageDataset(Config.TRAIN_DATASET)
    train_dl = tr_ds.create_data_loader()
    val_ds = ImageDataset(Config.VALIDATION_DATASET)
    validation_dl = val_ds.create_data_loader()
    print("train data batches:", len(train_dl))
    print("validation data batches:", len(validation_dl))



































