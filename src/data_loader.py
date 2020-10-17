
from torchvision.utils import make_grid
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, Dataset, DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

print("torch version:", torch.__version__)
data_dir  = Path(Path.cwd()).resolve() / "datasets" / "products"
print("dataset path:", data_dir)

class ImageDataLoader(Dataset):
    """ This class implements toys and products data loader """
    
    def __init__(self, root_dir, transforms=None, prefetch=False):
        """ this class initilizes the root path """
        self.root_dir = root_dir
        self.transforms = transforms
        self._prefetch = prefetch
        self.images = []
        self.labels = []
        self.len = None

        # read from Imagefolder
        image_path = list(root_dir.glob("*/*.jpg"))
        for image in image_path:
            self.images.append(str(image))
            self.labels.append(str(image).split("/")[-2])


        self.len = len(self.images)
        
    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

    def transform(self):
        return 

    def __getitem__(self, index):
        """ returns images from the folder """
        image = Image.open(self.images[index])
        label = self.labels[index]
        return torch.transform.ToTensor(ImageDataLoader.transform(image), label)
    

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
            
     
if __name__ == "__main__":
    img_dataset = ImageDataLoader(data_dir)

    # total images in set
    print("total images", img_dataset.len)
    train_len = int(0.7 * img_dataset.len)
    valid_len = img_dataset.len - train_len
    train, valid = random_split(img_dataset, lengths=[train_len, valid_len])
    print("train: ", len(train), "val:", len(valid))
    train_dl = DataLoader(
            dataset=train,
            batch_size=2,
            shuffle=True,
    )
    dataiter = iter(train_dl)
    images, masks = dataiter.next()

    plt.figure(figsize=(16,16))
    plt.subplot(211)
    imshow( make_grid(images))
    plt.subplot(212)
    imshow(make_grid(masks))