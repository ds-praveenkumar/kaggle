#!/usr/bin/env python

import torch
from torchvision.datasets import ImageFolder

from PIL import Image
from pathlib import Path
import numpy as np

print("torch version:", torch.__version__)
data_dir  = Path(Path.cwd()).resolve().parents[1] / "datasets"
print("dataset path:", data_dir)

class ImageDataLoader:
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
            img = image.open(str(image))
            self.images.append(img)
            img.close()

        self.len = len(self.images)
        
        if prefetch(self):
            pass

        def __len__(self):
            return self.len

        def transform(self):
            pass

        def __getitem__(self):
            """ returns images from the folder """
            
     