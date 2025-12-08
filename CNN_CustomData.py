import os
from PIL import Image
import matplotlib.pyplot as plt

import torch 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as t
import numpy as np
import cv2

from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, data_dir, data_type = 'Training', size=(227,227), is_augment = False, transform=None, target_transform = None):
        super().__init__()
        self.size = size
        self.is_augment = is_augment
        self.data_dir = data_dir
        self.data_type = data_type
        self.image_names, self.labels = self.__process_data()
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.image_names)
    
    def __preprocess_data(self, image, save = False, size = (227,227)):
        test_dir = r"Dataset\POC_Dataset\Testing"
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
            
        clean_image = image.copy()
        
        if save :
            plt.imshow(image)
            plt.savefig(os.path.join(test_dir,'original.jpg'))
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if save :
            plt.imshow(thresh)
            plt.savefig(os.path.join(test_dir,'binary.jpg'))
        
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        if save :
            plt.imshow(thresh)
            plt.savefig(os.path.join(test_dir,'morphological.jpg'))
            
        contour, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contour, key=cv2.contourArea)
        x , y ,w ,h = cv2.boundingRect(cnt)
        
        if save :
            cv2.drawContours(image, [cnt], -1, (0,255,0), 2)
            plt.imshow(image)
            plt.savefig(os.path.join(test_dir,'largest contour.jpg'))
            
        crop = clean_image[y:y+h , x:x+w]
        
        if save :
            plt.imshow(crop)
            plt.savefig(os.path.join(test_dir,'cropped.jpg'))
            
            
        resized = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
        
        if save :
            plt.imshow(resized)
            plt.savefig(os.path.join(test_dir,'resized.jpg'))
            
        return resized
    
    def __getitem__(self, index):

        image_name = self.image_names[index]
        label = self.labels[index]
        
        label_map = {0: "Chorionic_villi", 1: "Decidual_tissue", 2: "Hemorrhage", 3: "Trophoblastic_tissue"}
        
        image_path = os.path.join(self.data_dir, self.data_type, label_map[label], image_name)
        image = Image.open(image_path)
        
        resized_image = self.__preprocess_data(np.array(image), save=False, size=self.size)
        
        if self.transform :
            image = self.transform(resized_image)
            
        if self.target_transform :
            label = self.target_transform(label)
            
        if self.is_augment:
            saved_image_path = os.path.join(self.data_dir, self.data_type, label_map[label], image_name.split('.')[0]+'_augmented.jpg')
            while os.path.exists(saved_image_path):
                random_number = np.random.randint(0,100)
                saved_image_path = saved_image_path.split('.')[0] + f'{random_number}.jpg'
                
                image = t.RandomRotation(degrees=(0,90))(image)
                image = t.RandomHorizontalFlip(p=1)(image)
                image.save(saved_image_path)
                
        return image, torch.tensor(label).long()
    
    def __process_data(self):
        if isinstance(self.data_dir, tuple):
            self.data_dir = os.path.join(*self.data_dir)
            
        Trophoblastic_tissue_image = os.listdir(os.path.join(self.data_dir, self.data_type, "Trophoblastic_tissue"))
        Chorionic_villi_image = os.listdir(os.path.join(self.data_dir, self.data_type, "Chorionic_villi"))
        Decidual_tissue_image = os.listdir(os.path.join(self.data_dir, self.data_type, "Decidual_tissue"))
        Hemorrhage_image = os.listdir(os.path.join(self.data_dir, self.data_type, "Hemorrhage"))

        combined_image = Trophoblastic_tissue_image + Chorionic_villi_image + Decidual_tissue_image + Hemorrhage_image
        labels = [3]*len(Trophoblastic_tissue_image) + [0]*len(Chorionic_villi_image) + [1]*len(Decidual_tissue_image) + [2]*len(Hemorrhage_image)
        
        return combined_image, labels
    
def plot_grid_images(x, y, batch_size):
    rows = cols = int(batch_size**0.5)
    fig, axes = plt.subplots(rows, cols, figsize = (10,10))
    axes = axes.flatten()
    
    for i in range(batch_size):
        axes[i].imshow(x[i].permute(1, 2 ,0))
        axes[i].set_title(f"Label : {y[i]}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()
        
    
