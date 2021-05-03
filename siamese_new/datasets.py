import torch
import numpy as np
from torch.utils.data import Dataset

class SiameseImageFolder(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.transform = self.image_folder.dataset.transform
        self.labels = torch.tensor([i[1] for i in self.image_folder])
        self.labels_set = set(self.labels.numpy())
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}


    def __getitem__(self, index):
        target = np.random.randint(0, 2)
        img1, label1 = self.image_folder[index][0], self.labels[index].item()
        if target == 1:
            siamese_index = index
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - set([label1])))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        img2 = self.image_folder[siamese_index][0]
        return (img1, img2), target

    def __len__(self):
        return len(self.image_folder)
