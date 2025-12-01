from torch.utils.data import Dataset
from PIL import Image
import os


class CelebADataset(Dataset):
    def __init__(self, root_dir, image_list, transform=None):
        self.root_dir = root_dir
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return (image, 0)