import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image

from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    #y, _, _ = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, 
                 input_transform=None, target_transform=None, 
                 num_transforms=1, scale_factor=2, crop_size=224):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.cropper = CenterCrop(crop_size)
        self.scalers = [Scale(crop_size, interpolation=3)]
        self.tensorer = ToTensor()
        
        scale_size = crop_size
        for index in range(num_transforms):
            
            scale_size = int(scale_size / scale_factor)
            self.scalers.append(Scale(scale_size, interpolation=3))
        #throw_error()
        
    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        input = self.cropper(input)
        
        target = input.copy()
        all_targets = []
        
        input = self.scalers[-1](input)
        input = self.tensorer(input)
                
        for scaler_index in range(len(self.scalers) - 1):
            all_targets.append(self.tensorer(self.scalers[scaler_index](target)))
            
        all_targets = all_targets[::-1]
        return input, all_targets

    def __len__(self):
        return len(self.image_filenames)
