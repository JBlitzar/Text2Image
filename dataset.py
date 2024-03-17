import torchvision.datasets as dset
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader

transforms = v2.Compose([
    v2.PILToTensor(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def get_train_dataset():
    dataset = dset.CocoCaptions(root = 'data/train2017',
                            annFile = 'data/annotations/captions_train2017.json',
                            transform=transforms)
    return dataset

def get_test_dataset():
    dataset = dset.CocoCaptions(root = 'data/test2017',
                            annFile = 'data/annotations/captions_test2017.json',
                            transform=transforms)
    return dataset

def get_val_dataset():
    dataset = dset.CocoCaptions(root = 'data/val2017',
                            annFile = 'data/annotations/captions_val2017.json',
                            transform=transforms)
    return dataset

def get_dataloader(dataset, batch_size=64):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    cap = get_train_dataset()
    print('Number of samples: ', len(cap))
    img, target = cap[3] # load 4th sample

    print("Image Size: ", img.size())
    print(target)