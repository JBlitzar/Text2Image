import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader
import glob
from PIL import Image
from bert_vectorize import vectorize_text_with_bert

transforms = v2.Compose([
    v2.PILToTensor(),
    v2.RandomHorizontalFlip(p=0.5),
    v2.Resize((640,480)),
    v2.ToDtype(torch.float32, scale=True),
    #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class VectorizedCocoCaptionsDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.annFile = annFile
        self.transform = transform
        self._dataset = dset.CocoCaptions(root = root,
                            annFile = annFile,
                            transform=transform)

    def __len__(self):
        return self._dataset.__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, descriptions = self._dataset[idx]

        captions_vectorized = torch.Tensor([list(vectorize_text_with_bert(caption)) for caption in descriptions])
        new_item = torch.sum(captions_vectorized, 0)/len(descriptions)
    
        if self.transform and torch.max(image) > 1:
            image = self.transform(image)

        return image, new_item


def get_train_dataset():
    dataset = VectorizedCocoCaptionsDataset(root = 'data/train2017',
                            annFile = 'data/annotations/captions_train2017.json',
                            transform=transforms)
    return dataset

def get_test_dataset():
    dataset = VectorizedCocoCaptionsDataset(root = 'data/test2017',
                            annFile = 'data/annotations/captions_test2017.json',
                            transform=transforms)
    return dataset

def get_val_dataset():
    dataset = VectorizedCocoCaptionsDataset(root = 'data/val2017',
                            annFile = 'data/annotations/captions_val2017.json',
                            transform=transforms)
    return dataset

def get_dataloader(dataset, batch_size=64):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    cap = get_train_dataset()
    print('Number of samples: ', len(cap))
    img, target = cap[4] # load 4th sample

    print("Image Size: ", img.size())
    print(target)