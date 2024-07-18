import torchvision.datasets as dset
from torch.utils.data import Dataset, RandomSampler
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader
import glob
from PIL import Image
import os
from bert_vectorize import vectorize_text_with_bert



img_size = 64
transforms = v2.Compose([
    #v2.PILToTensor(),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(img_size),
    v2.CenterCrop(size=(img_size, img_size))
    

])



class CocoCaptionsDatasetWrapper(Dataset):
    def __init__(self, root, transform=None, split="train"):



        super(CocoCaptionsDatasetWrapper, self).__init__()


        self.internal_dataset = dset.CocoCaptions(os.path.join(root, f"{split}2017"), os.path.join(root, "annotations", f"captions_{split}2017.json"), transform=transform)

    def __len__(self):
        return len(self.internal_dataset)
    
    def __getitem__(self, idx):
        img, captions = self.internal_dataset.__getitem__(idx)

        caption = captions[torch.randint(len(captions), (1,)).item()]

        humanreadable_caption = caption

        caption = vectorize_text_with_bert(caption)


        return img, caption, humanreadable_caption
        



def get_train_dataset():
    dataset = CocoCaptionsDatasetWrapper(
        root=os.path.expanduser("~/torch_datasets/coco"),
        split="train",
        transform=transforms
    )
    return dataset

def get_test_dataset():
    dataset = CocoCaptionsDatasetWrapper(
        root=os.path.expanduser("~/torch_datasets/coco"),
        split="val",
        transform=transforms
    )
    return dataset

def get_random_test_data():
    testset = get_test_dataset()

    sampler = RandomSampler(testset)

    randomloader = DataLoader(testset, sampler=sampler, batch_size=1)

    return next(iter(randomloader))



def get_dataloader(dataset, batch_size=64):

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    cap = get_train_dataset()
    print('Number of samples: ', len(cap))
    img, target = cap[4] # load 4th sample

    print("Image Size: ", img.size())
    print(target)