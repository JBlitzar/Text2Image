from datasets import load_dataset
import torch
from torchvision.transforms import v2
import os
import glob
from PIL import Image
import tqdm
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import ConcatDataset, random_split


#from t5_vectorize import vectorize_text_with_t5 as vectorize
from bert_vectorize import vectorize_text_with_bert as vectorize
#from clip_vectorize import vectorize_text_with_clip as vectorize



# flowers = load_dataset("pranked03/flowers-blip-captions")
# birds = load_dataset("anjunhu/naively_captioned_CUB2002011_test_20shot")
#flowers.set_format(type="torch", columns=["text", "image"])
#birds.set_format(type="torch", columns=["text", "image"])


img_size = 64
transforms = v2.Compose([
    #v2.PILToTensor(),
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(img_size),
    v2.CenterCrop(size=(img_size, img_size))
    

])

class BirdsDataset(torch.utils.data.Dataset):
    def __init__(self, hf_path: str, transform = transforms):
        super().__init__()

        self._interalds = load_dataset(hf_path)["train"]

        self.transform = transform



    def __len__(self):
        return len(self._interalds)
    
    def __getitem__(self, idx: int):
        item = self._interalds[idx]

        text_lines = item["description"].split("\n")
        text = ""
        while not text.strip():
            text = text_lines[torch.randint(0, len(text_lines), (1,)).item()]
        #text = text[torch.randint(0, len(text), (1,)).item()]


        image = item["image"]
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        vtext = vectorize(text)

        return image, vtext, text
    

class FlowersDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transform=transforms):
        super().__init__()

        self.transform = transform

        self.imgs_dir = os.path.join(root, "flowers-102/jpg")
        self.ann_dir = os.path.join(root, "flowers-102/text_c10")
        self.ann_glob = glob.glob(os.path.join(self.ann_dir, "class_*/image_*.txt"))

        # Preload all annotations and images into memory
        self.data = []
        for ann_path in tqdm.tqdm(self.ann_glob, leave=False):
            with open(ann_path, "r") as file:
                annotation = file.read()

            img_id = os.path.splitext(os.path.basename(ann_path))[0]
            img_path = os.path.join(self.imgs_dir, f"{img_id}.jpg")
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            
            
            self.data.append((annotation, image))

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        annotation, image = self.data[idx]
        annotation = annotation.split("\n")
        annotation = annotation[torch.randint(0, len(annotation), (1,)).item()]

        if self.transform:
            image = self.transform(image)


        vannotation = vectorize(annotation)

        return image, vannotation, annotation


def get_train_test_datasets(train_split=0.8):
    # Load the datasets
    flowers = FlowersDataset(root=os.path.expanduser("~/torch_datasets/flowers-full"))
    birds = BirdsDataset("Multimodal-Fatima/CUB_train")
    
    # Combine the datasets
    combined_dataset = ConcatDataset([flowers, birds])
    
    # Calculate the split lengths
    train_len = int(len(combined_dataset) * train_split)
    test_len = len(combined_dataset) - train_len
    
    # Randomly split the combined dataset into train and test sets
    train_dataset, test_dataset = random_split(combined_dataset, [train_len, test_len])
    
    return train_dataset, test_dataset
trainset, testset = get_train_test_datasets()
def get_train_dataset():
    return trainset
def get_random_test_data(amount=1):

    sampler = RandomSampler(testset)

    randomloader = DataLoader(testset, sampler=sampler, batch_size=amount)

    return next(iter(randomloader))

def get_dataloader(dataset, batch_size=64):

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    flowers = FlowersDataset(root=os.path.expanduser("~/torch_datasets/flowers-full"))
    birds = BirdsDataset("Multimodal-Fatima/CUB_train")
    print(len(flowers))
    print(len(birds))
    flower = flowers[200]
    bird = birds[200]


    def examine(sample):
        i, v,t = sample
        print(v.shape)
        print(t)
        print(i.shape)
        
        print(torch.max(i))
        print(torch.min(i))
        print(torch.mean(i))

    examine(flower)
    examine(bird)


    print(get_random_test_data(amount=10)[-1])


