import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = 'data/train2017',
                        annFile = 'data/annotations/captions_train2017.json',
                        transform=transforms.PILToTensor())
print('Number of samples: ', len(cap))
img, target = cap[3] # load 4th sample

print("Image Size: ", img.size())
print(target)