import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root = 'data',
                        annFile = 'data/annotations.json',
                        transform=transforms.PILToTensor())