import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Load MNIST train and test datasets

train_dataset = datasets.MNIST(root="datasets/", download=True, train=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=60, shuffle=True)

test_dataset = datasets.MNIST(root="datasets/", download=True, train=False, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_dataset, batch_size=60, shuffle=True)

# Show a random batch of 60 digits

# Get random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

img = torchvision.utils.make_grid(images)
npimg = img.numpy()
plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()