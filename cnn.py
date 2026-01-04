import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # first conv layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # second conv layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # last fully connected layer
        self.fc = nn.Linear(16*7*7, num_classes)

    def forward(self, out):
        """
        Defines the forward pass of the neural network
        
        :param out: Input tensor

        Returns the output tensor after forward propagation
        """
        out = F.relu(self.conv1(out)) # apply first convolution operation and ReLU activation
        out = self.pool(out) # apply max pooling
        out = F.relu(self.conv2(out)) # apply second convolution operation and ReLU activation
        out = self.pool(out) # max pooling again
        out = out.reshape(out.shape[0], -1) # flatten tensor
        out = self.fc(out) # apply fully connected layer
        
        return out

# print model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN(in_channels=1, num_classes=10).to(device)
print(model)