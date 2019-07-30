"""
@author: Ankit Dhankhar
@contact: adhankhar@cs.iitr.ac.in
"""
import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    """
    Simple Baseline model for image classification.
    Input:
        Image
    Output:
        Classification probability (N classes)
    """
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    model = BaseNet()
    print(model)
