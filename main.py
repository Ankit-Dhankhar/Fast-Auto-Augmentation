"""
@author: Ankit Dhankhar
@contact: adhankhar@cs.iitr.ac.in
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.baseline import BaseNet
from model.squeezenet import SqueezeNet

parser = argparse.ArgumentParser(description='MicroNet Challenge')


parser.add_argument("--gpu-ids", nargs="+", type=int, default=None,
                   help="List of id of GPUs to use.")
parser.add_argument("--cpu-workers", type=int, default=4,
                   help="Number of CPU workers for reading data.")
parser.add_argument("--data-dir", default="./data",
                   help="Directory to load dataset from(or download to in"
                   " case not present)")
parser.add_argument("--seed", type=int, default=0,
                   help="Random seed (default: 0)")

parser.add_argument_group("Optimization related arguments")
parser.add_argument("--lr", type=float, default=1e-3,
                   help="learning rate (default:1e-3)")
parser.add_argument("--epochs", type=int, default=100,
                   help="number of epochs to train (default: 50)")
parser.add_argument("--batch-size", type=int, default=16,
                   help="Input batch size (default: 16)")

parser.add_argument("--overfit", action="store_true", default=False,
                   help="Overfit model on small batch size(128 examples), "
                    "meant for debugging")
parser.add_argument("--log-interval", type=int, default=1000,
                    help="Frequency of logging progress.")
parser.add_argument("--save-freq", type=int, default=100,
                  help="Interval to save model checkpoints")
parser.add_argument("--save-path", default="./checkpoint",
                   help="Path of directory to save checkpoints.")


args = parser.parse_args()

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

# set CPU/GPU device for execution
if args.gpu_ids is not None:
    if args.gpu_ids[0] >= 0:
        device = torch.device("cuda", args.gpu_ids[0])
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#  -----------------------------------------------------------------------
# Importing Dataset & DataLoader
#  -----------------------------------------------------------------------
transform = transforms.ToTensor()
trainDataset = datasets.CIFAR100(root=args.data_dir, train=True, transform=transform, target_transform=None,
                                       download=True)
testDataset  = datasets.CIFAR100(root=args.data_dir, train=False, transform=transform, target_transform=None,
                                        download=True)

if args.overfit is True:
    trainDataset.data = trainDataset.data[:256]
    testDataset.data  = trainDataset.data[:256]

trainLoader = DataLoader(trainDataset, shuffle=False, num_workers=8, batch_size=args.batch_size)
testLoader = DataLoader(testDataset, shuffle=False, num_workers=8, batch_size=args.batch_size)

print("Batch Size : ", args.batch_size)
print("Number of batches in training set : ", trainLoader.__len__())
print("Number of batches in testing set : ", testLoader.__len__())

#  -----------------------------------------------------------------------
# Setup Model, Loss function & Optimizer
#  -----------------------------------------------------------------------
model = BaseNet().to(device)
print("Device : ", device)
# if "cuda" in str(device):
#     model = torch.nn.DataParallel(model, args.gpu_ids)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for epoch in range(0, args.epochs):
    model.train()
    print("Training")
    for batch_idx, (data, target) in enumerate(trainLoader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} {:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainLoader.dataset),
                100. * batch_idx / len(trainLoader), loss.item()))
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in trainLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(trainLoader.dataset)

    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(trainLoader.dataset),
        100. * correct / len(trainLoader.dataset)))
    print("Testing")
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(testLoader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(testLoader.dataset),
        100. * correct / len(testLoader.dataset)))