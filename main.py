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
from utils import train, test

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
                   help="learning rate (default: 0)")
parser.add_argument("--epochs", type=int, default=100,
                   help="number of epochs to train (default: 50)")
parser.add_argument("--batch-size", type=int, default=128,
                   help="Input batch size (default: 128)")

parser.add_argument("--overfit", action="store_true", default=False,
                   help="Overfit model on small batch size(128 examples), "
                    "meant for debugging")
parser.add_argument("--log-interval", type=int, default=100,
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
    testDataset.data  = testDataset.data[:256]

trainLoader = DataLoader(trainDataset, shuffle=True, num_workers=8, batch_size=args.batch_size)
testLoader = DataLoader(testDataset, shuffle=True, num_workers=8, batch_size=args.batch_size)

print("Batch Size : ", args.batch_size)
print("Number of batches in training set : ", trainLoader.__len__())
print("Number of batches in testing set : ", testLoader.__len__())

#  -----------------------------------------------------------------------
# Setup Model, Loss function & Optimizer
#  -----------------------------------------------------------------------
model = BaseNet().to(device)
print("Device : ", device)
if "cuda" in str(device):
    model = torch.nn.DataParallel(model, args.gpu_ids)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


for epoch in range(0, args.epochs):
    train(args, model, device, trainLoader, optimizer, epoch)
    test(args, model, device, testLoader)
    if epoch % args.save_freq == 0:
        torch.save(model.state_dict(), os.path.join(args.save_path,"basenet_{}.pth".format(epoch)))