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
from model.baseline import BaseNet
from model.squeezenet import SqueezeNet
from model.densenet import DenseNet
import torchvision.models as models
from utils.transforms import train_transform, test_transform
from progress.bar import Bar as Bar
from utils import Logger, train, test, mkdir_p, savefig, save_checkpoint
from dataloader import get_dataloader

parser = argparse.ArgumentParser(description="MicroNet Challenge")


parser.add_argument(
    "--gpu-ids", nargs="+", type=int, default=None, help="List of id of GPUs to use."
)
parser.add_argument(
    "--cpu-workers", type=int, default=4, help="Number of CPU workers for reading data."
)
parser.add_argument(
    "--data-dir",
    default="./data",
    help="Directory to load dataset from(or download to in" " case not present)",
)
parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

parser.add_argument_group("Optimization related arguments")
parser.add_argument(
    "--lr", type=float, default=0.1, help="learning rate (default:1e-1)"
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=[150, 225],
    help="Decrease Learning rate at these epochs (default: 150 & 225)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.1,
    help="Learning rate is multipled by gamma on schedule (default: 0.1)",
)
parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
    metavar="M",
    help="Momentum for optimizer (default: 0.9)",
)

parser.add_argument(
    "--weight-decay",
    "--wd",
    default=5e-4,
    type=float,
    metavar="W",
    help="Weight decay for regularisation (dafult: 5e-4)",
)
parser.add_argument(
    "--epochs", type=int, default=300, help="number of epochs to train (default: 300)"
)
parser.add_argument(
    "--train-batch-size",
    type=int,
    default=100,
    help="Input batch size while training (default: 100)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=64,
    help="Input batch size while testing (default: 64)",
)
parser.add_argument(
    "--overfit",
    action="store_true",
    default=False,
    help="Overfit model on small batch size(128 examples), " "meant for debugging",
)
parser.add_argument(
    "--log-interval", type=int, default=1000, help="Frequency of logging progress."
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful for restart. default: 0)",
)
parser.add_argument(
    "--save-freq", type=int, default=100, help="Interval to save model checkpoints"
)
parser.add_argument(
    "--save-path", default="./checkpoint", help="Path of directory to save checkpoints."
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="Path to latest checkpoint (default: None)",
)
parser.add_argument(
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

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
trainLoader, testLoader = get_dataloader(
    batch_size=args.train_batch_size, data_dir=args.data_dir
)

print("Batch Size : ", args.train_batch_size)
print("Test Batch Size : ", args.test_batch_size)
print("Number of batches in training set : ", trainLoader.__len__())
print("Number of batches in testing set : ", testLoader.__len__())

#  -----------------------------------------------------------------------
# Setup Model, Loss function & Optimizer
#  -----------------------------------------------------------------------
# model = DenseNet(depth=100, growthRate=12, dropRate=0.25).to(device)
model = BaseNet().to(device)
print(
    "\tTotal params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0)
)
print("Device : ", device)
if "cuda" in str(device):
    model = torch.nn.DataParallel(model, args.gpu_ids)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)
criterion = nn.CrossEntropyLoss()


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state["lr"] *= args.gamma
        for param_group in optimizer.params_groups:
            param_group["lr"] = state["lr"]


best_acc = 0

start_epoch = args.start_epoch

if not os.path.isdir(args.save_path):
    mkdir_p(args.save_path)

title = "CIFAR-100"
if args.resume:
    print("==> Resuming from checkpoint..")
    assert os.path.isfile(args.resume), "Error: no checkpoint directory found !!!"
    args.save_path = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint["best_acc"]
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    logger = Logger(os.path.join(args.save_path, "log.txt"), title=title, resume=True)
else:
    logger = Logger(os.path.join(args.save_path, "log.txt"), title=title)
    logger.set_names(
        ["Learning Rate", "Train Loss", "Valid Loss", "Train Acc.", "Valid Acc."]
    )


if args.evaluate:
    print("\nEvaluation only")
    test_loss, test_acc = test(testLoader, model, criterion, start_epoch, device)
    print(" Test Loss: %0.8f, Test Acc: %.3f" % (test_loss, test_acc))
    exit()

# Train and Val
for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)

    print("\nEpoch: [%d | %d] LR: %f" % (epoch + 1, args.epochs, state["lr"]))

    train_loss, train_acc = train(
        trainLoader, model, criterion, optimizer, epoch, device
    )
    test_loss, test_acc = test(testLoader, model, criterion, epoch, device)

    # append logger file
    logger.append([state["lr"], train_loss, test_loss, train_acc, test_acc])

    # save model
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint(
        {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "acc": test_acc,
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        },
        is_best,
        checkpoint=args.save_path,
    )

logger.close()
logger.plot()
savefig(os.path.join(args.save_path, "log.eps"))
