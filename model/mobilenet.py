import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = models.mobilenet_v2(pretrained=True, num_classes=1000)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2), nn.Linear(in_features=1280, out_features=100, bias=True)
    )
    model = model.to(device)
    summary(model, input_size=(3, 32, 32))
