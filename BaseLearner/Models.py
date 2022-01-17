import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(2, 2), padding=3),
                                   nn.BatchNorm2d(16),
                                   nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=2),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                   nn.Conv2d(32, 64, kernel_size=(3, 3)),
                                   nn.BatchNorm2d(64),
                                   nn.Conv2d(64, 128, kernel_size=(3, 3)),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                   nn.Flatten(),
                                   nn.Dropout(),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(128 * 4 * 4, 128),
                                   nn.Dropout(),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(128, 10),
                                   nn.Sigmoid())

    def forward(self, x):
        """
        :param x: input image (batch, 3, 96, 96)
        :return: class probabilitys (batch, num_classes)
        """
        return self.model(x)


class BaseModel(nn.Module):
    def __init__(self, baseLearners, num_classes=10):
        super().__init__()
        self.baseLearners = baseLearners
        self.num_classes = num_classes

    def forward(self, x):
        out = F.softmax(self.baseLearners[0](x), dim=1)
        for BL in self.baseLearners[1:]:
            out += F.softmax(BL(x), dim=1)
        return torch.div(out, len(self.baseLearners))
