import torch.nn as nn
import torch.nn.functional as F
from torchvision. models import mobilenetv2,mobilenetv3


class MNIST_NET(nn.Module):
    def __init__(self,norm,act,num_classes=10):
        super(MNIST_NET, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.act1 = act()
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.act2 = act()
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.act3 = act()
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.act2(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleCifarNet(nn.Module):
    def __init__(self,norm,act,num_classes=10):
        super(SimpleCifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = norm(64)
        self.act1 = act()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = norm(128)
        self.act2 = act()
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = norm(256)
        self.act3 = act()
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = norm(512)
        self.act4 = act()
        self.fc1 = nn.Linear(512 * 2 * 2, 128)
        self.act5 = act()
        self.fc2 = nn.Linear(128, 256)
        self.act6 = act()
        self.fc3 = nn.Linear(256, 512)
        self.act7 = act()
        self.fc4 = nn.Linear(512, 1024)
        self.act8 = act()
        self.fc5 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(self.act1(self.bn1(self.conv1(x))))
        x = self.pool(self.act2(self.bn2(self.conv2(x))))
        x = self.pool(self.act3(self.bn3(self.conv3(x))))
        x = self.pool(self.act4(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 2 * 2)
        x = self.act5(self.fc1(x))
        x = self.act6(self.fc2(x))
        x = self.act7(self.fc3(x))
        x = self.act8(self.fc4(x))
        x = self.fc5(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, norm,act,num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = norm(6)
        self.act1 = act()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = norm(16)
        self.act2 = act()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.act3 = act()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = act()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(self.act1(self.bn1(self.conv1(x))))
        x = self.pool(self.act2(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MobileNet(mobilenetv2.MobileNetV2):
    def __init__(self, norm, num_classes=10):
        super(MobileNet, self).__init__(num_classes=num_classes,norm_layer=norm)

    def forward(self, x):
        x = super(MobileNet, self).forward(x)
        return x
