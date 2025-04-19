import torch
import torch.nn as nn
from torch.utils.data import Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=False, transform=transform,
    )
    subset_indices = range(512)
    test_subset = Subset(dataset, subset_indices)

    data_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader

test_loader = get_test_loader(data_dir = './datasets', batch_size = 32)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer6 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.layer8= nn.Sequential(
            nn.Linear(4096, 10))

    def forward(self, x):
        x = self.layer1(x)
        # print(x[0,48,:,:])
        # print(x[0,48,:,:].shape)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

class VGG16(nn.Module):
    def __init__(self, num_class = 10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer14 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())

        self.layer15 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())

        self.layer16 = nn.Sequential(
            nn.Linear(4096, num_class)
        )


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size(0), -1)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        return x


device = torch.device('cuda')
model = AlexNet().to(device)
#model = VGG16().to(device)
state_dict = torch.load('alex_dict_cifar10.pt')
#state_dict = torch.load('VGG16_dict_cifar10.pth')
model.load_state_dict(state_dict)
model.eval()

def zero_out_specific_indices_hook(module, input):
    # Assuming input[0] is the tensor going into the desired layer
    #print(input[0])
    # print(f"Input type: {type(input)}")
    # print(len(input))
    layer_input = input[0]  # Get the input tensor to the layer
    print(layer_input.shape)
    # print(f"Input type: {type(layer_input)}")
    if torch.Tensor(layer_input).dim() == 4:

        chanel_shape = torch.Tensor(layer_input).shape[1]
        x_shape = torch.Tensor(layer_input).shape[2]
        y_shape = torch.Tensor(layer_input).shape[3]
        #print(layer_input.shape)
        mask_shape = (3, 3, 3)
        #mask_3D = np.zeros(mask_shape)
        chanel_idx = random.randint(0, chanel_shape-1)
        x_idx = random.randint(0, x_shape-1)
        y_idx = random.randint(0, y_shape-1)
        layer_input[:,20:48,:,:] = 0
        # Modify specific indices to zero (example: set indices 2, 5, 8 to zero)
        # indices_to_zero = [chanel_idx, x_idx, y_idx]
        ###### mask shape (x,y)
        # layer_input[:, indices_to_zero] = 0
        #layer_input[:, chanel_idx:chanel_idx+2, x_idx:x_idx+2, y_idx:y_idx+2] = 0
    elif torch.Tensor(layer_input).dim() == 2:
        
        input_shape = torch.Tensor(layer_input).shape[1]
        idx = random.randint(0, input_shape-1)
        layer_input[:, idx] = 0
    else:
        print('not supported')
    #print(layer_input[0,48,:,:])

model.layer2.register_forward_pre_hook(zero_out_specific_indices_hook)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        # print(labels)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted)
        # print(labels.size(0))
        total += labels.size(0)
        # print(total)
        correct += (predicted == labels).sum().item()
        # print(correct)

        del images, labels, outputs
    acc_original = correct / total
    print('Accuracy of the network on the {} test images: {} %'.format(512, 100 * acc_original))