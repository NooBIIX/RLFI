
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import cupy as np
#import numpy as np
import random
# import tqdm

import math
import matplotlib
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torchvision
# import torch.utils.data.dataloader 
import torchvision.datasets 
import torchvision.transforms as transforms
import json

import cv2

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.dpi"] = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def get_test_loader(data_dir,
#                     batch_size,
#                     shuffle=True):
#     normalize = transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     )

#     # normalize = transforms.Normalize(
#     #     mean=[0.5, 0.5, 0.5],
#     #     std=[0.5, 0.5, 0.5],
#     # )

#     # define transform
#     transform = transforms.Compose([
#         transforms.Resize((227,227)),
#         transforms.ToTensor(),
#         normalize
#     ])

#     dataset = datasets.CIFAR10(
#         root=data_dir, train=False,
#         download=True, transform=transform,
#     )
    
#     # # Define the classes you want to include in the subset
#     # classes_to_include = [0] 

#     # # Create a subset containing only the examples from the specified classes
#     # subset_indices = [i for i in range(len(dataset)) if dataset.targets[i] in classes_to_include]
#     # subset = Subset(dataset, subset_indices)

#     # subsubset_indices = range(512)
#     # test_subset = Subset(subset, subsubset_indices)
#     subset_indices = range(1)
#     # print(type(subset_indices))
#     test_subset = Subset(dataset, subset_indices)

#     data_loader = torch.utils.data.DataLoader(
#         test_subset, batch_size=batch_size, shuffle=shuffle
#     )

#     return data_loader

# test_loader = get_test_loader(data_dir = './datasets', batch_size = 1)

# def hook_fn(module, input, output):
#     print(type(output))
#     activation_list[0] = output
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

transform = transforms.Compose(
    [transforms.Resize((227,227)),
     transforms.ToTensor(),
     transforms.Normalize(mean, std)])


dataset = torchvision.datasets.CIFAR10(
        root='./datasets', train=False,
        download=True, transform=transform,
    )
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()          
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = dataloader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def zero_out_specific_indices_hook(module, input):
    # Assuming input[0] is the tensor going into the desired layer
    #print(input[0])
    # print(f"Input type: {type(input)}")
    # print(len(input))
    activation_list.append(input[0].clone())
    layer_input = input[0]  # Get the input tensor to the layer
    #print(layer_input)
    # print(f"Input type: {type(layer_input)}")
    #mask = torch.ones_like(layer_input)
    mask = torch.zeros_like(layer_input)
    if torch.Tensor(layer_input).dim() == 4:
        for _ in range(11500):
            chanel_shape = torch.Tensor(layer_input).shape[1]
            x_shape = torch.Tensor(layer_input).shape[2]
            y_shape = torch.Tensor(layer_input).shape[3]
            #mask_3D = np.zeros(mask_shape)
            chanel_idx = random.randint(0, chanel_shape-1)
            x_idx = random.randint(0, x_shape-1)
            y_idx = random.randint(0, y_shape-1)
            #print([chanel_idx,x_idx,y_idx])
            # layer_input[:, chanel_idx-1:chanel_idx+2, x_idx-1:x_idx+2, y_idx-1:y_idx+2] = 0
            # Modify specific indices to zero (example: set indices 2, 5, 8 to zero)
            indices_to_zero = [chanel_idx, x_idx, y_idx]
            
            # mask[:, chanel_idx, x_idx, y_idx] = 0
            layer_input[0, chanel_idx, x_idx, y_idx]=0
            # inter.append(layer_input)
            ##### mask shape (x,y)
            #print(indices_to_zero)
            #layer_input[:, chanel_idx,x_idx,y_idx] = 0
            
    elif torch.Tensor(layer_input).dim() == 2:
        #print(layer_input.shape[1])
        # for k in range(800):
        #     shape = torch.Tensor(layer_input).shape[1]
        #     idx = random.randint(0, shape-1)
            
        #     layer_input[:, idx] = 1
        #print(len(layer_input[1]))
        idx_list = random.sample(range(len(layer_input[1])), 700)
        
        layer_input[:, idx_list] = 1
        
        
    else:
        print('not supported')
    # layer_input *= mask
    # num_injections = torch.count_nonzero(layer_input == 1)/32
    # print(num_injections)
    inter.append(layer_input.clone().detach())


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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.reshape(x.size(0), -1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x


model = AlexNet().to(device)
state_dict = torch.load('alex_dict_cifar10.pt')
model.load_state_dict(state_dict)

model.eval()

layer_index = 2

# target_layer_ini = model._modules[f'layer{layer_index-1}']

# handle_ini = target_layer_ini.register_forward_hook(hook_fn)

# Register the forward pre-hook to the desired layer
target_layer = model._modules[f'layer{layer_index}']  # Access the layer by index

# activation = torch.tensor(0)
activation_list = []
inter = []

handle1 = target_layer.register_forward_pre_hook(zero_out_specific_indices_hook)


img_ori = 0


with torch.no_grad():
    # dataiter = iter(trainloader)
    # images, labels = next(dataiter)
    # torch.save(images, 'image.pt')
    # for index, (images, labels) in enumerate(dataiter):
    images = torch.load('image_ship.pt')
    images = images.to(device)
    # labels = labels.to(device)
    outputs = model(images)
    img_ori = images.clone()
    # print(index)
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(1)))



activation = activation_list[0][0]

# activation_mean = torch.mean(activation,dim=0)
activation_max,_ = torch.max(activation,dim=0)

indices = np.load('indices_0_layer1.npy')

indices = torch.tensor(indices[1:3])
act_masked = activation
act_masked[indices] = 0
# act_masked_max,_ = torch.max(act_masked,dim=0)
act_masked =torch.mean(activation,dim=0)

# print(inter[0].shape)
# random_inject,_ = torch.max(inter[0][0], dim=0)
random_inject = torch.mean(inter[0][0], dim=0)
# print("tensor1 and tensor2 are equal:", torch.equal(act_masked, random_inject))

unnormalize = transforms.Normalize(
    mean=[-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]],
    std=[1 / std[0], 1 / std[1], 1 / std[2]]
)
image_original = unnormalize(img_ori[0])

fig, axs = plt.subplots(1, 4, figsize=(12, 4))
# fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, hspace=0., wspace=0.)
# fig.tight_layout(pad=0.0)


# fig, axs = plt.subplots(4, 1, figsize=(4, 12))
# fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.2)

plt.subplot(1, 4, 1)
# plt.subplot(4, 1, 1)
plt.imshow(image_original.permute(1,2,0).cpu().numpy())
plt.title('Image Original')
plt.axis('off')

plt.subplot(1, 4, 2)
# plt.subplot(4, 1, 2)
plt.imshow(activation_max.cpu().numpy())
plt.title('Activation')
plt.axis('off')

plt.subplot(1, 4, 3)
# plt.subplot(4, 1, 3)
# plt.imshow(act_masked_max.cpu().numpy())
plt.imshow(act_masked.cpu().numpy())
plt.title('RL Injected')
plt.axis('off')

plt.subplot(1, 4, 4)
# plt.subplot(4, 1, 4)
plt.imshow(random_inject.cpu().numpy())
# plt.imshow(act_diff.cpu().numpy(), cmap='gray')
plt.title('Randomly Injected')
plt.axis('off')

# plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
plt.show()
fig.savefig("tst.png",bbox_inches='tight')


# from mpl_toolkits.axes_grid1 import ImageGrid
# activation = activation_list[0].cpu()
# how_many_features_map = activation.shape[1]

# figure_size = how_many_features_map * 2
# fig = plt.figure(figsize=(figure_size, figure_size),)

# grid = ImageGrid(fig, 111,
#                     nrows_ncols=(how_many_features_map // 16, 16),
#                     axes_pad=0.1,  # pad between axes in inch.
#                     )
# images = [activation[0, i, :, :] for i in range(how_many_features_map)]

# for ax, img in zip(grid, images):
#     # Iterating over the grid returns the Axes.
#     ax.matshow(img)
# plt.show()

# plt.savefig('vis.png')




# plt.figure()
# plt.imshow(activation[0][0].permute(1,2,0).cpu().numpy())  # Assuming grayscale images
# plt.title(f'Layer {layer_index-1} Output')
# plt.colorbar()
# plt.axis('off')
# plt.show()
