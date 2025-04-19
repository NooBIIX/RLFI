import torch
import torch.nn as nn
from torch.utils.data import Subset
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import csv

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
        download=True, transform=transform,
    )
    # # Define the classes you want to include in the subset
    # classes_to_include = [0] 

    # # Create a subset containing only the examples from the specified classes
    # subset_indices = [i for i in range(len(dataset)) if dataset.targets[i] in classes_to_include]
    # subset = Subset(dataset, subset_indices)

    # subsubset_indices = range(512)
    # test_subset = Subset(subset, subsubset_indices)
    subset_indices = range(512)
    test_subset = Subset(dataset, subset_indices)

    data_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader
random_number = []


test_loader = get_test_loader(data_dir = './datasets', batch_size = 32)
def get_hook(episode):
    def zero_out_specific_indices_hook(module, input):
        # Assuming input[0] is the tensor going into the desired layer
        #print(input[0])
        # print(f"Input type: {type(input)}")
        # print(len(input))
        layer_input = input[0]  # Get the input tensor to the layer
        #print(layer_input)
        # print(f"Input type: {type(layer_input)}")
        #mask = torch.ones_like(layer_input)
        mask = torch.zeros_like(layer_input)
        if torch.Tensor(layer_input).dim() == 4:
            for _ in range(15000):
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
                layer_input[:, chanel_idx, x_idx, y_idx]=1
                
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
            idx_list = random.sample(range(len(layer_input[1])), 1300)
            
            layer_input[:, idx_list] = 1
                # mask[:, idx] = 1
            # #for k in range(50):
            #     # print(layer_input)
            #     # print(type(layer_input))
            # input_shape = torch.Tensor(layer_input).shape[1]
            # idx_start = random.randint(0, input_shape-1-64)
            # #idx_end = random.randint(idx_start, input_shape)
            # #layer_input[:,idx_start:idx_end] = 0
            # layer_input[:,idx_start:idx_start+128] = 0
                # neuron_list.append(idx)
                # neuron_idx = [idx]
                # layer_input[:, neuron_idx] = 0

            
            #print(layer_input[:, neuron_idx].shape)
            # torch.set_printoptions(profile='full')
            # print(layer_input)    
            # print(layer_input.shape)
        else:
            print('not supported')
        # layer_input *= mask
        num_injections = torch.count_nonzero(layer_input == 1)/32
        # print(num_injections)
        number_list[episode] = num_injections.cpu().numpy()
    return zero_out_specific_indices_hook

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
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        return out



device = 'cuda'
model = VGG16().to(device)
state_dict = torch.load('VGG16_dict_cifar10.pth')
model.load_state_dict(state_dict)
neuron_list = []

model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        del images, labels, outputs
    accuracy_original = correct / total
    print('Accuracy of the network on the {} test images: {} %'.format(512, 100 * accuracy_original))   
num_episodes = 200
accuracy_list = []
number_list = [0 for _ in range(num_episodes)]
for i in range(num_episodes):
    print(i)
    layer_index = 15

    # Register the forward pre-hook to the desired layer
    target_layer = model._modules[f'layer{layer_index}']  # Access the layer by index

    #target_layer = model.fc
    handle = target_layer.register_forward_pre_hook(get_hook(i))
    with torch.no_grad():
      correct = 0
      total = 0
        # print(self.state['layer1'].shape)
      for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs
        
      accuracy = correct / total
      print('Accuracy of the modified network on the {} test images: {} %'.format(512, 100 * accuracy))
    handle.remove()
    #print(neuron_list)
    #print(set(neuron_list))
    accuracy_list.append(accuracy)
    

random_file = "random_output.csv"
with open(random_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Final Accuracy', 'Injection Number'])  # Write header
    idx_list = list(range(1, num_episodes+1))
    # for idx, reward_t in enumerate(total_reward, start=1):
    #     writer.writerow([idx, reward_t])
    rows = zip(idx_list, accuracy_list, number_list)
    writer.writerows(rows)
print("Data written to", random_file)
