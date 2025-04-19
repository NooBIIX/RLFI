
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
import torch.utils.data.dataloader as dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import json
# from load_test import intermediate_outputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()
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

test_loader = get_test_loader(data_dir = './datasets', batch_size = 32)


def get_hook(mask):
    def zero_out_specific_indices_hook(module, input):
        layer_input = input[0]
        #print(mask)
        #input_original = layer_input
        layer_input *= mask        
        
        intermediate_output[0] = intermediate_output[1]
        intermediate_output[1] = layer_input.detach()
    return zero_out_specific_indices_hook

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


class NeuronEnv(gym.Env):

    def __init__(self):
        super(NeuronEnv, self).__init__()
        # Define your environment parameters
        
        self.max_steps = 20 - 1
        self.current_step = 0  # Current step in the episode
        
        self.state = intermediate_output[-1]  # Initial state (replace with your state initialization)

        self.observation_space = spaces.Box(low=0, high=100, shape=tuple(intermediate_output[-1].shape), dtype=np.float16)
               
        self.action_space = spaces.Box(low=-1, high=1, shape=tuple(intermediate_output[-1].shape), dtype=np.float16)
         

    def step(self, action):
        # Execute the given action and return the next state, reward, done flag, and additional info

        # action = torch.from_numpy(action)
 
          
        #zero = torch.zeros_like(action)
        idx_mask = torch.ones_like(action)
        flatten_action = action.view(-1)
        _, top_indices = torch.topk(flatten_action, k=8000)
           
        #idx_mask = torch.where(idx_mask > 0.5, zero, one).to(device)
        
        top_indices_4d = np.unravel_index(np.asnumpy(top_indices), action.shape)
        #print(top_indices_4d[2])
        # total_elements = sum(np.prod(arr.shape) for arr in top_indices_4d)
        # print(total_elements)
        # indices_list = list(top_indices_4d)
        # indices_list[0] = np.full(50, )
        # top_indices_4d = tuple(indices_list)
        
        #idx_mask[top_indices_4d] = 0
        idx_mask[:, top_indices_4d[1], top_indices_4d[2], top_indices_4d[3]] = 0
        #print(idx_mask[:, top_indices_4d[1:3]])
        #print(torch.count_nonzero(idx_mask == 0) /32)
        #print(idx_mask)
        handle = target_layer.register_forward_pre_hook(get_hook(idx_mask))
        # handle_list.append(handle)

        
        num_injections = torch.count_nonzero(idx_mask == 0) /32
        number_list[0] = number_list[1]
        number_list[1] = num_injections


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
        #print('Accuracy of the modified network on the {} test images: {} %'.format(512, 100 * accuracy))
        accuracy_list[0] = accuracy_list[1]
        accuracy_list[1] = accuracy

        if accuracy <= accuracy_list[0]:
            reward = 1
        else:
            reward = -1

        if num_injections <= number_list[0]:
            reward += 1
        else:
            reward += -1

        
        if accuracy > 0.9 * accuracy_original and accuracy < 0.95 * accuracy_original:
            reward += 5
        elif accuracy > 0.8 * accuracy_original and accuracy < 0.9 * accuracy_original:
            reward += 10
        elif accuracy > 0.7 * accuracy_original and accuracy < 0.8 * accuracy_original:
            reward += 20
        elif accuracy > 0.6 * accuracy_original and accuracy < 0.7 * accuracy_original:
            reward += 30
        elif accuracy > 0.5 * accuracy_original and accuracy < 0.6 * accuracy_original:
            reward += 40
        # elif accuracy < 0.5 * accuracy_original:
        #     reward += 50
        else:
            reward += -20
        
        # if num_injections < 10000 and num_injections > 8000:
        #     reward += 5
        # elif num_injections < 8000 and num_injections > 5000:
        #     reward += 10
        # elif num_injections < 5000 and num_injections > 3000:
        #     reward += 15
        # elif num_injections < 3000 and num_injections > 1000:
        #     reward += 20
        # else:
        #     reward -= 15
        # if num_injections < number_list[0]:
        #     reward += 5
        # else:
        #     reward += -5
        # if accuracy > 0.9 * accuracy_original and accuracy < 0.95 * accuracy_original:
        #     reward += 5
        # elif accuracy > 0.8 * accuracy_original and accuracy < 0.9 * accuracy_original:
        #     reward += 10
        # elif accuracy > 0.7 * accuracy_original and accuracy < 0.8 * accuracy_original:
        #     reward += 20
        # elif accuracy > 0.6 * accuracy_original and accuracy < 0.7 * accuracy_original:
        #     reward += 30
        # elif accuracy > 0.5 * accuracy_original and accuracy < 0.6 * accuracy_original:
        #     reward += 40
        # # elif accuracy < 0.5 * accuracy_original:
        # #     reward += 50
        # else:
        #     reward += -20
    
        # if accuracy  < 0.5 * accuracy_original:
        #      reward += 20

        info= {}
        
        done = bool(self.current_step >= self.max_steps)
                    #or accuracy < 0.5 * accuracy_original)
        self.current_step += 1
        # if self.current_step == self.max_steps:
        #     indices = torch.nonzero(idx_mask == 0)
        #     output_indices = indices.cpu().numpy()

        #     with open(file_path, 'a') as f:
        #         np.savetxt(f, output_indices)
        handle.remove()
        return self.state, reward, done, info, accuracy, idx_mask, top_indices_4d

    def reset(self):
        
    # Reset the environment to the initial state
        self.current_step = 0
        self.state = intermediate_output[-1]  # Replace with your state initialization
        return self.state

    def render(self, mode='human'):
        # Implement visualization of the environment if needed (e.g., for debugging)

        pass
        
        # def close(self):
        # # Clean up any resources or perform cleanup if necessary

model = AlexNet().to(device)
state_dict = torch.load('alex_dict_cifar10.pt')
model.load_state_dict(state_dict)

model.eval()

def hook_fn(module, input, output):
    #intermediate_output.append(output.detach())
    intermediate_output[0] = intermediate_output[1]
    intermediate_output[1] = output.detach()

# layer_index = 2    
intermediate_output = [torch.empty(0) for _ in range(2)]

# Choose the layer index you want to apply the hook to
#layer_index = 2  # layer 2: shut down corresponding neurons in layer 1

layer_index = 2
# Register the forward pre-hook to the desired layer
target_layer = model._modules[f'layer{layer_index}']  # Access the layer by index

#target_layer = model.fc

target_layer_ini = model._modules[f'layer{layer_index-1}']
handle_ini = target_layer_ini.register_forward_hook(hook_fn)
# target_layer.register_forward_hook(hook_fn)


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

handle_ini.remove()
accuracy_list = [accuracy_original, accuracy_original]

number_list = [0, 0]
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque([],maxlen=max_size)

    def add(self, state, action, next_state, reward, done):
        transition = (state, action, next_state, reward, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = map(np.array, zip(*batch))
        #states, actions, next_states, rewards, dones = map(lambda x: np.array(x.cpu().numpy()), zip(*batch))

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.buffer)


max_buffer_size = 1000
replay_buffer = ReplayBuffer(max_buffer_size)

class Actor(nn.Module):
    def __init__(self, tensor_channel):
        super(Actor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=tensor_channel, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.ConvTranspose2d(in_channels=512, out_channels=tensor_channel, kernel_size=3, stride=1),
            nn.Tanh(), 
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)  
        return x

class Critic(nn.Module):
    def __init__(self, tensor_channel):
        super(Critic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=tensor_channel*2, out_channels=512, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(9216,4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class DDPG:
    def __init__(self, tensor_channel, noise_std, ddpg_model):
        self.actor = Actor(tensor_channel).to(device)
        self.actor_target = Actor(tensor_channel).to(device)
        # self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.load_state_dict(ddpg_model['actor_state_dict']())
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=LR_ACTOR)
        self.actor_optimizer.load_state_dict(ddpg_model['actor_optimizer']())

        self.critic = Critic(tensor_channel).to(device)
        self.critic_target = Critic(tensor_channel).to(device)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.load_state_dict(ddpg_model['critic_state_dict']())
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=LR_CRITIC)
        self.critic_optimizer.load_state_dict(ddpg_model['critic_optimizer']())

        #self.max_action = max_action
        self.noise_std = noise_std

    def select_action(self, state):
        state = state.to(device)
        # action = self.actor(state).cpu().data.numpy()
        # action += np.random.normal(0, self.noise_std, size=action.shape)
        action = self.actor(state)
        noise = torch.rand_like(action, device=device) * self.noise_std
        action += noise
        return action

    def train(self, replay_buffer, batch_size):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        state = torch.squeeze(torch.FloatTensor(state)).to(device)
        action = torch.squeeze(torch.FloatTensor(action)).to(device)
        next_state = torch.squeeze(torch.FloatTensor(next_state)).to(device)
        reward = torch.FloatTensor(reward).to(device)
        not_done = torch.FloatTensor(1 - not_done).to(device)
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (not_done * GAMMA * target_Q).detach()

        current_Q = self.critic(state, action)

        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


BATCH_SIZE = 1
GAMMA = 0.99
TAU = 0.005
LR = 1e-4
LR_ACTOR = 0.001
LR_CRITIC = 0.001
env = NeuronEnv()
ddpg_model = torch.load('ddpg_model.pt')
# for param in model.parameters():
#     param.requires_grad = False
# print(ddpg_model)
# print(ddpg_model['actor_state_dict'])
agent = DDPG(tensor_channel=intermediate_output[-1].shape[1], noise_std=0.1, ddpg_model=ddpg_model)
# file_path = "indices.txt"                                                                                     
# with open(file_path, 'w') as f:
#     # Write new content to the file
#     f.write("This is the new content of the file.\n")

if torch.cuda.is_available():
    num_episodes = 30
else:
    num_episodes = 2 

handle_list = []
total_reward = []
average_reward = []
accuracy_list_out = []

num_list = []
for i_episode in range(1, num_episodes+1):
    print(f"This is episode {i_episode}")
    # Initialize the environment ange get it's state
    state = env.reset()
    #state = torch.tensor(state, dtype=torch.float32, device=device)
    score = 0
    
        
    for t in count():
        #print(state.shape)
        action = agent.select_action(state)

        next_state, reward, done, _, accuracy, mask, top_indices_4d = env.step(action)
        score += reward
        
        replay_buffer.add(np.asnumpy(state), np.asnumpy(action.detach()), np.asnumpy(next_state), reward, done)

        # Move to next_state
        state = next_state

        # for handle in handle_list:
        #         handle.remove()
        # handle_list.clear()

        

        if len(replay_buffer) > BATCH_SIZE:
            agent.train(replay_buffer, BATCH_SIZE)

        if done:
            indices = torch.nonzero(mask == 0)
            num_injections = indices.size(0) / 32 #/ (512 * 32)
            np.save('indices.npy', top_indices_4d)
            torch.save(top_indices_4d, 'indices.pt')
            
            # with open('indices.json', 'w') as f:
            #     json.dump(top_indices_4d, f)
            # episode_durations.append(t + 1)
            next_state = None
            total_reward.append(score)
            average_reward.append(score/(t+1))
            accuracy_list_out.append(accuracy)
            num_list.append(num_injections)
            torch.save({
                'actor_state_dict': agent.actor.state_dict,
                'actor_optimizer': agent.actor_optimizer.state_dict,
                'critic_state_dict': agent.critic.state_dict,
                'critic_optimizer': agent.critic_optimizer.state_dict,
            }, 'ddpg_model_tmp.pt')
            
            torch.cuda.empty_cache()
            break
#torch.save(policy_net, 'policy_net_final_model.pth')
# torch.save(optimizer.state_dict, 'optimizer_final.pth')
# torch.save(policy_net_state_dict, 'policy_net_final_dict.pth')
torch.save({
        'actor_state_dict': agent.actor.state_dict,
        'actor_optimizer': agent.actor_optimizer.state_dict,
        'critic_state_dict': agent.critic.state_dict,
        'critic_optimizer': agent.critic_optimizer.state_dict,
            }, 'ddpg_model_final.pt')
print('Complete')

plt.plot(torch.arange(len(total_reward)), total_reward)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('DQN Evaluation: Total Reward per Episode')
#plt.show()
plt.savefig('total_reward.png')


plt.plot(torch.arange(len(average_reward)), average_reward)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('DQN Evaluation: Average Reward per Episode')
#plt.show()
plt.savefig('average_reward.png')

import csv
# Assuming epoch_rewards contains the list of average rewards per epoch

# Define the file name for the CSV
csv_filename_final = "final_output.csv"


# Writing data to CSV file
with open(csv_filename_final, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Episode', 'Total Reward', 'Average Reward', 'Final Accuracy', 'Injection Number'])  # Write header
    idx_list = list(range(1, num_episodes+1))
    # for idx, reward_t in enumerate(total_reward, start=1):
    #     writer.writerow([idx, reward_t])
    rows = zip(idx_list, total_reward, average_reward, accuracy_list_out, num_list)
    writer.writerows(rows)
print("Data written to", csv_filename_final)
