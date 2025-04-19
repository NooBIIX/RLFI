import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define your neural network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x

# Load your data
# Example:
# data = ...
data = torch.rand(3, 32, 32)
# Create an instance of your network
model = MyNetwork()

# Load pretrained weights (optional)
# Example:
# model.load_state_dict(torch.load('model.pth'))

# Set the model to evaluation mode
model.eval()

# Forward pass to get activations
with torch.no_grad():
    activations = model(data)

# Plot the activations
num_activations = activations.size(1)  # Number of channels in the layer
num_rows = 4  # Number of rows in the grid
num_cols = num_activations // num_rows  # Number of columns in the grid
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

for i in range(num_rows):
    for j in range(num_cols):
        ax = axes[i, j]
        activation = activations[0, i * num_cols + j].cpu().numpy()  # Select the activation map
        ax.imshow(activation, cmap='viridis')  # Plot the activation map
        ax.axis('off')

plt.tight_layout()
plt.show()
