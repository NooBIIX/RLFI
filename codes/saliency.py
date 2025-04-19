import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

# Load pre-trained VGG16 model
model = models.vgg16(pretrained=True)
model.eval()

# Define preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to compute saliency map
def compute_saliency_map(model, input_image, target_class=None):
    input_image.requires_grad_()
    model.zero_grad()
    
    output = model(input_image)
    
    if target_class is None:
        target_class = torch.argmax(output)
    
    output[0, target_class].backward()
    
    gradient = input_image.grad.data
    saliency_map, _ = torch.max(gradient.abs(), dim=1)
    
    return saliency_map

# Path to your input image
img_path = '1.jpg'

# Load and preprocess the image
input_image = Image.open(img_path)
input_tensor = preprocess(input_image).unsqueeze(0)

# Compute the saliency map
saliency_map = compute_saliency_map(model, input_tensor)

# Normalize the saliency map
saliency_map = saliency_map.detach().numpy()
saliency_map /= np.max(saliency_map)

# Plot original image and saliency map
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(saliency_map[0], cmap='hot')
plt.title('Saliency Map')
plt.axis('off')

plt.show()
