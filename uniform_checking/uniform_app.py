import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import os
from PIL import Image
import random
import io

# Load the trained model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Linear(self.features.fc.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x

# Initialize the model
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Streamlit app
st.title("Swinburne Uniform Checking App")
st.write("Upload an image and the model will predict if the person is wearing Swinburne uniform or not.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an image
    img = Image.open(uploaded_file).convert('RGB')

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=False)

    # Preprocess the image
    img = transform(img)
    img = img.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = " This person is wearing Swinburne Uniform" if predicted.item() == 1 else "This person is not wearing Swinburne Uniform"

    # Display the prediction
    st.write(f"The model predicts: **{label}**")
