import requests
import os
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import streamlit as st

classification_model_path = "Classification_Model.pth"
model_url = "https://drive.google.com/file/d/1-1Ty56S3zITswAOr9LIQtctzcP5qsnvV/view?usp=drive_link"  # Replace with your actual model URL

def download_model(url, dest):
    if not os.path.exists(dest):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        if response.status_code == 200:
            with open(dest, 'wb') as file:
                for data in response.iter_content(1024):
                    file.write(data)
            print(f"Model downloaded to {dest}")
        else:
            print(f"Failed to download model. Status code: {response.status_code}")
    else:
        print(f"Model already exists at {dest}")

# Ensure model is downloaded
download_model(model_url, classification_model_path)

# CNN Architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=96)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout1 = nn.Dropout(p=0.3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(in_features=43264, out_features=1024)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=36)
        self.fc4 = nn.Linear(in_features=36, out_features=6)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.dropout2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

# Transformation to images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform_image(image):
    image = Image.open(image)
    transformed_image = transform(image)
    transformed_image = transformed_image.to(device)
    transformed_image = transformed_image.unsqueeze(0)
    return transformed_image

model = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(classification_model_path))
    else:
        model.load_state_dict(torch.load(classification_model_path, map_location=torch.device('cpu')))
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")

class_mapping = {
    0: 'Half Sleeves',
    1: 'Shorts',
    2: 'Caps',
    3: 'Footwear',
    4: 'Bottoms',
    5: 'Full Sleeves'
}

model.to(device)
model.eval()

def predict(image):
    output = model(transform_image(image))
    _, predicted = torch.max(output, 1)
    return class_mapping[predicted.item()]

# Streamlit application
st.title("Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(uploaded_file)
    st.write(f"Prediction: {label}")
