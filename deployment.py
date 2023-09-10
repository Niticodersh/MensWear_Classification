from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import joblib
import pickle

classification_model_path="Classification_Model.pth"

# CNN Architectue ---------------------------------------------------------------------------------------------------------------------
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=1)
    self.bn1 = nn.BatchNorm2d(num_features=96)
    self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride= 2)
    self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
    self.bn2 = nn.BatchNorm2d(num_features=256)
    self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.dropout1 = nn.Dropout(p=0.3)
    self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
    self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
    self.dropout2 = nn.Dropout(p=0.3)
    self.fc1= nn.Linear(in_features=43264, out_features=1024)
    self.dropout3 = nn.Dropout(p=0.5)
    self.fc2= nn.Linear(in_features=1024,out_features=128)
    self.fc3= nn.Linear(in_features=128,out_features=36)
    self.fc4= nn.Linear(in_features=36, out_features=6)
    self.relu = nn.ReLU(inplace = True)

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
    x = x.view(x.size()[0],-1)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout3(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.relu(x)
    x = self.fc4(x)

    return x
# Transformation to images -------------------------------------------------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform_image(image):
    # Load the image using PIL
    image = Image.open(image)
    # Apply the transformation pipeline
    transformed_image = transform(image)

    # Move the tensors to the appropriate device
    transformed_image = transformed_image.to(device)

    # Add an extra dimension to match the model input shape
    transformed_image = transformed_image.unsqueeze(0)
    return transformed_image



print(torch.cuda.is_available())
model = CNN()  # Create an instance of the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    model.load_state_dict(torch.load(classification_model_path))
else:
    model.load_state_dict(torch.load(classification_model_path, map_location=torch.device('cpu')))
class_mapping={0:'Half Sleeves',
 1:'Shorts',
 2:'Caps',
 3:'Footwear',
 4:'Bottoms',
 5:'Full Sleeves'}

model.to(device)
model.eval()
def predict(image):
    output = model(transform_image((image)))
    _,predicted = torch.max(output, 1)
    return class_mapping[predicted.item()]



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        image = request.files['image']
        if image:
            result = predict(image)

    return render_template('index.html', result1=result)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
