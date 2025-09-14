"""
# IFN680 project 1 Individual Report
Student Name: Le Khanh Linh Pham
Student ID: 10364960

Task 3 - small model

"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x
        
class MyClassifier():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = 'light_model.pth'
        
        self.class_labels = [
            'edible_1',
            'edible_2',
            'edible_3',
            'edible_4',
            'edible_5',
            'poisonous_1',
            'poisonous_2',
            'poisonous_3',
            'poisonous_4',
            'poisonous_5'
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))]) 
        self.threshold = 0.1

        
    def setup(self, f_load=True):
        self.model = Classifier()
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        self.classifier = self.model.classifier
            

    def test_image(self, image):
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            predicted = torch.argmax(outputs, dim=1).item()
        return self.class_labels[predicted]
        
 
    def test_image_calibrated(self, image):
        if self.model is None:
            self.setup()
            
        x = self.transform(image).unsqueeze(0)
        inputs = x.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=-1)[0]

        poison_prob = sum([probs[i].item() for i in range(5, 10)])

        is_poisonous = poison_prob > self.threshold
        
        return is_poisonous


        