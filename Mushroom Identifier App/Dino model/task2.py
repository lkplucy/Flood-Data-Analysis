"""
# IFN680 project 1 Individual Report
Student Name: Le Khanh Linh Pham
Student ID: n10364960

Task 2 - best model

"""

import torch
from torchvision import transforms
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dino.eval()
dino.to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))])


class Classifier(nn.Module):
    """
    Define your best model here.

    """

    def __init__(self,f_load=True):
        super().__init__()
        self.fc = nn.Linear(384, 10)
        
    def forward(self, x): 
        x = self.fc(x)
        return x

class MyClassifier():
    
    ''' Myclassifier wrapper for the best model. Do not change the class name. Do not change any of the existing function names.'''
    
    def __init__(self):
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
        self.threshold = 0.3
        

    def setup(self,f_load=True):
        ''' This function will initialise your model. 
            You will need to load the model architecture and load any saved weights file your model relies on.
        '''
        self.model = Classifier()
        self.model.load_state_dict(torch.load('dino_linear_classifier_best.pth', map_location =device))
        self.model.to(device)
        self.model.eval()
        
            

    def test_image(self, image):
        ''' This function will be given a PIL image, and should return the predicted class label for that image. 
            Currently the function is returning a random label.
                
        '''
        x = transform(image).unsqueeze(0)
        inputs = x.to(device)
        
        with torch.no_grad():
            features = dino(inputs)
            outputs = self.model(features)
            predicted = torch.argmax(outputs, axis = 1)

        return self.class_labels[predicted.item()]
        
    
    
    def test_image_calibrated(self, image):
        ''' This function will be given a PIL image, and should return True if the mushroom is poisonous and False otherwise.
        '''
        x = transform(image).unsqueeze(0)
        inputs = x.to(device)
        
        with torch.no_grad():
            features = dino(inputs)
            outputs = self.model(features)
            probs = torch.softmax(outputs, dim=-1)
            poison_prob = probs[0, 5:].sum().item()
        is_poisonous = poison_prob > self.threshold
        
        return is_poisonous
