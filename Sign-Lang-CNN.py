"""
NAME: Sign-Lang-CNN.py
DESCRIPTION: CNN algorithm to classify sign language for our capstone
PROGRAMMER: Caidan Gray
CREATION DATE: 3/3/2025
LAST EDITED: 4/4/2025   (please update each time the script is changed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import sklearn
import data_loader
import cv2
import PIL

# device config
# sets it to run on gpu if supported
start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# data loader
#Data_Loader = data_loader.Data_Loader()



# hyper-parameters
num_epochs = 1
batch_size = 4
learning_rate = 0.001

data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# dataset of PILImage images of range [0, 1]
# transform to tensors of normalized range [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
"""
train_dataset = Data_Loader.train_loader()

test_dataset =  Data_Loader.test_loader()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
"""

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# conv net implementation
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # input channel is the colors (RGB)
        # input channel must bethe same size as previous output channel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

# loss and optimization
criterion = nn.CrossEntropyLoss()  # applies softmax for us
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        epoch_time = time.time()
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, step {i+1} of {n_total_steps}, loss {loss.item():.4f}")
            print(f"time: {time.time()-epoch_time}")
print("training completed")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (values, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * (n_correct / n_samples)
    print(f"accuracy: {acc:.4f}%")

    for i in range(10):
        acc = 100.0 * (n_class_correct[i] / n_class_samples[i])
        print(f"accuracy of {classes[i]}: {acc:.4f}%")

def preprocess(image):
    image = PIL.Image.fromarray(image) #Webcam frames are numpy array format
                                       #Therefore transform back to PIL image
    print(image)
    image = data_transforms(image)
    image = image.float()
    image = Variable(image)
    #image = image.cuda()
    image = image.unsqueeze(0) #I don't know for sure but Resnet-50 model seems to only
                               #accpets 4-D Vector Tensor so we need to squeeze another
    return image                            #dimension out of our 3-D vector Tensor

def __draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 0.4
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

print(f"total time: {time.time()-start_time}")

torch.save(model, 'model.pth')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cam = cv2.VideoCapture(0)
frame_count = 0
model = torch.load('model.pth', weights_only=False)
model.eval()


while True:
    ret, frame = cam.read()  # Capture each frame

    if frame_count == 30:
        frame_count = 0
        img = preprocess(frame)
        output = model(img)



    __draw_label(frame, 'Hello World', (20,20), (255,0,0))

    cv2.imshow('frame', frame)
    frame_count += 1
    if cv2.waitKey(1) == 27:
        cv2.destroyWindow('frame')
        cam.release()





