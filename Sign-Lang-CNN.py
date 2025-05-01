"""
NAME: Sign-Lang-CNN.py
DESCRIPTION: CNN algorithm to classify sign language for our capstone
PROGRAMMER: Caidan Gray and Matteo Leonard
CREATION DATE: 3/3/2025
LAST EDITED: 5/1/2025   (please update each time the script is changed)
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
import pandas as pd
import time
# import data_loader
import cv2

# device config
# sets it to run on gpu if supported
start_time = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# hyper-parameters
num_epochs = 50
batch_size = 4
learning_rate = 0.01

data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# dataset of PILImage images of range [0, 1]
# transform to tensors of normalized range [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


class PixelDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = self.data.iloc[idx, 1:].values.astype(np.uint8)
        image = pixels.reshape(28, 28)
        label = self.data.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float()

        return image, label


# Example Usage
train_csv_file = 'archive (3)/sign_mnist_train.csv'
test_csv_file = 'archive (3)/sign_mnist_test.csv'

# Define transformations if needed
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = PixelDataset(train_csv_file, transform=transform)
test_dataset = PixelDataset(test_csv_file, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# conv net implementation
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # input channel is the colors (RGB)
        # input channel must bethe same size as previous output channel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 25)


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

        if i % 2000 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, step {i+1} of {n_total_steps}, loss {loss.item():.4f}")
            print(f"time: {time.time()-epoch_time}")
print("training completed")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(25)]
    n_class_samples = [0 for i in range(25)]
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

    for i in range(25):
        if i != 9 and i != 26:
            print(n_class_correct[i], n_class_samples[i])
            acc = 100.0 * (n_class_correct[i] / n_class_samples[i])
            print(f"accuracy of {classes[i]}: {acc:.4f}%")
    print(f"total time: {time.time() - start_time}")
    # save the model
    torch.save(model, 'model.pth')

def preprocess(image):
    #resize frame to 32x32
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
    # reshape frame to fit input prams
    image = torch.from_numpy(image).float().reshape(1, 3, 32, 32)
    print(f"{image}\n")
    return image

def __draw_label(img, text, pos, bg_color):
    # set font scale and color
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)
    #place the text at the top left
    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cam = cv2.VideoCapture(0)
frame_count = 0
# load the model in eval mode
model = torch.load('model.pth', weights_only=False)
model.eval()

predicted = 0

while True:
    ret, frame = cam.read()  # Capture each frame
    # run model every 10 frames
    if frame_count == 10:
        # reset frame count
        frame_count = 0
        img = preprocess(frame)
        output = model(img)
        predicted = output.data.numpy().argmax()
        print(classes[predicted])
    __draw_label(frame, classes[predicted], (20,35), (255,255,255))

    cv2.imshow('frame', frame)
    frame_count += 1
    # exit window is esc key is pressed
    if cv2.waitKey(1) == 27:
        cv2.destroyWindow('frame')
        cam.release()
