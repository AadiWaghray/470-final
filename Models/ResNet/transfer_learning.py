import torch
import torch.nn as nn

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
num_ft = model.fc.in_features
model.fc = nn.Linear(num_ft, 7)

criterion = nn.MSELoss()
#LR set according to MSPTL paper -> How set only for fc or conv
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#hold out CV 10-fold
#Train on Dataset1
#https://www.kaggle.com/datasets/msambare/fer2013

#hold out CV 10-fold
#Train on Dataset2
#https://www.kaggle.com/datasets/noamsegal/affectnet-training-data
#Processing needed
#Remove contempt

#hold out CV 5-fold
#Train on best Dataset3 or augmented Dataset?
#https://www.kaggle.com/datasets/shahzadabbas/expression-in-the-wild-expw-dataset

#Train 4 epochs
#Validate once per epoch
#Auto-stop if validation stops improving
