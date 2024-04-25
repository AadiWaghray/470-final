from torch.utils.data.dataset import ConcatDataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_save():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    num_ft = model.fc.in_features
    model.fc = nn.Linear(num_ft, 7)

    torch.save({"model_state_dict": model.state_dict()}, "init_resnet.pth")

def load(path):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34')
    model.load_state_dict(torch.load(path))

def train_validate(model, epochs, k_folds):
    pre_fc_param = [
        model.layer1.parameters(),
        model.layer2.parameters(),
        model.layer3.parameters(),
        model.layer4.parameters(),
    ]
    fc_param = model.fc.parameters()


    criterion = nn.MSELoss()
    #LR set according to MSPTL paper -> How set only for fc or conv
    optimizer = torch.optim.SGD([
        {"params": pre_fc_param},
        {"params": fc_param, "lr": 0.001},
    ], lr=0.0001, momentum=0.9)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    fer2013_train = '../../Data/FER2013/train'
    fer2013_test = '../../Data/FER2013/test'


    train_dataset = datasets.ImageFolder(root=fer2013_train, transform=data_transform)
    test_dataset = datasets.ImageFolder(root=fer2013_test, transform=data_transform)
    dataset = ConcatDataset([train_dataset, test_dataset])
    print("Loaded Dataset")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    loss_function = nn.MSELoss()

    results = {}

    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f"Started fold {fold+1}")
        train_sample = torch.utils.data.SubsetRandomSampler(train_ids) 
        test_sample = torch.utils.data.SubsetRandomSampler(test_ids) 

        train_loader = DataLoader(dataset, batch_size=10, sampler=train_sample)
        test_loader = DataLoader(dataset, batch_size=10, sampler=test_sample)
        model.train()
        train(train_loader,
              optimizer,
              epochs,
              model,
              criterion)

        torch.save({"model_state_dict": model.state_dict()}, f"renset34-FER2013-epoch{fold+1}.pth")

        correct, total = validate(model, test_loader)
        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')

  
def train(train_loader,
          optimizer,
          epochs,
          model,
          criterion):
    for epoch in range(epochs):
        print(f"Started epoch {epoch+1}")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}finished, Loss: {running_loss / len(train_loader)}')
        print("Saved model")

def validate(model, test_loader):
    # Evaluationfor this fold
    correct, total = 0, 0
    with torch.no_grad():

      # Iterate over the test data and generate predictions
      for i, data in enumerate(test_loader, 0):

        # Get inputs
        inputs, targets = data

        # Generate outputs
        outputs = model(inputs)

        # Set total and correct
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    return correct, total

      
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

def main():
    print("main")

if __name__ == "main":
    main()
