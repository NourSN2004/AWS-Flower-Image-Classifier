
import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

def get_input_args():
    # define command line arguments
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset")

    parser.add_argument('data_dir', type=str, help='Directory containing the training data')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory for saving checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', choices=['vgg13', 'resnet18'], help='Model architecture to use for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--layer_1_hidden_units', type=int, default=256, help='Number of hidden units in first hidden layer of classifier')
    parser.add_argument('--layer_2_hidden_units', type=int, default=128, help='Number of hidden units in second hidden layer of classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')

    return parser.parse_args()

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    # define transforms for the training and validation sets
    train_transforms = transforms.Compose([transforms.RandomRotation(15),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.CenterCrop((224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

    return trainloader, validloader, train_data.class_to_idx

def build_model(arch='vgg13', layer_1_hidden_units=256, layer_2_hidden_units=128):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_features = 25088
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_features = 512
    else:
        raise ValueError(f"Architecture '{arch}' is not supported. Choose 'vgg13' or 'resnet18'.")

    # freeze pre-trained model parameters
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, layer_1_hidden_units)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(layer_1_hidden_units, layer_2_hidden_units)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(layer_2_hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    if arch == 'vgg13':
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier  # only set the 'fc' layer for ResNet

    return model

def validation(model, validloader, criterion, device):
    model.eval()
    accuracy = 0
    loss = 0
    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss += criterion(output, labels).item()
            accuracy += (output.argmax(dim=1) == labels).float().mean().item()

    return loss / len(validloader), accuracy / len(validloader)

def train_model(model, trainloader, validloader, criterion, optimizer, epochs, device):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        valid_loss, valid_accuracy = validation(model, validloader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Training Loss: {running_loss/len(trainloader):.3f}.. "
              f"Validation Loss: {valid_loss:.3f}.. "
              f"Validation Accuracy: {valid_accuracy:.3f}")

    return model

def save_checkpoint(model, save_dir, class_to_idx, arch, layer_1_hidden_units, layer_2_hidden_units, learning_rate, epochs):
    checkpoint = {
        'arch': arch,
        'layer_1_hidden_units': layer_1_hidden_units,
        'layer_2_hidden_units': layer_2_hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'learning_rate': learning_rate,
        'epochs': epochs
    }

    save_path = os.path.join(save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)

def main():
    args = get_input_args()

    # load data
    trainloader, validloader, class_to_idx = load_data(args.data_dir)

    # build model
    model = build_model(args.arch, args.layer_1_hidden_units, args.layer_2_hidden_units)

    # define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    # set device to GPU if available and requested
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # train model
    model = train_model(model, trainloader, validloader, criterion, optimizer, args.epochs, device)

    # save checkpoint
    save_checkpoint(model, args.save_dir, class_to_idx, args.arch, args.layer_1_hidden_units, args.layer_2_hidden_units, args.learning_rate, args.epochs)

if __name__ == "__main__":
    main()
