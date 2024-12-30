
import json
import argparse
import torch
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from collections import OrderedDict
import numpy as np


def get_input_args():
    parser = argparse.ArgumentParser(description="Predict the class of an input image using a trained deep learning model.")
    parser.add_argument('input', type=str, help='Path to the input image.')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=str, default="./cat_to_name.json", help='Path to JSON file mapping categories to real names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available.')
    return parser.parse_args()
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))  # ensure compatibility
    arch = checkpoint['arch']
    layer1 = checkpoint['layer_1_hidden_units']
    layer2 = checkpoint['layer_2_hidden_units']
    
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_features = 25088
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_features = 512
    else:
        raise ValueError("Unsupported architecture.")
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, layer1)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(layer1, layer2)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(layer2, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    if arch == 'vgg13':
        model.classifier = classifier
    elif arch == 'resnet18':
        model.fc = classifier  # only set the 'fc' layer for ResNet
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path, device='cpu'):
    img = Image.open(image_path).convert("RGB")
    inference_transforms = transforms.Compose([
        transforms.Resize((256, 256)),                # resize to 256x256
        transforms.CenterCrop((224, 224)),            # center crop to 224x224
        transforms.ToTensor(),                        # convert PIL Image to Tensor and scale between [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = inference_transforms(img)
    img_tensor = img_tensor.unsqueeze(dim=0).to(device)
    return img_tensor

def predict(image_path, model, top_k=1, device='cpu'):
    model.to(device)
    model.eval()
    
    # process image
    tensor_image = process_image(image_path, device=device)

    # disable gradient computation for inference
    with torch.no_grad():
        log_output = model(tensor_image)

        # exponentiate the log-probabilities to get probabilities
        probabilities = torch.exp(log_output)

        # get the top K probabilities and their corresponding indices
        top_probs, top_indices = probabilities.topk(top_k, dim=1)

    # remove batch dimension by detaching and converting to numpy
    top_probs = top_probs.detach().cpu().numpy()[0]
    top_indices = top_indices.detach().cpu().numpy()[0]

    # convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probs.tolist(), top_classes

def main():
    args = get_input_args()

    # load the model
    model = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # run prediction
    probs, classes = predict(args.input, model, top_k=args.top_k, device=device)
    
    # load category names if provided
    if args.category_names:
        try:
            with open(args.category_names, 'r') as f:
                cat_to_name = json.load(f)
            classes = [cat_to_name.get(c, "Unknown") for c in classes]  # handle missing categories
        except FileNotFoundError:
            print(f"Category names file {args.category_names} not found.")
    
    # print results
    for i in range(len(classes)):
        print(f"Class: {classes[i]}, Probability: {probs[i]:.3f}")

if __name__ == "__main__":
    main()
