import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.datasets import CIFAR100   
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import csv
import os
import argparse
import sys    
from torchvision.models import alexnet, vgg19, resnet50, mobilenet_v3_large, inception_v3
 
from utils.datasets import SuperCIFAR100, GeneratedDataset, tf ,CIFAR_100_CLASS_MAP,generate_full_subclass_map
from torch.utils.data import ConcatDataset, DataLoader 

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a classifier with custom dataset')
    parser.add_argument('--gennum', type=int, required=True, help='Generator number for filename customization')
    parser.add_argument('--model_name', type=str, choices=['alexnet', 'vgg19', 'resnet50', 'mobilenetv3', 'inceptionv4'], required=True, help='Model to use for classification')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    return parser.parse_args()

args = parse_arguments()
def get_model(model_name, num_classes, device):
    if model_name.lower() == 'alexnet':
        model = alexnet(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name.lower() == 'vgg19':
        model = vgg19(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name.lower() == 'resnet50':
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.lower() == 'mobilenetv3':
        model = mobilenet_v3_large(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name.lower() == 'inceptionv4':
        model = inception_v3(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported model name")
    return model.to(device)
 
args = parse_arguments()

# 加载模型
def load_model(model_path, model_name, num_classes, device):
    model = get_model(model_name, num_classes, device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
 
def prepare_testset():
    testset = SuperCIFAR100(root='../data', train=False, download=True, transform=tf)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return test_loader
 
def test_model(model, test_loader, device):
    results = []
    image_id = 0 
    with torch.no_grad():
        for images, superclass, subclass in test_loader:
            images, superclass, subclass = images.to(device), superclass.to(device), subclass.to(device)
            outputs = model(images)
            _, predicted_superclass = torch.max(outputs, 1)
            for  sup_cls, sub_cls, pred_sup_cls in zip(  superclass, subclass, predicted_superclass):
                results.append((image_id, sup_cls.item(), sub_cls.item(), pred_sup_cls.item()))
                image_id += 1
    return results
 
def write_results_to_csv(results, model_name,full_subclass_map):
    results_dir = f'results/{model_name}'
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'test_results_{args.gennum}.csv')
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'True Superclass', 'True Subclass', 'Predicted Superclass', 'True Superclass Name', 'True Subclass Name'])
        for img, true_superclass, true_subclass, pred_superclass in results:
            true_superclass_name = list(CIFAR_100_CLASS_MAP.keys())[true_superclass]
            
            true_subclass_name = full_subclass_map[true_subclass]
            print(f"Superclass: {true_superclass_name},Subclass name {true_subclass_name}")
            
            writer.writerow([img, true_superclass, true_subclass, pred_superclass, true_superclass_name, true_subclass_name])
    print(f"Results saved to {csv_path}")

# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = f"./models/{args.model_name}/{args.model_name}_gen{args.gennum}.pth"
    model = load_model(model_path, args.model_name, 20, device)  
 
    test_loader = prepare_testset()
    results = test_model(model, test_loader, device)
    full_subclass_map = generate_full_subclass_map()
    print("full_subclass_map\n",full_subclass_map)
    write_results_to_csv(results, args.model_name, full_subclass_map)
    

if __name__ == "__main__":
    main()
