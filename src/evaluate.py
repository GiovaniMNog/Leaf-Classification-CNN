import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Import the function to create the modified model
from model import ResNet50WithDropout

# Parameters
data_dir = 'data'
batch_size = 32
model_path = 'src/resnet50_with_dropout(30ep).pth'
dropout_rate = 0.5

# Normalization transformations for validation
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the validation dataset
val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=val_transforms)

# DataLoader
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create the model
model = ResNet50WithDropout(dropout_rate)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Load the saved model
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
criterion = nn.CrossEntropyLoss()

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct.double() / len(val_loader.dataset)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print classification report
    print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))
    
    return epoch_loss, accuracy, cm

# Validate the model
val_loss, val_accuracy, cm = validate(model, val_loader, criterion)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_dataset.classes, yticklabels=val_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
