import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Importe a função para criar o modelo modificado
from model import create_resnet50_with_dropout

# Parameters
data_dir = 'data'
batch_size = 32
learning_rate = 0.001
num_epochs = 10
dropout_rate = 0.5

# Transformations for data augmentation and normalization
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the data 
train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transforms)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create the model
model = create_resnet50_with_dropout(dropout_rate)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Configurate loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}, Training Loss: {epoch_loss:.4f}')

# Model training
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)

print("Training concluded!")
