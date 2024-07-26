import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# Import the function to create the modified model
from model import ResNet50WithDropout

# Parameters
save_dir = 'src'
data_dir = 'data'
batch_size = 32
learning_rate = 0.001
num_epochs = 30
dropout_rate = 0.5

# Create directory to save models if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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
model = ResNet50WithDropout(dropout_rate)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(device)

# Configure loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, train_loader, criterion, optimizer, epoch, train_losses):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f'Epoch {epoch}, Training Loss: {epoch_loss:.4f}')

# List to store training losses
train_losses = []

# Train the model
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch, train_losses)

# Save the model and parameters after training
model_path = os.path.join(save_dir, 'resnet50_with_dropout(30ep).pth')
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
    'dropout_rate': dropout_rate,
}, model_path)

print(f"Training completed! Model saved to {model_path}")

# Plot the training loss
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
