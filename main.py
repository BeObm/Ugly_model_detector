import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Step 2: Prepare the Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root='dataset_ugly/train', transform=transform)
test_dataset = datasets.ImageFolder(root='dataset_ugly/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 3: Define the Model
class DressClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DressClassifier, self).__init__()
        self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        in_features = self.resnet.fc.in_features
        # Remove the last fully connected layer of the pre-trained ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten the output from the ResNet
        x = self.fc_layers(x)
        return x

# Step 4: Define Loss Function and Optimizer
model = DressClassifier(num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.00001)

# Step 5: Training Loop
num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Step 6: Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')
torch.save(model.state_dict(), 'ugly_model.pth')