# train_and_save_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import argparse
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Function to prepare the dataset
def prepare_dataset(train_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def define_model(num_classes, use_pretrained=True):
    class DressClassifier(nn.Module):
        def __init__(self):
            super(DressClassifier, self).__init__()
            self.resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
            in_feat=self.resnet.fc.in_features
            # Remove the last fully connected layer of the pre-trained ResNet
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

            # Add additional layers
            self.fc_layers = nn.Sequential(
                nn.Linear(in_feat, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.resnet(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x

    model = DressClassifier()
    return model


# Function to train and save the model
def train_and_save_model(train_loader, num_epochs=50, learning_rate=0.0001, momentum=0.9, weight_decay=0.0001,model_save_path='model.pth'):
    model = define_model(num_classes=len(train_loader.dataset.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,weight_decay=weight_decay)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f'Training finished. Model saved to {model_save_path}')
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", help="path_to_train_dataset", default='dataset_ugly/train')
    parser.add_argument("--test_path", help="path_to_test_dataset", default='dataset_ugly/test')
    parser.add_argument("--num_epochs", help="num_epochs", default=50)
    parser.add_argument("--lr", help="learning_rate", default=0.0001)
    parser.add_argument("--momentum", help="momentum", default=0.9)
    parser.add_argument("--weight_decay", help="weight_decay", default=0.00001)
    parser.add_argument("--model_save_path", help="model_save_path", default='ugly_model.pth')
    args = parser.parse_args()
    print('Preparing dataset for training...')
    train_loader = prepare_dataset(args.train_path)
    print('Start training...')
    model = train_and_save_model(train_loader, args.num_epochs, args.lr, args.momentum,args.weight_decay, args.model_save_path)

    test_loader = prepare_dataset(args.test_path)
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
