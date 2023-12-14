# test_and_save_predictions.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import pandas as pd
import os
import argparse
from train_model import *
from tqdm import tqdm
from PIL import ImageFile,Image
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image


ImageFile.LOAD_TRUNCATED_IMAGES = True


# Function to prepare the test dataset
def prepare_test_dataset(test_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


# Function to load the trained model
def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Function to make predictions and save results to Excel
def predict_and_save(model, dataloader, output_excel_path='predictions.xlsx'):
    model.eval()
    predictions = []
    file_paths = []

    with torch.no_grad():
        for inputs, paths in dataloader:
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs, 1)

            predictions.extend(predicted_labels.tolist())
            file_paths.extend(paths)

            # Remove duplicates from file_paths
        unique_file_paths = list(set(file_paths))
        unique_predictions = [predictions[file_paths.index(path)] for path in unique_file_paths]

        return unique_file_paths, unique_predictions

def save_to_excel(file_paths, predictions, excel_file='predictions.xlsx'):
    df = pd.DataFrame({'File Path': file_paths, 'Predicted Label': predictions})
    df.to_excel(excel_file, index=False)
    print(f"Results saved to {excel_file}")

def create_image_dataloader(data_folder, batch_size=32, transform=None):
    class CustomImageDataset(Dataset):
        def __init__(self, root_dir, transform=None, valid_extensions=(".jpg", ".jpeg", ".png")):
            self.root_dir = root_dir
            self.transform = transform
            self.valid_extensions=valid_extensions
            self.file_list = self.get_file_list()

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            # img_path = os.path.join(self.root_dir, self.file_list[idx])
            try:
                img_path = self.file_list[idx]
                image = Image.open(img_path).convert('RGB')

                if self.transform:
                    image = self.transform(image)

                return  image, img_path
            except:
                pass

        def get_file_list(self):
            file_list = set()  # Using a set to ensure unique file paths
            for subdir, dirs, files in os.walk(self.root_dir):
                for file in files:
                    if file.lower().endswith(self.valid_extensions):
                        file_list.add(f"{subdir}/{file}")
            return list(file_list)

    transform = transform

    # Create a custom dataset
    custom_dataset = CustomImageDataset(root_dir=data_folder, transform=transform)

    # Create a DataLoader
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    return data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", help="path_to_test_dataset", default="test_images")
    parser.add_argument("--model_path", help="saved_model_path", default='ugly_model.pth')
    parser.add_argument("--output_path", help="output_excel_path", default='predictions_results_test_sample.xlsx')
    args = parser.parse_args()

    print(f"Loading image data...")
    # test_loader = prepare_test_dataset(args.test_path)
    test_loader = create_image_dataloader(args.test_path, batch_size=32, transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]))

    print(f"Loading trained model from {args.model_path}")
    # Define the model and load the trained weights
    model = define_model(num_classes=2)
    model = load_model(model, args.model_path)

    print(f"Start Prediction...")
    file_paths, predictions = predict_and_save(model, test_loader, args.output_path)
    save_to_excel(file_paths, predictions,excel_file=args.output_path)
    print(f"prediction completed. Result saved at {args.output_path}")
