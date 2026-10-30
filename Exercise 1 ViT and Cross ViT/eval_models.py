import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Import your ViT class definition from your model skeleton file
from my_models_skeleton import ViT

def evaluate_models(models_dir='./models', data_dir='./cifar10', batch_size=64):
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Standard CIFAR-10 test data transformations (matching training normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Load CIFAR-10 test dataset
    testset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # CIFAR-10 Class labels
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    # Check if models directory exists
    if not os.path.exists(models_dir):
        print(f"Directory '{models_dir}' not found. Please create it and place your saved model files there.")
        return

    # Find all PyTorch model files in the models folder
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.pt', '.pth'))]

    if not model_files:
        print(f"No model checkpoint files (.pt or .pth) found in '{models_dir}'.")
        return

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        print("=" * 60)
        print(f"Evaluating model: {model_file}")
        print("=" * 60)

        try:
            # 1. Instantiate the correct architecture based on filename
            if 'r18' in model_file.lower():
                model = models.resnet18(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
            elif 'vit' in model_file.lower():
                model = ViT(image_size=32, patch_size=8, num_classes=10, dim=64,
                            depth=2, heads=8, mlp_dim=128, dropout=0.1,
                            emb_dropout=0.1)
            else:
                print(f"Unknown model architecture for file: {model_file}. Skipping.")
                continue

            # 2. Load the state dictionary weights into the model
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)

            # 3. Move model to device and set to evaluation mode
            model.to(device)
            model.eval()

            test_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    # Accumulate loss
                    loss = criterion(output, target)
                    test_loss += loss.item() * data.size(0)

                    # Get predictions
                    preds = output.argmax(dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

            # Compute metrics
            total_loss = test_loss / len(testloader.dataset)
            accuracy = accuracy_score(all_targets, all_preds) * 100

            print(f"Average Test Loss : {total_loss:.4f}")
            print(f"Test Accuracy     : {accuracy:.2f}%\n")

            print("Detailed Classification Report:")
            print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

        except Exception as e:
            print(f"Error loading or evaluating {model_file}: {e}\n")


if __name__ == '__main__':
    evaluate_models()