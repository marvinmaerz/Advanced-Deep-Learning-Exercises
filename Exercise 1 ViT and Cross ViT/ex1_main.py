import argparse
import torch
import copy
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch import optim

from my_models import ViT, CrossViT   # rename the skeleton file for your implementation / comment before testing for ResNet
from model_utils import save_model_ckpt, plot_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify CIFAR10')
    parser.add_argument('--model', type=str, default='r18', help='model to train (default: r18; options: r18, vit, cvit)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--save-name', type=str, help="Optional name suffix for the saved checkpoint name.")
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()

def train(model, trainloader, optimizer, criterion, device, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)/len(output)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, criterion, set="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        accuracy * 100.))

    return accuracy


def run(args):
    # Download and load the training data
    # ImageNet mean/std values should also fit okayish for CIFAR
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(size=32, padding=4, padding_mode="reflect"),
        # transforms.Pad(2, padding_mode="reflect"),  # needed to get back to 32x32 px for ViT and CViT
        transforms.RandomHorizontalFlip(p=0.5),
        normalize,
    ])
    # deterministic transforms for evaluation (nothing randomly applied)
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    ds_root_path = "C:/dev/adl/ds-cifar10"
    train_full = datasets.CIFAR10(ds_root_path, download=True, train=True, transform=train_transform)
    val_full = datasets.CIFAR10(ds_root_path, download=True, train=True, transform=eval_transform)
    # compute indices of shared permutation between train/val datasets with different transforms
    n = len(train_full)
    n_train = int(n * 0.9)  # keeping the same 0.9/0.1 ratio
    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=generator).tolist()
    train_idx, val_idx = perm[ : n_train], perm[n_train : ]

    trainset = Subset(train_full, train_idx)
    valset = Subset(val_full, val_idx)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    # # Download and load the test data
    testset = datasets.CIFAR10(ds_root_path, download=True, train=False, transform=eval_transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=True)

    # Build a feed-forward network
    print(f"Using {args.model}")
    if args.model == "r18":
        model = models.resnet18(pretrained=False)
    elif args.model == "vit":
        model = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64,
                    depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1,
                    emb_dropout = 0.1) 
    elif args.model == "cvit":
        model = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, 
                         lg_dim = 128, sm_patch_size = 8, sm_enc_depth = 2,
                         sm_enc_heads = 8, sm_enc_mlp_dim = 128, 
                         sm_enc_dim_head = 64, lg_patch_size = 16, 
                         lg_enc_depth = 2, lg_enc_heads = 8, 
                         lg_enc_mlp_dim = 128, lg_enc_dim_head = 64,
                         cross_attn_depth = 2, cross_attn_heads = 8,
                         cross_attn_dim_head = 64, depth = 3, dropout = 0.1,
                         emb_dropout = 0.1)

    # Define the loss
    criterion = nn.CrossEntropyLoss(reduction="sum")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    accuracy_max = -1.0
    accuracy_vals = []
    best_model_ckpt: dict = {}

    for epoch in range(1, args.epochs + 1):
        train(model, trainloader, optimizer, criterion, device, epoch)
        accuracy = test(model, device, valloader, criterion, set="Validation")
        accuracy_vals.append(accuracy)
        if accuracy > accuracy_max:
            accuracy_max = accuracy
            # Save whole checkpoint to possibly resume training
            print("Caching current best model w.r.t. accuracy.\n")
            best_model_ckpt = {
                "model": args.model,
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "accuracy": accuracy,
            }

    test(model, device, testloader, criterion)

    plot_accuracy(accuracy_vals, accuracy_max, args.model, args.epochs)

    if args.save_model:
        save_model_ckpt(best_model_ckpt, args.save_name)

if __name__ == '__main__':
    args = parse_args()
    run(args)
