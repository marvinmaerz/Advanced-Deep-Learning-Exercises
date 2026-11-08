import argparse
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim

from my_models import ViT, CrossViT   # rename the skeleton file for your implementation / comment before testing for ResNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to classify CIFAR10')
    parser.add_argument('--model', type=str, default='r18', help='model to train (default: r18)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'], help='optimizer (default: sgd)')
    parser.add_argument('--data-dir', type=str, default='./data', help='folder with the CIFAR10 data (default: ./data)')
    parser.add_argument('--download', action='store_true', default=False, help='download CIFAR10 if not present')
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

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        set, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / len(test_loader.dataset)


def run(args):
    torch.manual_seed(args.seed)   # reproducible train/val split and init

    # Download and load the training data
    transform = transforms.Compose([
                                    transforms.RandomCrop(32,padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    # ImageNet mean/std values should also fit okayish for CIFAR
									transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ),
                                    ])

    transform_test = transforms.Compose([
                                transforms.ToTensor(),
                                # ImageNet mean/std values should also fit okayish for CIFAR
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


    dataset = datasets.CIFAR10(args.data_dir, download=args.download, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)])
    # validation must not be augmented -> same images, but test transform
    valset.dataset = datasets.CIFAR10(args.data_dir, download=False, train=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10(args.data_dir, download=args.download, train=False, transform=transform_test)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # Build a feed-forward network
    print(f"Using {args.model}")
    if args.model == "r18":
        model = models.resnet18(weights=None, num_classes=10)   # 10 CIFAR classes instead of 1000
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
    else:
        raise ValueError(f"unknown model '{args.model}', use r18, vit or cvit")

    # Define the loss
    criterion = nn.CrossEntropyLoss(reduction="sum")
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    ckpt_path = f"best_{args.model}.pt"   # own checkpoint per model
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train(model, trainloader, optimizer, criterion, device, epoch)
        val_acc = test(model, device, valloader, criterion, set="Validation")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"New best validation accuracy {best_acc:.4f} -> saved to {ckpt_path}")

    # evaluate the best (not the last) model on the test set
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test(model, device, testloader, criterion)

if __name__ == '__main__':
    args = parse_args()
    run(args)