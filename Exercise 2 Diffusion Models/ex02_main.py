import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os

from ex02_model import Unet
from ex02_diffusion import Diffusion, cosine_beta_schedule, linear_beta_schedule, sigmoid_beta_schedule
from torchvision.utils import save_image

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--schedule', type=str, default='linear', choices=['linear', 'cosine', 'sigmoid']) # add scheduler type argument
    return parser.parse_args()


@torch.no_grad()
def sample_and_save_images(n_images, diffusor, model, device, store_path, reverse_transform):
    # TODO: Implement - adapt code and method signature as needed
    model.eval()                                    # eval mode for inference
    os.makedirs(store_path, exist_ok=True)          # make sure target directory exists

    # run full reverse diffusion process; sample() returns list of images from T-1 to 0
    imgs = diffusor.sample(model, image_size=diffusor.img_size, batch_size=n_images, channels=3)
    final_imgs = imgs[-1]                           # final denoised images at t = 0
    final_imgs = (final_imgs.clamp(-1, 1) + 1) / 2  # scale from model numeric range [-1, 1] to [0, 1] for saving

    # save final images at t=0
    save_image(final_imgs, os.path.join(store_path, "samples.png"), nrow=4)     # final_imgs shape (n_images, 3, 32, 32) arranges n_images into grdi with 4 images per row

    # save timestep progression grid for first sample (index 0) across 10 evenly spaced timesteps
    step_indices = np.linspace(0, len(imgs) - 1, num=10, dtype=int)
    progression_imgs = torch.stack([imgs[idx][0] for idx in step_indices])  # shape: (10, 3, H, W)
    progression_imgs = (progression_imgs.clamp(-1, 1) + 1) / 2              # convert tp range [0, 1]
    save_image(progression_imgs, os.path.join(store_path, "progression.png"), nrow=10)

    # GIF visualization: use reverse_transform (requires CPU) to build PIL frames for sample 0
    frames = [reverse_transform(step_img[0].cpu()) for step_img in imgs]
    frames[0].save(
        os.path.join(store_path, "generation.gif"),
        save_all=True,
        append_images=frames[1:],
        duration=50,  # 50 ms per frame
        loop=0        # loop forever
    )

@torch.no_grad()
def test(model, testloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed
    model.eval()
    timesteps = args.timesteps

    # eval fixed, evenly spaced timesteps
    eval_timesteps = [0, timesteps // 4, timesteps // 2, (3 * timesteps) // 4, timesteps - 1]    # pick timesteps at 0, 25%, 50%, 75%, 100% of total timesteps
    timestep_losses = {t_val: 0.0 for t_val in eval_timesteps}
    num_batches = 0

    for images, _ in testloader:
        images = images.to(device)                                                      # move to device (GPU)
        for t_val in eval_timesteps:           
            t = torch.full((len(images),), t_val, device=device, dtype=torch.long)      # create batch-sized tensor filled with the fixed timestep t_val
            loss = diffusor.p_losses(model, images, t, loss_type="l2")                  # compute loss for this batch at timestep t_val
            timestep_losses[t_val] += loss.item()                                       # accumulate loss for this timestep
        num_batches += 1

        if args.dry_run:
            break

    # average losses per-timestep and mean loss across all batches
    avg_losses = {t_val: total / num_batches for t_val, total in timestep_losses.items()}
    mean_loss = sum(avg_losses.values()) / len(avg_losses)

    # print eval results
    print(f"Eval Mean Loss: {mean_loss:.6f}")
    for t_val, l in avg_losses.items():
        print(f"  t={t_val}: {l:.4f}")

    return mean_loss


def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    model.train()  # ensure model is back in training mode after test()
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, loss_type="l2")

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # conditional beta schedule selection based on command-line argument
    if args.schedule == "linear":
        my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    elif args.schedule == "cosine":
        my_scheduler = lambda x: cosine_beta_schedule(x)
    elif args.schedule == "sigmoid":
        my_scheduler = lambda x: sigmoid_beta_schedule(0.0001, 0.02, x)
    else:
        raise ValueError(f"Unknown schedule type: {args.schedule}")
    
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    # set up data loaders for training, validation, and testing
    pin_memory = device == "cuda"               # use pinned memory for faster host-to-device transfer
    gen = torch.Generator().manual_seed(42)     # set seed for reproducibility of splits

    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)], generator=gen)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    # Download and load the test data
    testset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True, pin_memory=pin_memory)

    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        test(model, valloader, diffusor, device, args)

    test(model, testloader, diffusor, device, args)

    save_path = os.path.join("results", args.run_name)  # TODO: Adapt to your needs
    n_images = 8
    sample_and_save_images(n_images, diffusor, model, device, save_path, reverse_transform)
    if args.save_model:
        ckpt_dir = os.path.join("models", args.run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "ckpt.pt"))


if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)
