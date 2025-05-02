import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler

CHECKPOINT_PATH = "checkpoint_resnet.pth"

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, master_addr, master_port):
    setup(rank, world_size, master_addr, master_port)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = models.resnet18(num_classes=10).to(device)
    ddp_model = DDP(model, device_ids=[0] if device.type == "cuda" else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters())

    start_epoch = 0
    start_iter = 0

    if os.path.exists(CHECKPOINT_PATH):
        print(f"[Rank {rank}] Loading checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        ddp_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if rank == 0:
            start_iter = checkpoint.get('iteration', 0)
            print(f"[Rank {rank}] Resumed from epoch {start_epoch}, iteration {start_iter}")
    else:
        print(f"[Rank {rank}] No checkpoint found. Starting from scratch.")

    # Broadcast iteration across all nodes
    start_iter_tensor = torch.tensor([start_iter], device=device)
    dist.broadcast(start_iter_tensor, src=0)
    start_iter = start_iter_tensor.item()

    start_time = time.time()
    iterations = []
    timestamps = []

    for epoch in range(start_epoch, start_epoch + 1):  # One epoch for demo
        for i, (x, y) in enumerate(dataloader):
            if i < start_iter:
                continue

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = ddp_model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                now = time.time() - start_time
                iterations.append(i)
                timestamps.append(now)
                print(f"[Rank {rank}] Epoch {epoch}, Iteration {i}, Time Elapsed: {now:.2f}s")

            if rank == 0 and i % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'iteration': i,
                    'model_state_dict': ddp_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, CHECKPOINT_PATH)
                print(f"[Rank {rank}] Checkpoint saved at iteration {i}")

    if rank == 0:
        import matplotlib.pyplot as plt
        with open("training_log_resnet.json", "a") as f:
            json.dump({"iterations": iterations, "timestamps": timestamps}, f)
            f.write("\n")
        plt.plot(timestamps, iterations)
        plt.xlabel("Time (s)")
        plt.ylabel("Iterations")
        plt.title("Training Progress (ResNet-18 + CIFAR-10)")
        plt.savefig("training_progress_resnet.png")
        print("Plot saved as training_progress_resnet.png")

    cleanup()

if __name__ == "__main__":
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    master_addr = sys.argv[3]
    master_port = sys.argv[4]
    train(rank, world_size, master_addr, master_port)
