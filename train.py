import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from data.download import download_and_extract
from dataset.dataset import get_dataloaders
from models.custom_net import CustomNet
from utils.trainer import train, validate

def main():
    # Hyperparameters config and W&B setup
    config = {
        'architecture': 'CustomNet',
        'dataset': 'Tiny-ImageNet-200',
        'epochs': 10,
        'batch_size': 64,
        'learning_rate': 0.001,
        'optimizer': 'Adam',
    }

    # Start W&B run
    wandb.init(
        project='tiny-imagenet-classification', config=config
    )

    device = torch.device('cuda' if torch.cuda.is_avaible() else 'cpu')

    # Dataset download
    download_and_extract()
    # Dataloaders load
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=config['batch_size']
    )

    # Model, Loss and Optimizator
    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Gradient and Topology logs
    wandb.watch(model, criterion, log='all', log_freq=100)

    # Training loop
    epochs = 10
    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer, device=device)
        validate(model, val_loader, criterion, device=device, epoch=epoch)

        # Checkpoint save
        torch.save(
            model.state_dict(), f'checkpoints/custom_net_epoch_{epoch}.pth'
        )

    wandb.finish()


if __name__ == '__main__':
    main()