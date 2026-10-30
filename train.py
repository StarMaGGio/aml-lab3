import torch
import torch.nn as nn
import torch.optim as optim
from data.download import download_and_extract
from dataset.dataset import get_dataloaders
from models.custom_net import CustomNet
from utils.trainer import train, validate

def main():
    device = torch.device('cuda' if torch.cuda.is_avaible() else 'cpu')

    # Dataset download
    download_and_extract()
    # Dataloaders load
    train_loader, val_loader, _, _ = get_dataloaders()

    # Model, Loss and Optimizator
    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer, device=device)
        validate(model, val_loader, criterion, device=device)

        # Checkpoint save
        torch.save(
            model.state_dict(), f'checkpoints/custom_net_epoch_{epoch}.pth'
        )


if __name__ == '__main__':
    main()