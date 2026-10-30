import torch
import wandb

def train(epoch, model, train_loader, criterion, optimizer, device='cuda'):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Prediction e loss
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 500 == 0:
            acc = 100.0 * correct / total
            print(
                f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] | Loss:'
                f' {loss.item():.4f} | Acc: {acc:.2f}%'
            )

            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_acc': acc,
            })

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    wandb.log({
        'train/epoch_loss': train_loss,
        'train/epoch_acc': train_acc,
        'epoch': epoch,
    })

    return train_loss, train_acc    

def validate(model, val_loader, criterion, device='cuda', epoch=1):
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')

    wandb.log({
        'val/loss': val_loss,
        'val/acc': val_accuracy,
        'epoch': epoch,
    })

    return val_accuracy