from data.download import download_and_extract
from dataset.dataset import get_dataloaders
from utils.visualization import visualize_samples

def main():
    # Dataset download
    download_and_extract()

    # Dataloaders load
    train_loader, test_loader, train_dataset = get_dataloaders()


    # Print dataset info
    num_classes = len(train_dataset.classes)
    num_samples = len(train_dataset)
    print(f'Number of classes: {num_classes}')
    print(f'Number of samples: {num_samples}')

    # Visualize dataset samples
    visualize_samples(train_loader, train_dataset)


if __name__ == '__main__':
    main()