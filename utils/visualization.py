import matplotlib.pyplot as plt
import numpy as np

# Function to denormalize image for visualization
def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    return np.clip(image, 0, 1)

def visualize_samples(dataloader, dataset, num_classes_to_show=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    classes_sampled = []
    found_classes = 0

    for inputs, targets in dataloader:
        for j in range(inputs.shape[0]):
            class_idx = targets[j].item()
            if class_idx not in classes_sampled:
                img = denormalize(inputs[j])
                ax = axes[found_classes]
                ax.imshow(img)
                ax.set_title(dataset.classes[class_idx])
                ax.axis('off')

                classes_sampled.append(class_idx)
                found_classes += 1
                if found_classes == num_classes_to_show:
                    break
        if found_classes == num_classes_to_show:
            break

    plt.tight_layout()
    plt.show()