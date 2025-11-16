import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchgeo.datasets import LandCoverAI
import torchvision.transforms as T
import torch.nn.functional as F
import time


class PartialCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, binary_mask, full_mask):
        """
        predictions: [B, C, H, W] - model output logits
        binary_mask: [B, H, W] - binary mask (1=labeled, 0=unlabeled)
        full_mask: [B, H, W] - full ground truth mask (all class labels)

        FL = -alpha(1-pt)^gamma * log(pt)
        where:
        - alpha is the focal loss weight : default 0.25
        - gamma is the focal loss gamma: default 2.0
        - pt is the predicted probability of the true class
        """

        # convert logits to probabilities
        probs = F.softmax(predictions, dim=1)

        # Get probability of ground truth class using FULL mask
        probs_gt = probs.gather(1, full_mask.unsqueeze(1)).squeeze(1)

        # Focal loss calculation
        focal_weight = 0.25 * (1 - probs_gt) ** 2.0
        loss = -focal_weight * torch.log(probs_gt + 1e-7)

        num_labeled = torch.sum(binary_mask)
        if num_labeled > 0:
            partial_loss = torch.sum(loss * binary_mask) / num_labeled
        else:
            partial_loss = torch.tensor(0.0, device=predictions.device)

        return partial_loss


def generate_point_labels(masks, num_points=100):
    """
    Generate sparse point labels and binary mask
    Returns:
        point_labels: class labels at sampled points (0-4)
        binary_mask: binary mask (1 where points are sampled, 0 elsewhere)
    """
    point_labels = torch.zeros_like(masks)
    binary_mask = torch.zeros_like(masks, dtype=torch.float)

    for i in range(len(masks)):
        mask = masks[i]
        h, w = mask.shape

        # Sample from ALL pixels (including background)
        all_indices = torch.arange(h * w, device=mask.device)
        sampled_indices = all_indices[torch.randperm(len(all_indices))[
            :num_points]]

        # Convert flat indices to 2D coordinates
        sampled_y = sampled_indices // w
        sampled_x = sampled_indices % w

        # Set point labels (class values)
        point_labels[i, sampled_y, sampled_x] = mask[sampled_y, sampled_x]
        # Set binary mask (1 = labeled pixel)
        binary_mask[i, sampled_y, sampled_x] = 1.0

    return point_labels, binary_mask


def get_model(device, is_pretrained=True):
    if is_pretrained:
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    else:
        model = deeplabv3_resnet50()

    # Modify classifier for 5 classes
    model.classifier[4] = nn.Conv2d(256, 5, kernel_size=1)

    # Modify aux_classifier only if it exists (it's None for non-pretrained models)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 5, kernel_size=1)

    model = model.to(device)
    return model


def train_model(model, train_loader, device, num_epochs=5, learning_rate=0.001):
    """
    Train a model and return final validation loss and training time
    """
    criterion = PartialCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device).long()

            # Generate binary mask indicating which pixels are labeled
            _, binary_mask = generate_point_labels(masks)
            binary_mask = binary_mask.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']

            loss = criterion(outputs, binary_mask, masks)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Epoch Percent completed: {batch_idx/len(train_loader):.2f}")
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

    training_time = time.time() - start_time
    final_loss = avg_loss

    return final_loss, training_time


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset slicing configuration (set to None to use full dataset)
    # Use only small portion of the data otherwise it will take too long to train
    data_slice_percentage = 0.01

    # Load dataset
    print("Loading LandCoverAI dataset...")
    train_dataset = LandCoverAI(root='./data', split='train', download=True)

    # Slice dataset if configured
    if data_slice_percentage is not None and data_slice_percentage < 1.0:
        dataset_size = len(train_dataset)
        slice_size = int(dataset_size * data_slice_percentage)
        train_dataset = Subset(train_dataset, range(slice_size))
        print(f"Using {slice_size}/{dataset_size} samples "
              f"({data_slice_percentage*100:.1f}% of dataset)")

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])

        # Normalize images (assumes images are in [0, 1] range from TorchGeo)
        # If images are in [0, 255], divide by 255 first
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if images.max() > 1.0:
            images = images / 255.0
        images = normalize(images)

        return {'image': images, 'mask': masks}

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Define experiments
    experiments = [
        {"name": "Pretrained vs. Non-Pretrained", "pretrained": [True, False]},
        {"name": "Learning Rate Effect",
            "learning_rate": [0.01, 0.001]},
        {"name": "Epoch Count", "epochs": [5, 10]}
    ]

    # Default parameters
    default_params = {
        'pretrained': True,
        'learning_rate': 0.001,
        'epochs': 5
    }

    # Run experiments
    all_results = []
    report_lines = []

    for experiment in experiments:
        exp_name = experiment['name']
        print(f"\n{'='*60}")
        print(f"Running Experiment: {exp_name}")
        print(f"{'='*60}\n")

        report_lines.append(f"\nExperiment: {exp_name}\n")

        # Get the parameter being varied
        param_name = [k for k in experiment.keys() if k != 'name'][0]
        param_values = experiment[param_name]

        for param_value in param_values:
            # Set up parameters for this run
            params = default_params.copy()
            params[param_name] = param_value

            print(f"  Testing {param_name}: {param_value}")

            # Create model
            model = get_model(device, is_pretrained=params['pretrained'])

            # Train model
            final_loss, training_time = train_model(
                model, train_loader, device,
                num_epochs=params['epochs'],
                learning_rate=params['learning_rate']
            )

            # Add to report
            report_lines.append(f"  - {param_name}: {param_value}\n")
            report_lines.append(f"    * Training Completed. Time Taken: {training_time:.2f}s, "
                                f"Validation Loss: {final_loss:.4f}\n")

            print(
                f"    ✓ Completed in {training_time:.2f}s, Loss: {final_loss:.4f}\n")

    # Generate text report
    report_text = ''.join(report_lines)
    print(f"\n{'='*60}")
    print("EXPERIMENT REPORT")
    print(f"{'='*60}")
    print(report_text)

    # Save report to file
    with open('experiment_report.txt', 'w') as f:
        f.write("EXPERIMENT REPORT\n")
        f.write("="*60 + "\n")
        f.write(report_text)
    print("\nReport saved as 'experiment_report.txt'")

    print("\n✓ All experiments completed!")


if __name__ == '__main__':
    main()
