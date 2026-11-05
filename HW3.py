import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
def generate_sample(shape_type, image_size=128):

    image = np.random.randint(
        0, 46,
        (image_size, image_size)
    ).astype(np.uint8)

    mask = np.zeros(
        (image_size, image_size),
        dtype=np.uint8
    )

    
    intensity = random.randint(200, 255)

    if shape_type == "circle":

        radius = random.randint(10, 25)

        cx = random.randint(radius, image_size - radius - 1)
        cy = random.randint(radius, image_size - radius - 1)

        cv2.circle(
            image,
            (cx, cy),
            radius,
            intensity,
            -1
        )

        cv2.circle(
            mask,
            (cx, cy),
            radius,
            1,
            -1
        )

    elif shape_type == "square":

        half = random.randint(12, 25)

        cx = random.randint(half, image_size - half - 1)
        cy = random.randint(half, image_size - half - 1)

        angle = random.uniform(0, 360)

        rect = (
            (cx, cy),
            (2 * half, 2 * half),
            angle
        )

        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.fillPoly(
            image,
            [box],
            intensity
        )

        cv2.fillPoly(
            mask,
            [box],
            1
        )

    elif shape_type == "triangle":


        cx = random.randint(30, image_size - 31)
        cy = random.randint(30, image_size - 31)

        size = random.randint(15, 30)

        points = np.array([
            [cx, cy - size],
            [cx - size, cy + size],
            [cx + size, cy + size]
        ], dtype=np.float32)

        angle = np.deg2rad(random.uniform(0, 360))

        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])

        points = points - np.array([cx, cy])
        points = points @ rotation.T
        points = points + np.array([cx, cy])

        points = np.int32(points)

        cv2.fillPoly(
            image,
            [points],
            intensity
        )

        cv2.fillPoly(
            mask,
            [points],
            1
        )

    return image, mask


class ShapeDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

        self.images = []
        self.masks = []
        self.labels = []

        shapes = ["circle", "square", "triangle"]

        for i in range(num_samples):
            shape_type = random.choice(shapes)

            image, mask = generate_sample(shape_type)

            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32)

            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)

            self.images.append(image)
            self.masks.append(mask)
            self.labels.append(shape_type)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.tensor(
            self.images[idx],
            dtype=torch.float32
        )

        mask = torch.tensor(
            self.masks[idx],
            dtype=torch.float32
        )

        return image, mask


train_dataset = ShapeDataset(1000)
test_dataset = ShapeDataset(200)

print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))



batch_size = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Downsampling block 1
        self.enc1 = ConvBlock(1, 16)
        self.pool1 = nn.MaxPool2d(2)

        # Downsampling block 2
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(32, 64)

        # Upsampling block 1
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Upsampling block 2
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32, 16)

        # Final output
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u1 = self.up1(b)
        u1 = torch.cat((u1, e2), dim=1)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat((u2, e1), dim=1)
        d2 = self.dec2(u2)

        output = self.output(d2)

        return output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Number of epochs
num_epochs = 15

# Store loss for learning curve
loss_history = []

for epoch in range(num_epochs):

    model.train()
    epoch_loss = 0.0

    for images, masks in train_loader:

        # Move data to GPU
        images = images.to(device)
        masks = masks.to(device)

        # Clear old gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Save loss
        loss_history.append(loss.item())

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}"
    )

# Save trained model

torch.save(model.state_dict(), "unet_model.pth")
print("Model saved!")

# Evaluate on Test Set
model.eval()

TP = 0
TN = 0
FP = 0
FN = 0

with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        # Convert logits to probabilities
        probabilities = torch.sigmoid(outputs)

        # Convert probabilities to binary predictions
        predictions = (probabilities > 0.5).float()

        # Count TP, TN, FP, FN
        TP += ((predictions == 1) & (masks == 1)).sum().item()
        TN += ((predictions == 0) & (masks == 0)).sum().item()
        FP += ((predictions == 1) & (masks == 0)).sum().item()
        FN += ((predictions == 0) & (masks == 1)).sum().item()


# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# IoU
iou = TP / (TP + FP + FN)

print("\nTest Results")
print("--------------------")
print("TP:", TP)
print("TN:", TN)
print("FP:", FP)
print("FN:", FN)
print(f"Accuracy: {accuracy:.4f}")
print(f"IoU: {iou:.4f}")


# Plot Learning Curve
plt.figure(figsize=(7, 5))

plt.plot(loss_history)

plt.xlabel("Training Updates")
plt.ylabel("BCE Loss")
plt.title("Training Loss Curve")

plt.grid(True)
plt.tight_layout()

plt.savefig("learning_curve.png", dpi=300)
plt.show()


# Visualize 6 Test Samples
model.eval()

selected_indices = []

for shape in ["circle", "square", "triangle"]:

    indices = [
        i for i, label in enumerate(test_dataset.labels)
        if label == shape
    ]

    selected_indices.extend(
        random.sample(indices, 2)
    )


fig, axes = plt.subplots(3, 6, figsize=(14, 6))

for col, idx in enumerate(selected_indices):

    image, mask = test_dataset[idx]
    input_image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        probability = torch.sigmoid(output)
        prediction = (probability > 0.5).float()

    image_np = image.squeeze().numpy()
    mask_np = mask.squeeze().numpy()
    prediction_np = prediction.squeeze().cpu().numpy()

    shape_name = test_dataset.labels[idx]

    axes[0, col].imshow(image_np, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(shape_name.capitalize(), fontsize=10)

    axes[1, col].imshow(mask_np, cmap="gray", vmin=0, vmax=1)

    axes[2, col].imshow(prediction_np, cmap="gray", vmin=0, vmax=1)

    for row in range(3):
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])

axes[0, 0].set_ylabel("Original", fontsize=10)
axes[1, 0].set_ylabel("Ground Truth", fontsize=10)
axes[2, 0].set_ylabel("Prediction", fontsize=10)

plt.tight_layout()

plt.savefig(
    "segmentation_results_compact.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()