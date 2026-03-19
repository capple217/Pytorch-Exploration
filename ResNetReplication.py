import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Model Architecture

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.rel = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.rel(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.rel(out)

        return out

class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.rel = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.rel(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.rel(out)
        return out

class Net(nn.Module):
    def __init__(self, block_class, n):
        super().__init__()
        self.block_class = block_class
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.g1 = self._make_group(16, 16, n, stride=1)
        self.g2 = self._make_group(16, 32, n, stride=2)
        self.g3 = self._make_group(32, 64, n, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def _make_group(self, in_channels, out_channels, n_blocks, stride):
        blocks = [self.block_class(in_channels, out_channels, stride)]
        for _ in range(n_blocks - 1):
            blocks.append(self.block_class(out_channels, out_channels, 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        stem = self.stem(x)
        g1 = self.g1(stem)
        g2 = self.g2(g1)
        g3 = self.g3(g2)
        pool = self.pool(g3)
        x = pool.view(pool.size(0), -1)
        return self.fc(x)

# Kaiming initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

# Training function
def train(model, train_loader, test_loader, device, epochs=182):
    model.apply(init_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[91, 136], gamma=0.1)

    accuracies = []
    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            loss = loss_fn(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    out = model(images)
                    _, predicted = torch.max(out, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            accuracies.append(acc)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

    return losses, accuracies

# Data loading
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Train ResNet-20
print("\n--- Training ResNet-20 ---")
resnet = Net(ResidualBlock, n=3).to(device)
resnet_losses, resnet_accuracies = train(resnet, train_loader, test_loader, device)

# Train PlainNet-20
print("\n--- Training PlainNet-20 ---")
plainnet = Net(PlainBlock, n=3).to(device)
plain_losses, plain_accuracies = train(plainnet, train_loader, test_loader, device)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(resnet_losses, label="ResNet-20")
ax1.plot(plain_losses, label="PlainNet-20")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

epochs_eval = list(range(10, 183, 10))
ax2.plot(epochs_eval, resnet_accuracies, label="ResNet-20")
ax2.plot(epochs_eval, plain_accuracies, label="PlainNet-20")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()

plt.tight_layout()
plt.savefig("resnet_vs_plain.png")
plt.show()
