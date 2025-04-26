"""
main.py — Сравнение моделей для бинарной классификации изображений (PyTorch)

- SimpleCNN, AlexNet (transfer learning), ResNet18 (transfer learning)
- Визуализация accuracy/loss
- Сохранение лучших моделей
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

# === 0. Установка SEED для воспроизводимости ===
torch.manual_seed(123)
random.seed(123)
np.random.seed(123)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === 1. Устройство ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. Преобразования и загрузка данных ===
# обработка изображений
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])
# Загрузка картинок
train_data = datasets.ImageFolder('data/train', transform=transform)
val_data = datasets.ImageFolder('data/val', transform=transform)
test_data = datasets.ImageFolder('data/test', transform=transform)
# Разбиваем на пачки
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# === 3. Модели ===
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), # "Разворачиваем" 3D-тензор в 1D-вектор
            nn.Linear(128 * 16 * 16, 128), nn.ReLU(), # 128*16*16 входов → 128 выходов
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)


# Улучшенная AlexNet с частичным размораживанием
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        # Замораживаем все слои, кроме последних
        for param in self.model.parameters():
            param.requires_grad = False
        # Размораживаем последние слои
        for param in self.model.features[-4:].parameters():
            param.requires_grad = True
        self.model.classifier[6] = nn.Linear(4096, 2)

    def forward(self, x):
        return self.model(x)


def create_resnet():
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False  # Freeze everything

    model.fc = nn.Linear(model.fc.in_features, 2)  # Only fc will train
    return model



# === 4. Тренировка и оценка (Функции для обучения и оценки) ===
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, criterion=None):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion:
                total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    loss = total_loss / len(loader) if criterion else None
    return acc, loss


# === 5. Эксперименты ===
def run_experiment(model, name, epochs=5):
    print(f"\n--- {name} ---")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    train_accs, val_accs, train_losses, val_losses = [], [], [], []
    best_val_acc = 0
    best_state_dict = None

    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        train_acc, _ = evaluate(model, train_loader)
        val_acc, val_loss = evaluate(model, val_loader, criterion)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    test_acc, _ = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")

    return {
        "name": name,
        "test_acc": test_acc,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "model_state_dict": best_state_dict
    }


# === 6. Запуск всех моделей ===
experiments = [
    run_experiment(SimpleCNN(), "Simple CNN"),
    run_experiment(AlexNet(), "AlexNet"),
    run_experiment(create_resnet(), "ResNet18 (Transfer)")
]

# === 7. Общий вывод результатов ===
print("\nFinal Results:")
for exp in experiments:
    print(f"{exp['name']}: {exp['test_acc']:.2f}%")

# === 8. Сохранение лучших моделей ===
if not os.path.exists('results'):
    os.makedirs('results')

for exp, model_name in zip(experiments, ["Simple CNN", "AlexNet", "ResNet18 (Transfer)"]):
    # Сохраняем веса только если accuracy > 80 (пример)
    if exp['test_acc'] > 80:
        torch.save(exp.get('model_state_dict', {}), f"results/best_{model_name.replace(' ', '_')}.pth")

# === 9. Графики всех моделей ===
plt.figure(figsize=(14, 6))

# Accuracy
plt.subplot(1, 2, 1)
for exp in experiments:
    plt.plot(exp['train_accs'], label=f"{exp['name']} - Train")
    plt.plot(exp['val_accs'], linestyle='--', label=f"{exp['name']} - Val")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
for exp in experiments:
    plt.plot(exp['train_losses'], label=f"{exp['name']} - Train")
    plt.plot(exp['val_losses'], linestyle='--', label=f"{exp['name']} - Val")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("results/All_Models_Metrics.png")
plt.show()
