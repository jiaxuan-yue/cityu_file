import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import copy

# ------------------------------
# 配置参数
# ------------------------------
data_dir = 'data/cub200/images'
batch_size = 32
num_epochs = 100
lr = 1e-4
save_dir = 'checkpoints'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ratio = 0.8
pretrained_path = './model/resnet101.pth'
early_stop_patience = 7  # 早停轮数
start_saving_epoch = 30  # 最佳模型开始保存轮次
os.makedirs(save_dir, exist_ok=True)

# ------------------------------
# 数据增强与加载
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(train_ratio * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ------------------------------
# 模型定义：ResNet101 + Dropout
# ------------------------------
model = models.resnet101(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, len(full_dataset.classes))
)
checkpoint = torch.load(pretrained_path, map_location=device)
model_dict = model.state_dict()

# 过滤掉 fc 层
pretrained_dict = {k: v for k, v in checkpoint.items() if k not in ['fc.weight', 'fc.bias']}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

# ------------------------------
# 辅助函数：保存loss+acc曲线
# ------------------------------
def save_metrics_curve(train_losses, val_losses, train_accs, val_accs, path):
    plt.figure(figsize=(12, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ------------------------------
# 训练循环，加入早停与实时绘图
# ------------------------------
train_losses, val_losses = [], []
train_accs, val_accs = [], []
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
early_stop_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_acc = correct / total
    train_losses.append(epoch_train_loss)
    train_accs.append(epoch_train_acc)

    # 验证
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    epoch_val_acc = val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accs.append(epoch_val_acc)

    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | '
          f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')

    # 保存最佳模型（30轮后才开始）
    if epoch + 1 >= start_saving_epoch:
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'best_model.pth'))
            print("Best model updated!")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 每轮保存训练曲线（实时更新）
    save_metrics_curve(train_losses, val_losses, train_accs, val_accs,
                       os.path.join(save_dir, 'train_val_metrics.png'))

print("训练完成，所有曲线和最佳模型已保存。")
