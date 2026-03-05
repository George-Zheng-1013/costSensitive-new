import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

# from RSGmain.RSG import *
import gzip
import os
import csv
import time
import torchvision
import cv2
import matplotlib.pyplot as plt


BATCH_SIZE = 100
EPOCHS = 10000
LOG_INTERVAL = 50
NUM_CLASSES = 12
DATA_DIR = os.path.join("processed_full", "mnist")

# 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


learning_rate = 0.1  # 设定初始的学习率


class DealDataset(Dataset):
    """
    读取数据、初始化数据
    """

    def __init__(self, folder, data_name, label_name, transform=None):
        (train_set, train_labels) = load_data(
            folder, data_name, label_name
        )  # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)


def load_data(data_folder, data_name, label_name):
    with gzip.open(
        os.path.join(data_folder, label_name), "rb"
    ) as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        y_train = y_train.copy()  # 使numpy可写

    with gzip.open(os.path.join(data_folder, data_name), "rb") as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(
            len(y_train), 28, 28
        )
        x_train = x_train.copy()  # added
    return (x_train, y_train)


trainDataset = DealDataset(
    DATA_DIR,
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    transform=transforms.ToTensor(),
)

testDataset = DealDataset(
    DATA_DIR,
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
    transform=transforms.ToTensor(),
)

# 训练数据和测试数据的装载
train_loader = torch.utils.data.DataLoader(
    dataset=trainDataset,
    batch_size=100,  # 一个批次可以认为是一个包，每个包中含有10张图片
    shuffle=True,
)

# test数据装载
test_loader = torch.utils.data.DataLoader(
    dataset=testDataset,
    batch_size=100,  # 一个批次可以认为是一个包，每个包中含有10张图片
    shuffle=False,
)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        """self.RSG = RSG(n_center=5, feature_maps_shape=[32, 24, 24], num_classes=5, contrastive_module_dim=64,
                     head_class_lists=[0, 1, 2, 3, 4], transfer_strength=0.5, epoch_thresh=1)"""
        self.trainStatus = False
        self.epoch = 0
        self.target = []

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),  # group narmalization
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2)
        ).to(DEVICE)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        """if (self.trainStatus):
            #print(self.trainStatus)
            x, cesc_total, loss_mv_total, combine_target = self.RSG.forward(x, [0,1],
                                                                            self.target, self.epoch)"""
        """if(self.trainStatus):
            print(self.trainStatus)
            x, cesc_total, loss_mv_total, combine_target = self.RSG.forward(x, [],
                                                                            self.target, 0)"""
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self.trainStatus:
            return x  # , cesc_total, loss_mv_total, combine_target
        else:
            return x

    # 计算预测正确率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案
    def retrieve_features(
        self, x
    ):  # 该函数专门用于提取卷积神经网络的特征图的功能，返回feature_map1, feature_map2为前两层卷积层的特征图

        feature_map1 = self.layer1(x)
        feature_map2 = self.layer2(feature_map1)
        return (feature_map1, feature_map2)

    def setTrainFlag(self):
        self.trainStatus = True

    def getTarget(self, batch_target):
        self.target = batch_target

    def closeTrain(self):
        self.trainStatus = False

    def getEpoch(self, epoch):
        self.epoch = epoch


def accuracy(predictions, labels):
    # torch.max的输出：out (tuple, optional维度) – the result tuple of two output tensors (max, max_indices)
    # predictions.data = torch.unsqueeze(predictions.data,0)
    pred = torch.max(predictions.data, 1)[
        1
    ]  # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    right_num = pred.eq(
        labels.data.view_as(pred)
    ).sum()  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return right_num, len(labels)  # 返回正确的数量和这一次一共比较了多少元素


image_size = 28
num_classes = NUM_CLASSES
num_epochs = 6
batch_size = 100


def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total += target.size(0)
    return total_correct / total if total > 0 else 0.0


if __name__ == "__main__":
    net = ConvNet()
    # 如果希望使用GPU
    net = net.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-08)

    num_epochs = 6
    os.makedirs("pytorch_model", exist_ok=True)
    log_path = os.path.join("pytorch_model", "train_log.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_loss",
                "train_acc",
                "test_acc",
                "epoch_seconds",
            ]
        )

    print("=" * 88)
    print(f"开始训练 | device={DEVICE} | epochs={num_epochs} | batch_size={batch_size}")
    print(f"日志文件: {log_path}")
    print("=" * 88)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_id, (data, target) in enumerate(train_loader):
            net.train()
            data, target = data.to(DEVICE), target.to(DEVICE)
            # 如果RSG模块不可用，forward 只返回 logits
            output = net(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate accuracy (简单示例)
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            running_correct += correct
            running_total += target.size(0)
            running_loss += loss.item() * target.size(0)

            if (batch_id + 1) % LOG_INTERVAL == 0 or (batch_id + 1) == len(
                train_loader
            ):
                avg_loss = running_loss / running_total if running_total > 0 else 0.0
                avg_acc = (
                    100.0 * running_correct / running_total
                    if running_total > 0
                    else 0.0
                )
                print(
                    "[Epoch {}/{}][Batch {}/{}] loss={:.6f} acc={:.2f}%".format(
                        epoch + 1,
                        num_epochs,
                        batch_id + 1,
                        len(train_loader),
                        avg_loss,
                        avg_acc,
                    )
                )

        train_loss = running_loss / running_total if running_total > 0 else 0.0
        train_acc = running_correct / running_total if running_total > 0 else 0.0
        test_acc = evaluate(net, test_loader, DEVICE)
        lr = optimizer.param_groups[0]["lr"]
        epoch_seconds = time.time() - epoch_start

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    f"{lr:.8f}",
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{test_acc:.6f}",
                    f"{epoch_seconds:.2f}",
                ]
            )

        print(
            "[Epoch Summary {}/{}] lr={} train_loss={:.6f} train_acc={:.2f}% test_acc={:.2f}% time={:.2f}s".format(
                epoch + 1,
                num_epochs,
                f"{lr:.8f}",
                train_loss,
                train_acc * 100.0,
                test_acc * 100.0,
                epoch_seconds,
            )
        )

    # 保存 state_dict 到文件
    torch.save(net.state_dict(), "pytorch_model/convnet.pth")
    print("Saved weights -> pytorch_model/convnet.pth")

    # 可选：在训练后直接测试一次
    final_test_acc = evaluate(net, test_loader, DEVICE)
    print("Test accuracy:", final_test_acc)
    print(f"训练日志已保存 -> {log_path}")
