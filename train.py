import os
import time
import warnings
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import models

from customData import customData
import data_transform

warnings.filterwarnings("ignore", category=UserWarning)

# use scipy.io.loadmat to read .mat file(train)
data = scipy.io.loadmat('cars_train_annos.mat')
annotations = data['annotations']
with open('./train.txt', 'w') as f_train:
    for i in range(annotations.shape[1]):
        num = int(annotations[0, i][4])
        f_train.write(str(num) + '\n')

# use scipy.io.loadmat to read .mat file(val)
data = scipy.io.loadmat('cars_test_annos_withlabels.mat')
annotations = data['annotations']
with open('./val.txt', 'w') as f_train:
    for i in range(annotations.shape[1]):
        num = int(annotations[0, i][4])
        f_train.write(str(num) + '\n')


def train_model(model, criterion, optimizer, scheduler, num_epochs, use_gpu, only_val=True):
    """
    训练模型
    :param model: 模型
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param scheduler: 调整LR
    :param num_epochs: epochs times
    :param use_gpu: according to cuda.is_available()
    :param only_val: 是否仅验证
    :return: model
    """
    since = time.perf_counter()  # train start time
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        begin_time = time.perf_counter()
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for phase in (['val'] if only_val else ['train', 'val']):
            count_batch = 0
            if phase == 'train':
                scheduler.step()  # 调整模型中学习率的超参数
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # 迭代数据
            for data in dataloaders[phase]:
                count_batch += 1
                inputs, labels = data
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # 初始化参数梯度
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

                # print result every 10 batch
                if count_batch % 10 == 0:
                    batch_loss = running_loss / (batch_size * count_batch)
                    batch_acc = running_corrects / (batch_size * count_batch)
                    print(
                        f'Dataset:[{phase}] | Epoch:[{epoch + 1}] | Batch:[{count_batch}] | Loss:{batch_loss:.4f} | Acc:{batch_acc:.4f}'
                        f' | Time:{(time.perf_counter() - begin_time):.4f}s')
                    begin_time = time.perf_counter()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print(f'Dataset[{phase}] | Loss mean:{epoch_loss:.4f} | Acc mean:{epoch_acc:.4f}')

            # save model
            if phase == 'train':
                if not os.path.exists('output'):
                    os.makedirs('output')
                torch.save(model, f'output/resnet_epoch{epoch}.pkl')

            # save best model
            if phase == 'val' and epoch_acc > best_acc:  # 仅使用在val数据中表现最好的
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    end_time = time.perf_counter() - since
    print(f'Training complete in {end_time:.4f}s')
    # best_acc
    print(f'Best Val Acc: {best_acc:.4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    # cuda状态
    use_gpu = torch.cuda.is_available()
    print('cuda status:', use_gpu)

    # batch_size and num_class
    batch_size = 32
    num_class = 196  # 类别总数

    # return dataset class
    image_datasets = {x: customData(img_path='./' + x,
                                    label_path='./' + x + '.txt',
                                    dataset=x,
                                    data_transforms=data_transform.data_transforms) for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 获取resnet50模型,并替换fc层
    model_ft = models.resnet152(pretrained=True)

    # 迁移学习
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features, num_class)  # 将resnet50 fc features替换为类别总数

    # cuda setting
    if use_gpu:
        model_ft = model_ft.cuda()

    # 损失函数,define cost function(交叉熵损失)
    criterion = nn.CrossEntropyLoss()

    # 优化器SGD,lr=学习率/momentum=冲量
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)

    # 调整LR,每训练5个epoch,对参数*0.2
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.2)

    # train model
    model_ft = train_model(
        # model=model_ft,
        model=torch.load('Best_Resnet_152.pkl'),  # 测试已经训练好的模型用这行
        # model = ResNetModel(197,[3, 4, 6, 3]],True), # Resnet手动实现用这行
        criterion=criterion,  # 损失函数
        optimizer=optimizer_ft,  # 优化器
        scheduler=exp_lr_scheduler,  # 调整LR
        num_epochs=10,  # 训练次数
        use_gpu=True,
        only_val=True)
    # save best model
    torch.save(model_ft, "./output/Best_Resnet_152.pkl")
