import scipy.io
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# 此文件用于单图片预测测试
transform1 = transforms.Compose([
    # 600*400 → 256*256
    transforms.Resize((256, 256)),
    # 随机裁剪 → 224*224
    transforms.RandomCrop(224),
    # # 中心裁剪 → 224*224
    # transforms.CenterCrop(224),
    # (H, W, C) → (C, H, W) 且归一化
    transforms.ToTensor(),
    # 将数据转换为标准正太分布,使模型更容易收敛
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data = scipy.io.loadmat('cars_meta.mat')
model = torch.load('Best_Resnet_152.pkl')
test = Image.open('./val/00001.jpg').convert('RGB')
test = transform1(test)
test = test.reshape(1, 3, 224, 224)
inputs = Variable(test.cuda())
outputs = model(inputs)
_, preds = torch.max(outputs.data, 1)
preds = (preds.cpu().numpy().tolist())[0]
print(data['class_names'][0][preds - 1], torch.max(F.softmax(outputs)).item())
