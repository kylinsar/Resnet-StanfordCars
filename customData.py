import os
from torch.utils.data import Dataset
from tools import default_loader


# 以Dataset为基类,重载__init__/__len__/__getitem__,构造数据集
# __len__为尺寸/__getitem__为索引获取标签和数据
class customData(Dataset):
    def __init__(self, img_path, label_path, dataset='', data_transforms=None, loader=default_loader):
        """
        图片路径存入self.img_name数组,对应标签写入self.img_label数组
        :param img_path: image path
        :param label_path: label path
        :param dataset: dataset name
        :param data_transforms: data transforms
        :param loader: image loader
        """
        with open(label_path) as label:
            # 将label的每一行作为一项读入到list-lines
            lines = label.readlines()
            self.img_label = [int(line.strip()) for line in lines]  # line.strip()删除首位空格
        self.img_name = []
        # os.walk遍历目录,root目录地址/dirs目录名数组/files文件名数组
        for root, dirs, files in os.walk(img_path):
            for name in sorted(files):
                self.img_name.append(os.path.join(img_path, name))
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        """

        :return: return dataset size
        """
        return len(self.img_name)

    def __getitem__(self, item):
        """
        迭代器
        :param item:
        :return: return image and label
        """
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print(f"Can't transform images: {img_name}")
        return img, label