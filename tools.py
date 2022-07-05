from PIL import Image


# 将RGBA图像convert为RGB图像
def default_loader(path):
    img = Image.open(path)
    return img.convert('RGB')
