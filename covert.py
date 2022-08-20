import os
from PIL import Image
if __name__ == "__main__":
    rootdir = "dataset"

    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        image=Image.open(path)
        out = image.convert("RGB")
        out.save(path)
        # if os.path.isfile(path):
        # # 你想对文件的操作
        print(path)
