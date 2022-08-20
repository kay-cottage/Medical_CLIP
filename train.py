# -*- coding: utf-8 -*-
"""
Created on Wed Aug.17 09:12:41 2022
@author: gw.kayak
"""
#https://github.com/openai/CLIP/issues/83
from torch.utils.data import Dataset, DataLoader
import torch
import clip
from torch import nn, optim
import pandas as pd
from PIL import Image
from model import Net
from my_dataset import image_caption_dataset
from sentence_transformers import SentenceTransformer,util
from Backbone.shufflenet.nets.model import shufflenet_v2_x1_0
import numpy as np
from torchvision import transforms, datasets, utils



unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image






def VIT_Encoder():
    with torch.no_grad():
        VIT_encoder=SentenceTransformer(r"E:\2022_Pro\code\model\clip-ViT-B-32").encode
        print('loaded VIT_encoder successfully')
        return VIT_encoder
    

def text_encoder():
    with torch.no_grad():
        text_encoder=SentenceTransformer(r"E:\2022_Pro\code\model\clip-ViT-B-32-multilingual-v1").encode
        print('loaded text_encoder successfully')
        return text_encoder



    
BATCH_SIZE = 9
EPOCH = 360


'''
df = pd.DataFrame({"image":[r"dataset\3.png",
                            r"dataset\R-C.jpg",
                            r"dataset\1.jpg",
                            r"dataset\a.jpg",
                            r"dataset\0.png",
                            r"dataset\nn.png",
                            r"dataset\n.png",
                            r"dataset\e.png",
                            r"dataset\ee.png"],

                   "caption":["the picture of neutrophil and red blood cells,two type of cells.The neutrophil with nucleus is surrounded by many red blood cells in the center of this picture is the biggest cell.Neutrophils are round,and blue-purple.",
                              "the picture of eosinophil and red blood cells,two type of cells.The eosinophil with nucleus is in the center of this picture and others are red blood cells", 
                              "the picture of eosinophil and red blood cells,two type of cells.Eosinophils with nucleus are often bigger than red blood cells,rounded and red-purple, around the eosinophil are red blood cells",
                              "the picture of neutrophils and red blood cells,two type of cells.There are six neutrophils with nucleus in this picture and other cells are red blood cells.",
                              "the picture of red blood cell,a type of cells,it is red and small and without nucleus.",
                              "the picture of nucleus of eosinophil,a type of structure of cell,it is purple-blue.",
                              "the picture of nucleus of eosinophil,a type of structure of cell,it is purple-blue.",
                              "the picture of nucleus of neutrophil,a type of structure of cell,it is grey-blue.",
                              "the picture of nucleus of neutrophil,a type of structure of cell,it is grey-blue."]})

'''

'''
df = pd.DataFrame({"image":[r"dataset\0.jpg",
                            r"dataset\1.jpg",
                            r"dataset\2.jpg",
                            r"dataset\3.jpg",
                            r"dataset\4.jpg",
                            r"dataset\5.jpg",
                            r"dataset\6.jpg",
                            r"dataset\7.jpg",
                            r"dataset\8.jpg",
                            
                            ],

                   "caption":["两只黑色的藏獒犬趴着",
                              "一只白色的藏獒犬趴着",
                              "狗主人跟他的黑色藏獒犬一起站着",
                              "女孩跟金毛犬坐在湖边看日落",
                              "一只金毛犬站在自行车旁边",
                              "4只金毛在玩耍",
                              "一只柯基犬站在雪地上",
                              "两只柯基犬在草地上玩耍",
                              "一条柯基躺在草地上睡觉"
               ]})

'''
df = pd.DataFrame({"image":[r"dataset\0.jpg",
                            r"dataset\1.jpg",
                            r"dataset\2.jpg",
                            r"dataset\3.jpg",
                            r"dataset\4.jpg",
                            r"dataset\5.jpg",
                            r"dataset\6.jpg",
                            r"dataset\7.jpg",
                            r"dataset\8.jpg",
                            
                            ],

                   "caption":["两只黑色藏獒犬趴在围栏上",
                              "一只白色藏獒犬趴在蓝天下",
                              "一个黑衣男人背着一只黑色藏獒犬一旁跟着一个灰色衣服的男人",
                              "女孩和金毛犬在环山的湖边看日落",
                              "一只金毛犬在山地自行车旁发呆",
                              "4只金毛犬站在红色的墙边",
                              "一只黄色的柯基犬站在雪地",
                              "两只柯基犬坐在草地上吐舌",
                              "一条黄色的柯基犬躺在草地上睡觉"
               ]})


# 
#df=pd.DataFrame(dic_dataset)
print(df)
dataset = image_caption_dataset(df)


train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)  # Define your own dataloader



# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


device = "cpu"  # If using GPU then use mixed precision training.
#model, preprocess = clip.load("test\dataset\RN50.pt", device=device, jit=False)  # Must set jit=False for training
model=shufflenet_v2_x1_0()
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)

#print('model',model)
text_encoder=text_encoder()
vit_encoder=VIT_Encoder()




loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
loss_vit = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)  # Params from paper

for epoch in range(EPOCH):
    for batch in train_dataloader:


        img,list_image, list_txt = batch  # list_images is list of image in numpy array(np.uint8), or list of PIL images
        #print(list_image.shape,list_txt.shape)
        #print(list_image.dtype, list_txt.dtype)
        # images = torch.stack([preprocess(Image.fromarray(img)) for img in list_image],
        #                      dim=0)  # omit the Image.fromarray if the images already in PIL format, change this line to images=list_image if using preprocess inside the dataset class
        #texts = clip.tokenize(list_txt)
        images = list_image # torch.Size([9, 3, 224, 224])
        
        

        #text_features=torch.from_numpy(SentenceTransformer(r"E:\2022_Pro\code\model\clip-ViT-B-32-multilingual-v1").encode(list_txt))
        image_features=model(images)

        vit_features= torch.from_numpy(vit_encoder([tensor_to_PIL(i) for i in img]))
        text_features=torch.from_numpy(text_encoder(list_txt))
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        #logits_per_image = logit_scale * image_features @ vit_features.t()
        #logits_per_vit = logits_per_image.t()

        if device == "cpu":
            ground_truth = torch.arange(BATCH_SIZE).long().to(device)
        else:
            ground_truth = torch.arange(BATCH_SIZE).half().to(device)


        #total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss = loss_img(logits_per_image, ground_truth)
        #total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2 + (loss_img(logits_per_image, ground_truth) + loss_vit(logits_per_text, ground_truth)) / 2
        optimizer.zero_grad()
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        print('[%d] loss: %.7f' %
              (epoch + 1, total_loss))
        torch.save(model.state_dict(),'model.pth')
