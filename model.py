# -*- coding: utf-8 -*-
"""
Created on Wed Aug.17 09:12:41 2022
@author: gw.kayak
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer,util
from Backbone.shufflenet.nets.model import shufflenet_v2_x1_0
import numpy as np

def text_encoder():
    with torch.no_grad():
        text_encoder=SentenceTransformer(r"E:\2022_Pro\code\model\clip-ViT-B-32-multilingual-v1")
    print('loaded text_encoder successfully')
    return text_encoder
    
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.img_encoder=self.img_encoder()
        self.text_encoder=text_encoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def img_encoder(self):
        img_encoder=shufflenet_v2_x1_0()
        return img_encoder.forward

        
    def forward(self,img,text):
        image_features=self.img_encoder(img)
        text_features=torch.from_numpy(self.text_encoder.encode(text))
        
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text
        

'''
a=Net()
pic=torch.rand(1,3,224,224)
text=['我爱你','i am fine']
a,b=a(pic,text)
print(a.shape,b.shape)
'''
