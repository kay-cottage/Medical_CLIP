from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
from Backbone.shufflenet.nets.model import shufflenet_v2_x1_0
from torchvision import transforms, datasets, utils
import numpy as np
import requests
import torch




def img_encoder():
    model=shufflenet_v2_x1_0()
    model.load_state_dict(torch.load(r'model.pth'))
    print("loaded shufflenet successfully")
    return model

# We use the original clip-ViT-B-32 for encoding images
#img_model = SentenceTransformer('clip-ViT-B-32')
img_model = img_encoder()

# Our text embedding model is aligned to the img_model and maps 50+
# languages to the same vector space
text_model = SentenceTransformer(r"E:\2022_Pro\code\model\clip-ViT-B-32-multilingual-v1")


# Now we load and encode the images
def load_image(url_or_path):
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        return Image.open(requests.get(url_or_path, stream=True).raw)
    else:
        return Image.open(url_or_path)

# We load 3 images. You can either pass URLs or
# a path on your disc
img_paths = [
    # Dog image
    "https://unsplash.com/photos/QtxgNsmJQSs/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjM1ODQ0MjY3&w=640",
]

images = [load_image(img) for img in img_paths]

# Map images to the vector space
#img_embeddings = img_model.encode(images)
img=Image.open(r'E:\2022_Pro\code\VIT\dataset\5.jpg')
img=resize = transforms.Resize([224,224])(img)
img=transforms.ToTensor()(img)
print(img.shape)
img_embeddings =img_model(img.unsqueeze(0))

# Now we encode our text:
texts = [
    #"柯基犬看日落",
    #"两只柯基犬在草地",
    #"自行车和女孩",
    #"草地看日落",
    "没有人",
    "两个人一只狗",
    "女孩跟日落",
    "主人跟柯基",
    "主人跟藏獒",
    "金毛犬在自行车旁边",  # German: A cat
      # Spanish: a beach with palm trees
]

text_embeddings = text_model.encode(texts)

# Compute cosine similarities:
cos_sim = util.cos_sim(text_embeddings, img_embeddings)

for text, scores in zip(texts, cos_sim):
    max_img_idx = torch.argmax(scores)
    print("Text:", text)
    print("Score:", scores[max_img_idx] )
    print("Path:", img_paths[max_img_idx], "\n")
