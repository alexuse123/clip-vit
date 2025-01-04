
import sys
import os

sys.path.append("D:/DeepLearning")
import torch
import open_clip
from PIL import Image
import streamlit as st
from torchvision import transforms
from config.config import Config

config = Config()
# 配置路径和参数
DATA_DIR = "D:/DeepLearning/data/train/images"  # 替换为你的数据路径
ANNOTATIONS_FILE = "D:/DeepLearning/data/train/annotations.json"  # 替换为你的标注文件路径
# ↓↓↓ 你自定义的模型权重路径
CHECKPOINT_PATH = r"D:\DeepLearning\models\checkpoints\best_model.pth"

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 加载 CLIP 模型及预处理
result = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
model = result[0]  # 获取模型
preprocess = result[1]  # 获取预处理工具

# 1.1) 加载你训练好的权重到 model
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

# 2) 加载数据集并生成索引库
def load_dataset(annotations_file, img_dir):
    import json
    dataset = []
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    for ann in annotations:
        img_path = os.path.join(img_dir, ann['image_name'])
        dataset.append(img_path)
    return dataset

dataset = load_dataset(ANNOTATIONS_FILE, DATA_DIR)
image_features_dataset = []

# 3) 逐张图片生成特征并保存
for img_path in dataset:
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    image_features = model.encode_image(image).cpu().detach()
    image_features_dataset.append(image_features)

# 拼成一个 [N, D] 大小的张量
image_features_dataset = torch.stack(image_features_dataset).squeeze(1)

# 4) Streamlit 前端
st.title("Image retrieval system")
query = st.text_input("Enter a text description to search:")

if query:
    with st.spinner("searching..."):
        # 将文本转为 CLIP 的 token
        query_tokens = open_clip.tokenize([query]).to(device)
        # 编码文本为特征
        query_features = model.encode_text(query_tokens).cpu().detach()  # [1, D]

        # 计算和所有图像特征的余弦相似度 -> similarities: [N]
        similarities = torch.nn.functional.cosine_similarity(query_features, image_features_dataset, dim=1)

        # 获取相似度最高的图片索引
        max_index = torch.argmax(similarities).item()
        max_score = similarities[max_index].item()

        # ---- 阈值控制（可选） ----
        threshold = 0.25  # 具体数值可调
        if max_score < threshold:
            st.write(f"No related images found（Similarity below threshold {threshold}）。")
        else:
            # 显示相似度最高的图像及其信息
            img_path = dataset[max_index]
            st.write(f"Most relevant images (similarity: {max_score:.4f}):")
            st.image(Image.open(img_path), caption=f"image path: {os.path.basename(img_path)}")
