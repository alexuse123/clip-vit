import torch
import torch.nn as nn
from transformers import ViTModel, BertModel, BertTokenizer
from .base.base_model import BaseModel
from config import config
import numpy as np


class VisionTextModel(BaseModel):
    def __init__(self,config):
        super(VisionTextModel, self).__init__()
        self.config = config
        # 初始化图像编码器
        self.image_encoder = ViTModel.from_pretrained(config.MODEL_NAME)

        # 初始化文本编码器
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # 投影层
        self.image_projection = nn.Linear(768, 512)
        self.text_projection = nn.Linear(768, 512)

        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        features = self.image_encoder(image).last_hidden_state[:, 0]
        return self.image_projection(features)

    def encode_text(self, text):

        text = [t if t is not None else "" for t in text]
        # 文本标记化
        tokens = self.tokenizer(text, padding=True, truncation=True,
                                max_length=77, return_tensors="pt")
        tokens = {k: v.to(config.DEVICE) for k, v in tokens.items()}

        features = self.text_encoder(**tokens).last_hidden_state[:, 0]
        return self.text_projection(features)

    def forward(self, batch):
        print("DEBUG => batch['text']:", batch['text'])
        print("DEBUG => type of batch['text']:", type(batch['text']))

        image_features = self.encode_image(batch['image'])
        text_features = self.encode_text(batch['text'])
        # 标准化特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features

    def save(self, path):
        """保存模型"""
        print(f"Saving model to {path}")
        torch.save({
            'model_state_dict': self.state_dict(),  # 保存模型参数
            'config': self.config  # 保存配置
        }, path)

    def load(cls, path):
        """从文件加载模型"""
        print(f"Loading model from {path}")
        checkpoint = torch.load(path)
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

