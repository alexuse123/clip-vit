# models/base/base_model.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.image_encoder = None
        self.text_encoder = None
        self.image_projection = None
        self.text_projection = None

    @abstractmethod
    def encode_image(self, image):
        """将图像编码为特征向量"""
        pass

    @abstractmethod
    def encode_text(self, text):
        """将文本编码为特征向量"""
        pass

    def forward(self, batch):
        """前向传播"""
        image_features = self.encode_image(batch['image'])
        text_features = self.encode_text(batch['text'])
        return image_features, text_features

    def save(self, path):
        """保存模型"""
        torch.save({
            'state_dict': self.state_dict(),
            'model_config': self.config
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])