# config/config.py
import os
import torch

class Config:
    def __init__(self):
        # 项目根目录配置
        self.ROOT_DIR = 'D:/DeepLearning'

        # 数据集配置
        self.TRAIN_DATA_DIR = os.path.join(self.ROOT_DIR, "data/train/images")
        self.VAL_DATA_DIR= os.path.join(self.ROOT_DIR, "data/val/images")
        self.TEST_DATA_DIR = os.path.join(self.ROOT_DIR, "data/test/images")

        # 模型配置
        self.MODEL_DIR = 'models/checkpoints'           #模型存储的目录路径
        self.MODEL_NAME = 'google/vit-base-patch16-224'  # Google 提供的 ViT（Vision Transformer）模型
        self.MODEL_DIR2 = 'models/checkpointscoca'
        self.MODEL_NAME2 = "coca_ViT-B-32"
        self.PRETRAINED = "mscoco_finetuned_laion2B-s13B-b90k"
        # 训练配置
        self.BATCH_SIZE = 16
        self.LEARNING_RATE = 1e-4
        self.NUM_EPOCHS = 64
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 图像预处理配置
        self.IMAGE_SIZE = 224
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        # 创建必要的目录
        self._create_directories()

    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
           # self.DATA_DIR,
            self.TRAIN_DATA_DIR,
            self.VAL_DATA_DIR,
            self.TEST_DATA_DIR,
            self.MODEL_DIR,
            self.MODEL_DIR2
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# 创建全局配置实例
config = Config()