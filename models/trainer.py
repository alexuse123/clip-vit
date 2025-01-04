# models/trainer.py
import torch
from tqdm import tqdm
from utils import train_logger, eval_logger
from .loss import CLIPLoss


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.config = config
        self.criterion = CLIPLoss()

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch in pbar:
                # 将数据移到设备
                batch = {k: v.to(self.config.DEVICE) if torch.is_tensor(v)
                else v for k, v in batch.items()}

                # 前向传播
                image_features, text_features = self.model(batch)

                # 计算损失
                loss = self.criterion(image_features, text_features,
                                      self.model.logit_scale)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 更新进度条
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        train_logger.info(f'Epoch {epoch} - Average Loss: {avg_loss:.4f}')
        return avg_loss

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                batch = {k: v.to(self.config.DEVICE) if torch.is_tensor(v)
                else v for k, v in batch.items()}

                image_features, text_features = self.model(batch)
                loss = self.criterion(image_features, text_features,
                                      self.model.logit_scale)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        eval_logger.info(f'Validation Loss: {avg_loss:.4f}')
        return avg_loss