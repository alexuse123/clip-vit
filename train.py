# train.py
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.vit_model import VisionTextModel
from models.trainer import Trainer
from utils import get_data_loader
from config import config

import  os

# 获取项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_loader = get_data_loader(config.TRAIN_DATA_DIR, config.BATCH_SIZE, is_train=True)
val_loader = get_data_loader(config.VAL_DATA_DIR, config.BATCH_SIZE, is_train=False)
test_loader = get_data_loader(config.TEST_DATA_DIR, config.BATCH_SIZE, is_train=False)
# 配置路径
annotations_path = os.path.join(ROOT_DIR, "data/train/annotations.json")
images_dir = os.path.join(ROOT_DIR, "data/train/images")


print("Annotations Path:", annotations_path)
print("Images Directory:", images_dir)
def main():
    # 初始化模型
    model = VisionTextModel(config).to(config.DEVICE)

    # 获取数据加载器
    train_loader = get_data_loader(config.TRAIN_DATA_DIR, config.BATCH_SIZE, is_train=True)
    val_loader = get_data_loader(config.VAL_DATA_DIR, config.BATCH_SIZE, is_train=False)

    # 优化器
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # 训练器
    trainer = Trainer(model, train_loader, val_loader, optimizer, config)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        # 训练一个epoch
        train_loss = trainer.train_epoch(epoch)

        # 验证
        val_loss = trainer.validate()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(f'{config.MODEL_DIR}/best_model.pth')

        # 更新学习率
        scheduler.step()

        # 保存检查点
        if (epoch + 1) % 5 == 0:
            model.save(f'{config.MODEL_DIR}/checkpoint_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    main()