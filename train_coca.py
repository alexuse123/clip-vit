import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.coca_model import CoCaModel

from models.trainer import Trainer
from utils import get_data_loader
from config import config

# 获取项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 你也可以在此处初始化 data_loader，或在 main() 内初始化
train_loader = get_data_loader(config.TRAIN_DATA_DIR, config.BATCH_SIZE, is_train=True)
val_loader   = get_data_loader(config.VAL_DATA_DIR,   config.BATCH_SIZE, is_train=False)
test_loader  = get_data_loader(config.TEST_DATA_DIR,  config.BATCH_SIZE, is_train=False)

# 配置路径（如果需要）
annotations_path = os.path.join(ROOT_DIR, "data/train/annotations.json")
images_dir = os.path.join(ROOT_DIR, "data/train/images")

print("Annotations Path:", annotations_path)
print("Images Directory:", images_dir)

def main():
    # 初始化 CoCaModel
    model = CoCaModel(config).to(config.DEVICE)

    # 获取数据加载器
    train_loader = get_data_loader(config.TRAIN_DATA_DIR, config.BATCH_SIZE, is_train=True)
    val_loader   = get_data_loader(config.VAL_DATA_DIR,   config.BATCH_SIZE, is_train=False)
    print(model.coca_model.text.token_embedding.weight.shape)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    # 定义学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # 初始化训练器
    trainer = Trainer(model, train_loader, val_loader, optimizer, config)

    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        # 训练一个 epoch
        train_loss = trainer.train_epoch(epoch)

        # 验证
        val_loss = trainer.validate()

        # 如果出现更好（更低）的验证损失，则保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(os.path.join(config.MODEL_DIR2, 'best_model.pth'))

        # 更新学习率
        scheduler.step()

        # 周期性保存检查点
        if (epoch + 1) % 5 == 0:
            model.save(f'{config.MODEL_DIR2}/checkpoint_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    main()
