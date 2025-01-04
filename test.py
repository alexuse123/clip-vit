# test.py
import torch
from models.vit_model import VisionTextModel
from utils import get_data_loader
from config import config
import numpy as np
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


def evaluate_model():
    # 加载模型
    model = VisionTextModel().to(config.DEVICE)
    model.load(f'{config.MODEL_DIR}/best_model.pth')
    model.eval()

    # 获取测试数据加载器
    test_loader = get_data_loader(config.TEST_DATA_DIR, config.BATCH_SIZE, is_train=False)

    # 收集所有预测结果
    all_image_features = []
    all_text_features = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            batch = {k: v.to(config.DEVICE) if torch.is_tensor(v)
            else v for k, v in batch.items()}

            image_features, text_features = model(batch)

            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())

    # 连接所有特征
    all_image_features = torch.cat(all_image_features)
    all_text_features = torch.cat(all_text_features)

    # 计算相似度矩阵
    similarity = all_image_features @ all_text_features.t()

    # 计算准确率@K
    def accuracy_at_k(similarity, k):
        _, indices = similarity.topk(k)
        correct = (indices == torch.arange(len(similarity)).view(-1, 1))
        return correct.float().mean().item()

    results = {
        'accuracy@1': accuracy_at_k(similarity, 1),
        'accuracy@5': accuracy_at_k(similarity, 5),
        'accuracy@10': accuracy_at_k(similarity, 10)
    }

    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")


if __name__ == '__main__':
    evaluate_model()