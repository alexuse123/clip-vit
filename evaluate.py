import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import os
import sys

# 添加项目根目录到 Python 路径
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models.vit_model import VisionTextModel
from models.coca_model import CoCaModel
from utils import get_data_loader
from config.config import config


class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

        # 加载测试数据
        self.test_loader = get_data_loader(
            data_dir=config.TEST_DATA_DIR,
            batch_size=config.BATCH_SIZE
        )

    def load_models(self):
        """加载两个模型的检查点"""
        try:
            # 加载ViT模型
            print("Loading Vision-Text model...")
            vit_checkpoint = os.path.join(self.config.MODEL_DIR, 'best_model.pth')
            print(f"Loading checkpoint from {vit_checkpoint}")

            # 加载模型
            checkpoint = torch.load(vit_checkpoint, map_location=self.device)
            print("Creating Vision-Text model instance...")
            self.vit_model = VisionTextModel(config=self.config)  # 使用正确的类名
            print("Loading state dict...")
            self.vit_model.load_state_dict(checkpoint['model_state_dict'])
            self.vit_model.to(self.device)
            self.vit_model.eval()
            print("ViT model loaded successfully")

            # 加载CoCa模型
            print("\nLoading CoCa model...")
            coca_checkpoint = os.path.join(self.config.MODEL_DIR2, 'best_model.pth')
            print(f"Loading checkpoint from {coca_checkpoint}")

            checkpoint = torch.load(coca_checkpoint, map_location=self.device)
            self.coca_model = CoCaModel(self.config)
            self.coca_model.load_state_dict(checkpoint['model_state_dict'])
            self.coca_model.to(self.device)
            self.coca_model.eval()
            print("CoCa model loaded successfully")

        except FileNotFoundError as e:
            print(f"Error: Could not find checkpoint file - {e}")
            raise
        except KeyError as e:
            print(f"Error: Checkpoint file is missing expected keys - {e}")
            raise
        except Exception as e:
            print(f"Error loading models: {e}")
            print(f"Detailed error: {str(e)}")
            raise

    def compute_similarity(self, image_features, text_features):
        """计算相似度矩阵"""
        # 归一化特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        similarity = (image_features @ text_features.T)
        return similarity.cpu().numpy()

    def evaluate_model(self, model, name):
        """评估单个模型的性能"""
        total_correct = 0
        total_samples = 0
        similarities_list = []
        retrieval_at_k = {1: 0, 5: 0, 10: 0}  # top-k检索指标

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f'Evaluating {name}'):
                images = batch['image'].to(self.device)
                texts = batch['text']

                # 获取特征
                image_features, text_features = model(batch)

                # 计算相似度
                similarity = self.compute_similarity(image_features, text_features)
                similarities_list.append(similarity)

                # 计算准确率
                predictions = similarity.argmax(axis=1)
                correct = (predictions == np.arange(len(predictions))).sum()

                # 计算top-k检索准确率
                for k in retrieval_at_k.keys():
                    top_k_indices = similarity.argsort(axis=1)[:, -k:]
                    for i, indices in enumerate(top_k_indices):
                        if i in indices:
                            retrieval_at_k[k] += 1

                total_correct += correct
                total_samples += len(predictions)

        accuracy = total_correct / total_samples
        # 打印相似度列表的形状以进行调试
        print("Similarities shapes:", [s.shape for s in similarities_list])

        # 确保所有批次的相似度矩阵具有相同的形状
        similarities = []
        for sim in similarities_list:
            if len(sim.shape) == 2:
                similarities.append(sim)
            else:
                print(f"Warning: Skipping similarity matrix with unexpected shape: {sim.shape}")

        if similarities:
            avg_similarity = np.mean([s.mean() for s in similarities])

        # 计算最终的top-k检索准确率
        for k in retrieval_at_k.keys():
            retrieval_at_k[k] = retrieval_at_k[k] / total_samples

        return {
            'accuracy': accuracy,
            'avg_similarity': avg_similarity,
            'similarities': similarities_list,
            'retrieval_at_k': retrieval_at_k
        }

    def visualize_results(self, vit_results, coca_results):
        """可视化对比结果"""
        plt.style.use('default')

        # 创建保存目录
        results_dir = os.path.join(self.config.ROOT_DIR, 'evaluation_results')
        os.makedirs(results_dir, exist_ok=True)

        fig = plt.figure(figsize=(15, 10))

        # 1. 准确率对比柱状图
        ax1 = plt.subplot(2, 2, 1)
        models = ['ViT', 'CoCa']
        accuracies = [vit_results['accuracy'], coca_results['accuracy']]
        bars = ax1.bar(models, accuracies, color=['lightblue', 'lightgreen'])
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom')

        # 2. 相似度分布直方图
        ax2 = plt.subplot(2, 2, 2)
        vit_sims = np.concatenate([s.flatten() for s in vit_results['similarities']])
        coca_sims = np.concatenate([s.flatten() for s in coca_results['similarities']])

        ax2.hist(vit_sims, bins=50, alpha=0.5, label='ViT', color='lightblue')
        ax2.hist(coca_sims, bins=50, alpha=0.5, label='CoCa', color='lightgreen')
        ax2.set_title('Similarity Distribution')
        ax2.set_xlabel('Similarity Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # 3. Top-K检索准确率
        ax3 = plt.subplot(2, 2, 3)
        k_values = list(vit_results['retrieval_at_k'].keys())
        vit_topk = [vit_results['retrieval_at_k'][k] for k in k_values]
        coca_topk = [coca_results['retrieval_at_k'][k] for k in k_values]

        x = np.arange(len(k_values))
        width = 0.35
        ax3.bar(x - width / 2, vit_topk, width, label='ViT', color='lightblue')
        ax3.bar(x + width / 2, coca_topk, width, label='CoCa', color='lightgreen')

        ax3.set_title('Top-K Retrieval Accuracy')
        ax3.set_xlabel('K')
        ax3.set_ylabel('Accuracy')
        ax3.set_xticks(x)
        ax3.set_xticklabels(k_values)
        ax3.legend()

        plt.tight_layout()
        results_path = os.path.join(results_dir, 'model_comparison.png')
        plt.savefig(results_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to {results_path}")
        plt.close()

    def run_evaluation(self):
        """运行完整评估流程"""
        print("Loading models...")
        self.load_models()

        print("\nEvaluating ViT model...")
        vit_results = self.evaluate_model(self.vit_model, "ViT")

        print("\nEvaluating CoCa model...")
        coca_results = self.evaluate_model(self.coca_model, "CoCa")

        print("\nGenerating visualization...")
        self.visualize_results(vit_results, coca_results)

        # 创建结果目录并保存结果
        results_dir = os.path.join(self.config.ROOT_DIR, 'evaluation_results')
        os.makedirs(results_dir, exist_ok=True)

        # 保存详细结果到文本文件
        results_file = os.path.join(results_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write("Model Evaluation Results\n")
            f.write("=" * 50 + "\n\n")

            f.write("ViT Model:\n")
            f.write(f"  Accuracy: {vit_results['accuracy']:.3f}\n")
            f.write(f"  Average Similarity: {vit_results['avg_similarity']:.3f}\n")
            f.write("  Top-K Retrieval Accuracy:\n")
            for k, acc in vit_results['retrieval_at_k'].items():
                f.write(f"    Top-{k}: {acc:.3f}\n")

            f.write("\nCoCa Model:\n")
            f.write(f"  Accuracy: {coca_results['accuracy']:.3f}\n")
            f.write(f"  Average Similarity: {coca_results['avg_similarity']:.3f}\n")
            f.write("  Top-K Retrieval Accuracy:\n")
            for k, acc in coca_results['retrieval_at_k'].items():
                f.write(f"    Top-{k}: {acc:.3f}\n")

        print(f"\nDetailed results saved to {results_file}")


if __name__ == "__main__":
    evaluator = ModelEvaluator(config)
    evaluator.run_evaluation()