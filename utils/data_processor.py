# data_processor.py
import os
import shutil
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging


class DataProcessor:
    def __init__(self):
        self.setup_logging()
        self.raw_data_path = Path('Data_Herbier_trait_segmentation')
        self.processed_path = [Path('train'), Path('val'), Path('test')]

        # 创建必要的目录
        self._create_directories()

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _create_directories(self):
        """创建必要的目录结构"""
        for path in self.processed_path:
            images_dir = path / 'images'
            masks_dir = path / 'masks'
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
            self.logger.info(f"Created directory: {images_dir}")
            self.logger.info(f"Created directory: {masks_dir}")

    def process_images(self):
        """处理并划分数据集"""
        # 获取所有图像和mask
        image_dir = self.raw_data_path / 'images'
        mask_dir = self.raw_data_path / 'masks'

        # 检查目录
        if not image_dir.exists():
            self.logger.error(f"图像目录不存在: {image_dir}")
            return
        if not mask_dir.exists():
            self.logger.error(f"Mask目录不存在: {mask_dir}")
            return

        # 获取所有图像和对应的mask
        all_images = list(image_dir.glob('*.jpg'))
        self.logger.info(f"找到的图像文件数量: {len(all_images)}")

        all_masks = set(f.name for f in mask_dir.glob('*.jpg'))
        self.logger.info(f"找到的mask文件数量: {len(all_masks)}")

        # 获取有效的图像-mask对
        image_files = []
        for img_path in sorted(all_images):
            if img_path.name in all_masks:
                image_files.append(img_path)
            else:
                self.logger.warning(f"图像 {img_path.name} 没有对应的mask文件")

        if not image_files:
            self.logger.error("没有找到有效的图像-mask对！")
            return

        self.logger.info(f"找到 {len(image_files)} 个有效的图像-mask对")

        # 随机划分数据集
        np.random.seed(42)
        total_samples = len(image_files)
        indices = np.random.permutation(total_samples)

        # 计算划分点
        train_end = int(0.7 * total_samples)
        val_end = int(0.85 * total_samples)

        # 划分数据集
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        self.logger.info(f"训练集: {len(train_idx)} 样本")
        self.logger.info(f"验证集: {len(val_idx)} 样本")
        self.logger.info(f"测试集: {len(test_idx)} 样本")

        # 处理并保存数据
        self._process_subset(image_files, mask_dir, train_idx, 'train')
        self._process_subset(image_files, mask_dir, val_idx, 'val')
        self._process_subset(image_files, mask_dir, test_idx, 'test')

    def _process_subset(self, image_files, mask_dir, indices, subset):
        """处理数据子集"""
        subset_dir = Path(subset)
        self.logger.info(f"处理{subset}集...")

        images_dir = subset_dir / 'images'
        masks_dir = subset_dir / 'masks'

        # 处理数据
        annotations = []
        for idx in tqdm(indices, desc=f'处理{subset}集'):
            image_path = image_files[idx]
            mask_path = mask_dir / image_path.name

            if not mask_path.exists():
                continue

            try:
                # 处理图像
                processed_image = self._process_image(image_path)
                processed_mask = self._process_mask(mask_path)

                # 生成新的文件名
                new_filename = f"{subset}_{len(annotations):04d}.jpg"

                # 保存处理后的图像和mask
                cv2.imwrite(str(images_dir / new_filename),
                            cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(masks_dir / new_filename), processed_mask)

                # 收集标注信息
                annotations.append({
                    'image_name': new_filename,
                    'original_image': image_path.name,
                })

            except Exception as e:
                self.logger.error(f"处理 {image_path.name} 时出错: {e}")

        # 保存标注文件
        ann_file = subset_dir / 'annotations.json'
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
            self.logger.info(f"保存{subset}集标注文件: {len(annotations)}个样本")

    def _process_image(self, image_path):
        """处理单个图像"""
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            raise Exception(f"无法读取图像: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize到指定大小
        image = cv2.resize(image, (224, 224),
                           interpolation=cv2.INTER_LANCZOS4)

        return image

    def _process_mask(self, mask_path):
        """处理单个mask"""
        # 读取mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise Exception(f"无法读取mask: {mask_path}")

        # resize到指定大小（使用最近邻插值以保持标签值）
        mask = cv2.resize(mask, (224, 224),
                          interpolation=cv2.INTER_NEAREST)

        return mask

    def verify_processed_data(self):
        """验证处理后的数据"""
        for subset in ['train', 'val', 'test']:
            subset_dir = Path(subset)

            # 检查目录和文件
            for subdir in ['images', 'masks']:
                dir_path = subset_dir / subdir
                if not dir_path.exists():
                    self.logger.error(f"{subset}集的{subdir}目录不存在")
                    continue

                file_count = len(list(dir_path.glob('*.jpg')))
                self.logger.info(f"{subset}集的{subdir}目录包含{file_count}个文件")

            # 检查标注文件
            ann_file = subset_dir / 'annotations.json'
            if not ann_file.exists():
                self.logger.error(f"{subset}集的标注文件不存在")
                continue

            with open(ann_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
                self.logger.info(f"{subset}集包含{len(annotations)}个标注")


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_images()
    processor.verify_processed_data()