# utils/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
from torchvision import transforms
from config import config
from torch.utils.data.dataloader import default_collate


class HerbierDataset(Dataset):
    """Herbier数据集加载器"""

    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        print(f"Data directory: {self.data_dir}")
        self.transform = transform or self._get_default_transform(is_train)
        self.samples = []
        self._load_annotations()

    def _get_default_transform(self, is_train):
        """获取默认的数据转换"""
        if is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(config.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD)
            ])
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])

    def _load_annotations(self):
        """加载数据集标注文件"""

        annotation_file = os.path.join(os.path.dirname(self.data_dir), 'annotations.json')
        print(f"正在加载标注文件: {annotation_file}")

        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"找不到标注文件: {annotation_file}")

        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        for ann in annotations:
            image_path = os.path.join(self.data_dir, ann['image_name'])
            if os.path.exists(image_path):
                description = ann.get('description', "")
                self.samples.append({
                    'image_path': image_path,
                    'description': description
                })
            else:
                print(f"图像文件不存在: {image_path}")

        print(f"成功加载样本数量: {len(self.samples)}")
        if len(self.samples) == 0:
            raise ValueError("没有找到有效的样本，请检查标注文件和数据路径。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载并转换图像
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        description = sample.get("description", "")
        if description is None:
            description = ""
        return {
            'image': image,
            'text': sample['description'],
        }


def my_collate_fn(batch):
    # 过滤掉含有 None 的样本
    filtered_batch = [sample for sample in batch if sample["image"] is not None and sample["text"] is not None]

    # 如果所有样本都被过滤掉，则抛出异常或返回一个默认 batch
    if len(filtered_batch) == 0:
            raise ValueError("All samples in this batch are invalid!")


    return default_collate(filtered_batch)


def get_data_loader(data_dir, batch_size, is_train=True, num_workers=4):
    """获取数据加载器"""
    print(f"Initializing HerbierDataset with data_dir={data_dir}")
    dataset = HerbierDataset(data_dir, is_train=is_train)
    print("HerbierDataset initialized.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        collate_fn=my_collate_fn,
        pin_memory=True
    )