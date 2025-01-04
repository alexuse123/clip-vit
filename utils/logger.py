# utils/logger.py
import logging
import os
from config import config


def setup_logger(name, log_file, level=logging.INFO):
    """设置日志器"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger



# 创建日志目录
log_dir = os.path.join(config.ROOT_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

# 创建训练日志器
train_log_file = os.path.join(log_dir, 'train.log')
train_logger = setup_logger('train', train_log_file)

# 创建评估日志器
eval_log_file = os.path.join(log_dir, 'eval.log')
eval_logger = setup_logger('eval', eval_log_file)