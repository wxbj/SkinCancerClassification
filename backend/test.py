import logging
import os
import argparse

import numpy as np
import torch
from PIL import Image
import torch.nn.parallel
import torch.utils.data
import setproctitle
import torch.utils.data.distributed
from torchvision import transforms

from backend.config import cfg
from backend.util.misc import setup_logger
from backend.core.models.BayesianNetwork import CombinedBayesianNetwork

# 允许加载破损的图像
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 忽视警告信息
import warnings
warnings.filterwarnings('ignore')

def test(cfg, image_path, path=""):
    # 定义子日志记录器trainer。
    logger = logging.getLogger("SkinCanverClassification.tester")
    # 设置设备
    device = torch.device(cfg.DEVICE)

    # 图像预处理（与训练时一致）
    transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # 创建模型并加载权重
    model = CombinedBayesianNetwork(dim_encoder_out=2048, num_classes=7, use_cuda=True, pretrained=False)
    model.to(device)
    model.eval()

    checkpoint_path = os.path.join(path, cfg.OUTPUT_DIR, "model_best1.pt")
    print(f"=> loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    class_names = [
        'Actinic keratoses',
        'Basal cell carcinoma',
        'Benign keratosis-like lesions',
        'Dermatofibroma',
        'Melanocytic nevi',
        'Melanoma',
        'Vascular lesions'
    ]

    # 推理并获取置信度、不确定度
    with torch.no_grad():
        probabilities, uncertainty, beliefs = model.predict(image_tensor, return_uncertainty=True)

        # 获取预测标签
        pred_index = torch.argmax(probabilities, dim=1).item()

        # 获取最大类别概率作为置信度
        confidence = float(torch.max(probabilities, dim=1).values.item()) * 100.0

        # 将不确定度转换为1维，并确保其为单个数值
        uncertainty = torch.atleast_1d(uncertainty).cpu().numpy()

        # 打印预测结果
        logger.info(f"预测类别为: {class_names[pred_index]}, 置信度: {confidence:.2f}%, 不确定度: {uncertainty[0][0]:.2f}")

    return class_names[pred_index], confidence


def main():
    parser = argparse.ArgumentParser(description="Single Image Inference")
    parser.add_argument("opts", help="Modify config options", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.opts is not None and len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    # 加载配置
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    setproctitle.setproctitle(f'{cfg.PROCTITLE}')
    logger = setup_logger("SkinCanverClassification", cfg.OUTPUT_DIR, 0)
    logger.info("Starting single image inference")

    test(cfg, cfg.IMAGE_PATH)


if __name__ == "__main__":
    main()