import os
import argparse
import logging
import torch
import tqdm
import numpy as np

import torch.optim as optim
import torch.nn.parallel
import torch.utils.data
import setproctitle
import torch.utils.data.distributed
from torch import nn
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from backend.config import cfg
from util.misc import set_random_seed,setup_logger,save_checkpoint
from core.dataset.dataset_input import SkinDataset, train_df, validation_df, test_df
from core.models.BayesianNetwork import CombinedBayesianNetwork
from core.loss.loss import SSBayesRiskLoss,KLDivergenceLoss

# 允许加载破损的图像
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 忽视警告信息
import warnings
warnings.filterwarnings('ignore')


def trains(cfg):
    # 定义子日志记录器trainer。
    logger = logging.getLogger("SkinCanverClassification.trainer")

    # 数据集处理
    composed = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.CenterCrop(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    logger.info(f"composed transform: {composed}")

    training_set = SkinDataset(train_df, transform=composed)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                                     num_workers=4, pin_memory=True)
    validation_set = SkinDataset(validation_df, transform=composed)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                                       num_workers=4, pin_memory=True)
    test_set = SkinDataset(test_df, transform=composed)
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                                 num_workers=4, pin_memory=True)
    logger.info(f"training set size: {len(training_set)}, validation set size: {len(validation_set)}, test set size: {len(test_set)}")

    # 模型选择
    device = torch.device(cfg.DEVICE)
    model = CombinedBayesianNetwork(dim_encoder_out=2048,num_classes=7,pretrained=True)
    model.to(device)
    logger.info(model)

    # 损失函数
    bayes_risk = SSBayesRiskLoss()  # 贝叶斯损失
    kld_loss = KLDivergenceLoss()  # KL损失
    cross_entropy_loss = nn.CrossEntropyLoss()  # 交叉熵损失

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR, betas=(0.9, 0.99), weight_decay=cfg.WEIGHT_DECAY)
    optimizer.zero_grad()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)

    # 开始训练
    best_acc = 0

    for epoch in tqdm.tqdm(range(cfg.EPOCHS)):
        running_loss = 0
        model.train()
        for data_sample, y in training_generator:
            data_gpu = data_sample.cuda(non_blocking=True)
            y_gpu = y.cuda(non_blocking=True)

            optimizer.zero_grad()
            # 量化分类不确定性的证据深度学习
            eye = torch.eye(7, dtype=torch.float32, device=device)
            labels = eye[y_gpu]
            output = model(data_gpu)

            annealing_coef = min(1.0, epoch / cfg.EPOCHS)
            loss = bayes_risk(output, labels) + annealing_coef * kld_loss(output, labels) + cross_entropy_loss(output, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        logger.info(f"此时的损失函数值为：{running_loss}")
        scheduler.step(running_loss)
        # 对测试集进行评估
        if epoch % 5 == 0:
            model.eval()
            result_array = [] # 预测结果
            gt_array = [] # 真实值
            confidence_array = [] # 置信度
            uncertainty_array = [] # 不确定度

            with torch.no_grad():
                for data_sample, y in test_generator:
                    data_gpu = data_sample.to(device)
                    y_gpu = y.to(device)

                    # 使用 predict 方法获取概率、不确定度等
                    probs, uncertainty, _ = model.predict(data_gpu)  # [B, num_classes]、[B, 1]

                    pred_labels = torch.argmax(probs, dim=1)
                    confidence = torch.max(probs, dim=1).values  # 最大类别概率作为置信度

                    result_array.extend(pred_labels.cpu().numpy())
                    gt_array.extend(y_gpu.cpu().numpy())
                    confidence_array.extend(confidence.cpu().numpy())

                    uncertainty = torch.atleast_1d(uncertainty)  # 保证至少是一维
                    uncertainty_array.extend(uncertainty.view(-1).cpu().numpy())

            correct_results = np.array(result_array) == np.array(gt_array)
            sum_correct = np.sum(correct_results)
            accuracy = sum_correct / len(result_array)

            logger.info(
                f'Epoch: {epoch}  准确度: {accuracy*100:.2f}%  置信度: {np.mean(confidence_array)*100:.2f}%  不确定度: {np.mean(uncertainty_array)*100:.2f}%')

            is_best = accuracy > best_acc
            best_acc = max(accuracy, best_acc)

        save_checkpoint(cfg,{
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec': best_acc,
            }, is_best)

def main():
    # 设置命令行
    parser = argparse.ArgumentParser(description="SkinCanverClassificationRun")
    parser.add_argument("--proctitle",
                        type=str,
                        default="BYS-CNN",
                        help="allow a process to change its title", )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    if args.opts is not None and len(args.opts) > 0:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 修改进程名称
    setproctitle.setproctitle(f'{cfg.PROCTITLE}')

    # 设置随机种子
    set_random_seed(cfg.SEED)

    # 根据用户提供的文件夹创建保存结果的文件夹
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # 训练前将训练的一些参数输出到日志文件里，运行配置
    logger = setup_logger("SkinCanverClassification", cfg.OUTPUT_DIR, 0)
    logger.info("Running with config:\n{}".format(cfg))

    trains(cfg)


if __name__ == '__main__':
    main()