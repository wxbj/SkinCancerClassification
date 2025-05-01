from yacs.config import CfgNode as CN

_C = CN()

# 结果保存的文件夹
_C.OUTPUT_DIR = "results"
# 随机种子
_C.SEED= 1
# 批大小
_C.BATCH_SIZE = 64
# GPU训练
_C.DEVICE= "cuda:0"
# 学习率
_C.LR= 1e-5
# Adm权重衰减
_C.WEIGHT_DECAY = 0.0005
# 训练轮次
_C.EPOCHS= 100
# 学习率策略
_C.LR_POLICY = "platform"
# 学习率衰减周期
_C.LR_DECAY_ITERS = 1
# 实验名称
_C.NAME= "BayesianNetwork"
# 进程名
_C.PROCTITLE = "RunBayesianNetwork"

# 测试图片的地址
_C.IMAGE_PATH = "data/HAM10000_images_part_1/ISIC_0024320.jpg"
