"""
肺部MRI图像肺实质分割 - 基于UNet
一键运行：训练 + 预测 + 后处理（层间约束）
"""
# 导入依赖库
import os
import random
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== 全局配置 ====================
# 训练/测试数据集路径配置
TRAIN_IMG_DIR = 'lung_data/img_train'    # 训练图像路径
TRAIN_LAB_DIR = 'lung_data/lab_train'    # 训练标签路径
TEST_IMG_DIR = 'lung_data/img_test'      # 测试图像路径

# 模型超参数配置
IMG_SIZE = 256          # 统一图像尺寸
BATCH_SIZE = 4          # 训练批次大小
EPOCHS = 15             # 训练迭代轮数
LR = 1e-4               # 学习率

# 输出文件配置
MODEL_PATH = 'best_model.pth'    # 最优模型保存路径
RESULT_DIR = 'results'           # 预测结果保存文件夹
CURVE_PATH = 'training_curve.png'# 训练曲线保存路径

# 运行开关
DO_TRAIN = True         # 是否执行训练
DO_PREDICT = True       # 是否执行预测

# =================================================


# ============ 训练数据集类：加载图像+标签 ============
class LungTrainDataset(Dataset):
    """读取训练图像和标签"""

    def __init__(self, img_dir, lab_dir, img_size=256, augment=True):
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.img_size = img_size
        self.augment = augment
        files = sorted(os.listdir(img_dir))
        self.valid = []

        # 【关键】匹配标签文件名：xxx.png → xxx.tif.png（适配老师数据集命名规则）
        for f in files:
            if f.endswith('.png'):
                base = os.path.splitext(f)[0]
                lab_name = base + ".tif.png"
                lab_path = os.path.join(lab_dir, lab_name)
                if os.path.exists(lab_path):
                    self.valid.append(f)

        print(f"[训练集] 找到 {len(self.valid)} 对 图像-标签")

    def __len__(self):
        return len(self.valid)

    # 数据增强：翻转、旋转，提升模型泛化能力
    def _aug(self, img, mask):
        if random.random() > 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() > 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)
        if random.random() > 0.5:
            a = random.randint(-15, 15)
            img, mask = TF.rotate(img, a), TF.rotate(mask, a)
        return img, mask

    # 单张数据读取+预处理
    def __getitem__(self, idx):
        name = self.valid[idx]
        base = os.path.splitext(name)[0]
        lab_name = base + ".tif.png"

        # 读取灰度图像
        img = Image.open(os.path.join(self.img_dir, name)).convert('L')
        mask = Image.open(os.path.join(self.lab_dir, lab_name)).convert('L')

        # 统一缩放图像尺寸
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # 训练阶段开启数据增强
        if self.augment:
            img, mask = self._aug(img, mask)

        # 转换为张量格式
        img = TF.to_tensor(img)
        mask = torch.from_numpy((np.array(mask) > 127).astype(np.float32)).unsqueeze(0)
        return img, mask


# ============ 测试数据集类：仅加载图像，解析文件名 ============
class LungTestDataset(Dataset):
    """读取测试图像（不需要标签）"""

    def __init__(self, img_dir, img_size=256):
        self.img_dir = img_dir
        self.img_size = img_size
        self.files = sorted([f for f in os.listdir(img_dir)
                             if f.endswith('.png') or f.endswith('.tif')])
        print(f"[测试集] 找到 {len(self.files)} 张测试图像")

    def __len__(self):
        return len(self.files)

    # 【关键】解析文件名：提取病例号 + 层号（老师核心要求）
    def parse(self, filename):
        n = filename
        # 去除文件后缀
        for ext in ['.tif.png', '.png', '.tif']:
            if n.endswith(ext):
                n = n[:-len(ext)]
                break
        # 分割病例号和层号
        parts = n.split('_IM_')
        if len(parts) == 2:
            try:
                return parts[0], int(parts[1])
            except:
                return 'unknown', 0
        return 'unknown', 0

    # 测试图像预处理
    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert('L')
        orig_size = img.size  # 保存原始图像尺寸
        img_r = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        tensor = TF.to_tensor(img_r)
        case_id, slice_id = self.parse(name)
        return tensor, name, orig_size, case_id, slice_id


# ============ UNet基础模块：双层卷积 ============
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 卷积+归一化+激活函数 ×2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.conv(x)


# ============ UNet网络主体：编码-解码结构 ============
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        # 编码器：下采样，提取图像特征
        self.e1 = DoubleConv(in_ch, 64)
        self.e2 = DoubleConv(64, 128)
        self.e3 = DoubleConv(128, 256)
        self.e4 = DoubleConv(256, 512)
        self.bt = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(2)  # 最大池化下采样

        # 解码器：上采样，恢复图像尺寸
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.d4 = DoubleConv(1024, 512)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d3 = DoubleConv(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2 = DoubleConv(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_ch, 1)

    # 【关键】跳跃连接：融合浅层细节+深层语义特征
    def _cat(self, x1, x2):
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return torch.cat([x2, x1], dim=1)

    # 前向传播
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bt(self.pool(e4))
        # 上采样+特征融合
        d4 = self.d4(self._cat(self.u4(b), e4))
        d3 = self.d3(self._cat(self.u3(d4), e3))
        d2 = self.d2(self._cat(self.u2(d3), e2))
        d1 = self.d1(self._cat(self.u1(d2), e1))
        # Sigmoid输出0~1概率图
        return torch.sigmoid(self.out(d1))


# ============ 损失函数与评价指标 ============
# Dice损失：解决医学图像正负样本不均衡问题
def dice_loss(pred, target, smooth=1e-6):
    i = (pred * target).sum(dim=(2, 3))
    u = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * i + smooth) / (u + smooth)).mean()

# Dice系数：分割精度评价指标（0~1，数值越高效果越好）
def dice_coeff(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    i = (pred * target).sum(dim=(2, 3))
    u = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2 * i + smooth) / (u + smooth)).mean().item()


# ============ 模型训练函数 ============
def train_model():
    # 自动选择GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===== 开始训练 =====\n使用设备: {device}")

    # 加载训练数据集
    full = LungTrainDataset(TRAIN_IMG_DIR, TRAIN_LAB_DIR, IMG_SIZE, augment=True)

    # 无数据直接退出
    if len(full) == 0:
        print("错误：没有找到训练数据！")
        return

    # 数据集划分：85%训练，15%验证
    n_val = max(1, int(len(full) * 0.15))
    n_tr = len(full) - n_val
    tr, va = random_split(full, [n_tr, n_val], generator=torch.Generator().manual_seed(42))

    print(f"训练: {n_tr} 张  验证: {n_val} 张")

    # 创建数据加载器
    tr_loader = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    va_loader = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 初始化模型、优化器、损失函数
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=5)
    bce = nn.BCELoss()  # 二分类交叉熵损失

    # 记录训练指标
    tL, vL, tD, vD = [], [], [], []
    best = 0.0

    # 训练迭代循环
    for ep in range(1, EPOCHS + 1):
        # 训练模式
        model.train()
        tl = td = 0.0
        pbar = tqdm(tr_loader, desc=f"Epoch {ep}/{EPOCHS} 训练")
        for img, mask in pbar:
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            # 混合损失：DiceLoss + BCELoss
            loss = dice_loss(out, mask) + bce(out, mask)
            # 反向传播更新参数
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
            td += dice_coeff(out, mask)
        tl /= len(tr_loader);
        td /= len(tr_loader)

        # 验证模式
        model.eval()
        vl = vd = 0.0
        with torch.no_grad():
            for img, mask in va_loader:
                img, mask = img.to(device), mask.to(device)
                out = model(img)
                vl += (dice_loss(out, mask) + bce(out, mask)).item()
                vd += dice_coeff(out, mask)
        vl /= len(va_loader);
        vd /= len(va_loader)

        # 更新学习率
        sched.step(vd)
        tL.append(tl);
        vL.append(vl);
        tD.append(td);
        vD.append(vd)
        print(f"Epoch {ep:02d} | 训练Loss {tl:.4f} Dice {td:.4f} | 验证Loss {vl:.4f} Dice {vd:.4f}")

        # 【关键】保存精度最高的最优模型
        if vd > best:
            best = vd
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"   ★ 保存最佳模型 (Dice={best:.4f})")

    # 绘制并保存训练曲线
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(tL, label='Train');
    ax[0].plot(vL, label='Val')
    ax[0].set_title('Loss');
    ax[0].legend();
    ax[0].set_xlabel('Epoch')
    ax[1].plot(tD, label='Train');
    ax[1].plot(vD, label='Val')
    ax[1].set_title('Dice');
    ax[1].legend();
    ax[1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=150)
    plt.close()
    print(f"\n训练完成! 最佳验证Dice={best:.4f}")


# ============ 【核心】层间约束后处理（老师要求） ============
def inter_slice_smoothing(case_slices):
    # 层数过少，直接阈值分割
    if len(case_slices) < 3:
        return [(sid, (p > 0.5).astype(np.uint8)) for sid, p in case_slices]

    smoothed = []
    probs = [p for _, p in case_slices]
    ids = [sid for sid, _ in case_slices]

    # 相邻层加权平滑：利用连续断层的解剖结构连续性优化分割结果
    for i in range(len(probs)):
        if i == 0:
            avg = 0.6 * probs[i] + 0.4 * probs[i + 1]
        elif i == len(probs) - 1:
            avg = 0.6 * probs[i] + 0.4 * probs[i - 1]
        else:
            avg = 0.2 * probs[i - 1] + 0.6 * probs[i] + 0.2 * probs[i + 1]
        smoothed.append((ids[i], (avg > 0.5).astype(np.uint8)))
    return smoothed


# ============ 模型预测+结果保存 ============
def predict_test():
    # 创建结果保存文件夹
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, 'overlay'), exist_ok=True)

    # 加载设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n===== 开始预测 =====")

    # 加载最优模型
    model = UNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 加载测试集
    ds = LungTestDataset(TEST_IMG_DIR, IMG_SIZE)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    cases = defaultdict(list)

    # Step1：模型推理，预测所有测试图像
    print("Step 1/2: 模型推理...")
    with torch.no_grad():
        for img, name, osize, cid, sid in tqdm(loader):
            img = img.to(device)
            prob = model(img).squeeze().cpu().numpy()
            w = int(osize[0].item())
            h = int(osize[1].item())
            cases[cid[0]].append((int(sid.item()), prob, name[0], (w, h)))

    # Step2：层间约束+保存分割结果
    print("Step 2/2: 层间约束 + 保存结果...")
    for cid, lst in tqdm(cases.items()):
        lst.sort(key=lambda x: x[0])  # 按层号排序
        slice_probs = [(sid, p) for sid, p, _, _ in lst]
        smoothed = inter_slice_smoothing(slice_probs)  # 执行层间约束
        sid2mask = {sid: m for sid, m in smoothed}

        for sid, _prob, fname, (ow, oh) in lst:
            mask256 = sid2mask[sid]
            # 缩放回原始尺寸并保存掩码
            mask_img = Image.fromarray(mask256 * 255).resize((ow, oh), Image.NEAREST)
            mask_arr = np.array(mask_img)
            Image.fromarray(mask_arr).save(os.path.join(RESULT_DIR, 'mask', fname))

            # 生成可视化对比图（原图+预测+叠加）
            orig = np.array(Image.open(os.path.join(TEST_IMG_DIR, fname)).convert('L'))
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(orig, cmap='gray');
            ax[0].set_title('原图');
            ax[0].axis('off')
            ax[1].imshow(mask_arr, cmap='gray');
            ax[1].set_title('预测Mask');
            ax[1].axis('off')
            ax[2].imshow(orig, cmap='gray')
            ax[2].imshow(np.ma.masked_where(mask_arr == 0, mask_arr), cmap='autumn', alpha=0.5)
            ax[2].set_title('叠加');
            ax[2].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, 'overlay', fname), dpi=100)
            plt.close()

    print(f"\n预测完成! 共处理 {sum(len(v) for v in cases.values())} 张图")


# ============ 主函数：程序入口 ============
if __name__ == '__main__':
    # 解决matplotlib中文显示乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 按配置执行训练和预测
    if DO_TRAIN:
        train_model()
    if DO_PREDICT:
        if os.path.exists(MODEL_PATH):
            predict_test()
        else:
            print("请先训练模型！")
    print("\n全部完成！")