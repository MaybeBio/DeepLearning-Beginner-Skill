# Chapter 10: 现代CNN架构

## Core Idea
LeNet 之后 CNN 沉寂多年,直到 2012 年 AlexNet 引爆深度学习。从此 CNN 架构演进沿着三条主线: ① **更深**(VGG 块堆叠) ② **更巧的结构**(NiN 1×1卷积+GAP、GoogLeNet Inception 并行、ResNet 残差连接) ③ **更密的连接**(DenseNet)。**ResNet 的残差连接是分水岭**--让网络能从 20 层训到 152 层,深刻改变了所有后续深度网络设计。

## Frameworks Introduced

- **AlexNet (2012) (d2l)**
  - When to use: 理解 DL 兴起的历史起点
  - How: 8 层(5 卷 + 3 全),首次用 **ReLU**(替代 sigmoid 加速训练)、**Dropout**、**GPU 训练**、**数据增强**。ImageNet top-5 错误率从 26% 降到 15%

- **VGG 块 (2014) (d2l)**
  - When to use: 学习"块"的模块化设计
  - How: VGG 块 = 多个 3×3 卷积 + 1 个 2×2 最大池化。**用 2 个 3×3 替代 1 个 5×5**(参数少、感受野相同、非线性更多)。VGG-11/13/16/19 表示层数

- **NiN (Network in Network) (2014) (d2l)**
  - When to use: 理解 1×1 卷积和全局平均池化的作用
  - How: NiN 块 = 卷积 + 1×1 卷积(等效每像素全连接)+ 1×1 卷积。**结尾用全局平均池化(GAP)替代全连接**--大幅减少参数,定位更准

- **GoogLeNet / Inception 块 (2014) (d2l)**
  - When to use: 多尺度特征并行提取
  - How: Inception 块**并行**用 1×1、3×3、5×5 卷积 + 3×3 池化,各路径输出在通道维拼接。用 1×1 卷积做"瓶颈"降维减少计算量。Inception v1/v2/v3/v4 不断改进

- **ResNet 残差块 (2015) (d2l)**
  - When to use: 任何超过 30 层的网络都该用残差连接
  - How: 残差块学**残差 `f(x) - x`** 而非 `f(x)`。结构: 输入 x -> Conv-BN-ReLU -> Conv-BN -> **+ x** -> ReLU。**改变通道/尺寸时用 1×1 卷积调整 x**。让深网训练成为可能

- **DenseNet 稠密连接 (2017) (d2l)**
  - When to use: 需要特征重用、参数效率高的场景
  - How: 每层输出与所有后续层**拼接**(而非相加)。Dense 块内每层都直接连到块内所有后续层;块间用过渡层(1×1 卷积+2×2 池化)降维。**特征重用**降低参数

- **架构演进的关键洞察 (d2l + LeeDL)**
  - When to use: 设计自己的网络时
  - How:
    - **小核堆叠 > 大核单层**: 3 个 3×3 感受野=1 个 7×7,但参数少、非线性多
    - **块化设计**: VGG/ResNet/DenseNet 都是"块 + 块堆叠"
    - **GAP 替代 FC**: 减少参数、支持任意输入尺寸
    - **残差连接是基础**: 深网标配
    - **通道数翻倍 + 空间减半**: 经典下采样模式

## Key Concepts

- **AlexNet**: 8 层 CNN,首次大规模 GPU 训练的深网
- **VGG 块**: 多个 3×3 卷积 + 池化,模块化设计
- **NiN 块**: 卷积 + 1×1 卷积,引入"MLP 卷积"
- **全局平均池化 (GAP)**: 把每个通道的 H×W 平均成 1 个值
- **Inception 块**: 多尺度并行卷积
- **瓶颈结构 (bottleneck)**: 1×1 降维 -> 3×3 卷积 -> 1×1 升维,减少计算
- **残差连接 (residual/skip connection)**: 跨层把输入加到输出
- **残差块 (residual block)**: 含残差连接的基本单元
- **恒等映射 (identity mapping)**: f(x) = x,残差连接让网络容易学这个
- **函数类的嵌套 (nested function class)**: F₁ ⊆ F₂ ⊆ ... 保证更深 ≥ 更浅
- **Dense 块**: 每层都连到后续所有层
- **过渡层 (transition layer)**: Dense 块间的降维层
- **特征重用 (feature reuse)**: DenseNet 的核心思想

## Mental Models

- **小核堆叠**: 3 个 3×3 ≈ 1 个 7×7 感受野,但参数 27 vs 49,非线性 3 vs 1
- **块 = 子网络**: VGG/ResNet/DenseNet 都把"块"作为复用单元,块内同质,块间异构
- **GAP = 通道-wise 池化**: 把空间信息压成 1×1,只剩通道维--天然支持不同输入尺寸
- **残差连接 = 梯度高速公路**: 反向传播时梯度可直接经 skip 通路流回,深网梯度不消失
- **学残差比学原函数容易**: 恒等映射时残差=0,把权重设 0 即可;学原函数需精确拟合
- **Inception = 多尺度并行**: 不同核大小抓不同尺度特征,让网络自己学权重
- **DenseNet 特征重用**: 每层不再"重新学",而是基于前面所有层的拼接做增量

## Anti-patterns

- **深网不用残差连接**: 30 层以上几乎必崩(梯度消失/退化)
- **Inception 不用瓶颈**: 5×5 卷积直接做计算量爆炸;**先用 1×1 降维再做 5×5**
- **GAP 后接大 FC**: 抹掉 GAP 的参数节省优势
- **第一层就用 1×1**: 第一层要混合空间信息,用 3×3 或更大
- **Dense 块不接过渡层**: 通道数爆炸,显存耗尽
- **ResNet 改通道不调 1×1**: 残差相加要求形状匹配,改通道必须用 1×1 卷积调整 skip
- **盲目堆层数**: ResNet-152 在小数据集上可能不如 ResNet-18--架构容量要匹配数据

## Code Examples

**VGG 块**:

```python
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
```

**Inception 块**:

```python
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        # 路径 1: 1x1 卷积
        self.p1 = nn.Sequential(nn.Conv2d(in_channels, c1, kernel_size=1), nn.ReLU())
        # 路径 2: 1x1 -> 3x3
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1), nn.ReLU()
        )
        # 路径 3: 1x1 -> 5x5
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1), nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2), nn.ReLU()
        )
        # 路径 4: 3x3 pool -> 1x1
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1), nn.ReLU()
        )
    
    def forward(self, x):
        return torch.cat([self.p1(x), self.p2(x), self.p3(x), self.p4(x)], dim=1)
```

**ResNet 残差块**:

```python
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        # 改通道/尺寸时,用 1x1 调整 skip
        self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides) if use_1x1conv else None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X               # 残差连接!
        return F.relu(Y)
```
- **What it demonstrates**: 残差块的核心是 `Y += X`--把输入直接加到卷积输出上。**1×1 卷积用于调整 x 的形状以便相加**

**ResNet-18 完整骨架**:

```python
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

net = nn.Sequential(
    # stem: 把 224x224 图像降到 56x56
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    # 4 个 stage, 每个 2 个残差块
    resnet_block(64, 64, 2, first_block=True),
    resnet_block(64, 128, 2),
    resnet_block(128, 256, 2),
    resnet_block(256, 512, 2),
    # 头部: GAP + FC
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10)
)
```

## Reference Tables

**CNN 架构演进时间线**:

| 年份 | 架构 | 关键创新 | ImageNet top-5 err |
|---|---|---|---|
| 1998 | LeNet | 卷积+池化 | (手写数字) |
| 2012 | AlexNet | ReLU、Dropout、GPU | 15.3% |
| 2014 | VGG | 3×3 块堆叠 | 7.3% |
| 2014 | GoogLeNet | Inception 多尺度 | 6.7% |
| 2015 | ResNet | 残差连接,152 层 | 3.6% |
| 2017 | DenseNet | 稠密连接、特征重用 | ~3.4% |

**块结构对照**:

| 架构 | 块内连接 | 块内层间 | 关键参数减少 |
|---|---|---|---|
| VGG | 串接 | 3×3 卷积堆 | - |
| NiN | 串接 + 1×1 | MLP 卷积 | GAP 替代 FC |
| Inception | 并行 | 多核+瓶颈 | 1×1 瓶颈 |
| ResNet | 串接 + skip | 加性残差 | 残差易学 |
| DenseNet | 串接 + 全连 | 拼接特征 | 特征重用 |

## Worked Example

**为什么残差连接让深网能训?** -- 函数类嵌套的论证。

**问题**: 不用残差时,加层不保证网络更强。函数类 F' 可能不包含 F--更深的网络可能学得更差("退化现象")。

**残差的论证**: 设理想函数 `f*(x)`,普通层学 `H(x) = f*(x)`,残差层学 `F(x) = H(x) - x`,即 `H(x) = F(x) + x`。

**关键洞察**: 若 `f*(x) ≈ x`(恒等映射就够好),普通层要把权重精调到拟合 x 很难;残差层只要把 F 的权重设成 0 就完美了--**学"零函数"远比学"恒等函数"容易**。

**函数类嵌套**:
- 普通网络: F' 不一定 ⊇ F
- 残差网络: 残差块总能通过把 F 设为 0 退化为恒等映射,所以 F' ⊇ F **保证嵌套**

**梯度流**: 反向传播时,`H = F + x` 的梯度 = `dF/dx + 1`,**那个 `+1` 让梯度能直接流回浅层**,深网不再梯度消失。

**结果**: ResNet 把网络从 VGG 的 19 层扩到 152 层,ImageNet 错误率从 7% 降到 3.6%。

## Key Takeaways

1. AlexNet 引爆 DL: ReLU + Dropout + GPU + 数据增强
2. VGG 引入"块"思想: 3×3 卷积堆叠,小核胜过大核
3. NiN 引入 1×1 卷积 + GAP,大幅减少参数
4. Inception 块用多尺度并行 + 瓶颈降维
5. ResNet 的残差连接是分水岭,让深网(>30 层)能训
6. 残差让函数类嵌套(F' ⊇ F),保证更深 ≥ 更浅
7. DenseNet 用拼接替代相加,强调特征重用
8. 现代架构 = stem + 多个 stage(块堆叠) + GAP + FC

## Connects To

- **Ch 09**: LeNet 是所有现代 CNN 的祖先
- **Ch 04**: 残差连接也是"加 x",但更深--它是 ReLU 之外让深网可训的关键
- **Ch 12**: Transformer 也用残差连接,思想直接继承自 ResNet
- **Ch 13**: CV 任务的 backbone 都是 ResNet 系列变体
