# Chapter 22: 网络压缩与可解释性 AI

## Core Idea
**网络压缩**让大模型(BERT/GPT)适配边缘设备(手表、自驾车)--降低延迟、保护隐私。五种技术: ① **网络剪枝**(去不重要参数/神经元,彩票假说解释为何大网络易训);② **知识蒸馏**(教师网络教学生网络,Softmax 加温度 T 平滑分布);③ **参数量化**(权重聚类、哈夫曼编码、二值网络 +1/-1);④ **网络架构设计**(深度可分离卷积 = 深度卷积 + 点卷积,参数降至 1/k²);⑤ **动态计算**(网络按资源调整深度/宽度)。**可解释性 AI (XAI)** 解答"模型为何这么判断"--银行贷款、医疗诊断、法律、自动驾驶都需要理由。分两类: **局部解释**(为何这张图是猫?用显著图、SmoothGrad、积分梯度、探针);**全局解释**(对模型而言猫长什么样?找让滤波器/类别激活最大的 X*,加正则或用生成器约束)。

## Frameworks Introduced

- **网络剪枝 (Network Pruning) (LeeDL)**
  - When to use: 大网络变小,去冗余参数
  - How:
    1. 训练大网络
    2. 评估每个参数/神经元重要性(绝对值、输出非零次数等)
    3. 剪掉不重要的(每次 ~10%,避免伤太大)
    4. 微调剩余参数恢复性能
    5. 重复 2-4 直到网络够小
  - **权重剪枝**: 网络形状不规则,GPU 难加速(实际不快)
  - **神经元剪枝**: 形状规则,易实现,GPU 可加速(推荐)
  - 关键: 95% 参数可剪,准确率仅掉 1-2%

- **彩票假说 (Lottery Ticket Hypothesis) (LeeDL)**
  - When to use: 解释为何大网络易训,小网络难训
  - How:
    - 大网络 = 多个子网络的组合
    - 训练大网络 = 同时训多个子网络(买多张彩票)
    - 只要有子网络成功,大网络就成功
    - **关键实验**: 剪枝后小网络用**原初始参数**能训起来,随机重新初始化训不起来
  - 变体: Deconstructing Lottery Tickets--正负号是关键,绝对值不重要
  - 反驳: Rethinking the Value of Network Pruning--学习率大时观察不到彩票现象
  - 关键: ICLR 2019 最佳论文,解释大网络优势

- **知识蒸馏 (Knowledge Distillation) (LeeDL)**
  - When to use: 大网络(教师)教小网络(学生)
  - How:
    1. 训练教师网络(大)
    2. 学生网络输入相同图片,目标 = 教师输出(非真实标签)
    3. 学生逼近教师的**软标签**(分布,如 1:0.7, 7:0.2, 9:0.1)
    4. **Softmax 加温度**: `y'_i = exp(y_i/T) / Σ exp(y_j/T)`
    - T > 1: 分布平滑,让学生学到"1 与 7 像"等关系
    - T 太大: 所有类差不多,学不到东西
  - **教师可集成**: 多个网络平均当教师,学生学集成结果
  - 关键: 软标签比硬标签信息更多;Hinton 2015 提出

- **参数量化 (Parameter Quantization) (LeeDL)**
  - When to use: 减少参数存储空间
  - How:
    - **降低精度**: 64 位 -> 16 位 -> 8 位(性能不掉甚至更好)
    - **权重聚类**: 用 K-means 分 N 群,每群用一个代表值;只需 log₂(N) 位存所属群
    - **哈夫曼编码**: 高频值用少位,低频值用多位
    - **二值网络**: 权重只 ±1,1 位存参;BinaryConnect 等甚至比原网络好(防过拟合)
  - 关键: 16/8 位几乎无损;权重聚类 + 哈夫曼可压到 1 位

- **深度可分离卷积 (Depthwise Separable Convolution) (LeeDL)**
  - When to use: CNN 参数压缩(MobileNet 核心)
  - How:
    - **深度卷积 (depthwise)**: 每通道一个滤波器,只管空间卷积,通道数不变。参数量: `k×k×I`
    - **点卷积 (pointwise)**: 1×1 卷积,只管通道间关系。参数量: `I×O`
    - 总参数: `k×k×I + I×O`,vs 标准卷积 `k×k×I×O`
    - 比值 ≈ `1/O + 1/(k×k)` ≈ **1/(k×k)**(O 通常大)
    - k=3: 参数降至 1/9;k=2: 降至 1/4
  - 关键: 把一层拆两层(类比低秩近似),空间与通道分离

- **低秩近似 (Low Rank Approximation) (LeeDL)**
  - When to use: 全连接层压缩
  - How:
    - 原: W 是 N×M,参数量 NM
    - 拆: W = U·V,V 是 N×K,U 是 K×M,参数量 K(N+M)
    - K 远小于 N、M 时大幅压缩
    - 限制: W 的秩 ≤ K,表达能力受限
  - 关键: 深度可分离卷积本质是低秩近似

- **动态计算 (Dynamic Computation) (LeeDL)**
  - When to use: 同一网络适配不同计算资源
  - How:
    - **动态深度**: 每层接分类头,资源少时早停;损失 `L = e_1 + e_2 + ... + e_L`(所有头都逼近正确答案)
    - **动态宽度**: 同一网络多宽度配置,所有配置都训;资源少用窄
    - **自适应**: 网络自己决定深度/宽度(简单图早停,难图跑完)--SkipNet、BlockDrop
  - 关键: 一个网络替代多个网络,节省存储

- **可解释性 AI 目标 (LeeDL)**
  - When to use: 模型需解释决策理由
  - How:
    - **局部解释**: 为何这张图是猫?(针对特定输入)
    - **全局解释**: 对模型而言猫长什么样?(针对模型参数)
    - **目标本质**: 好解释 = 让人高兴的解释(哈佛打印机实验: "因为我需要先印"也是理由,93% 接受)
  - 关键: 不必了解模型一切,只需给出可接受理由

- **局部解释 - 显著图 (Saliency Map) (LeeDL)**
  - When to use: 哪些像素对分类重要
  - How:
    - **遮挡法**: 灰色方块滑过图片,放哪使输出大变 = 重要区域
    - **梯度法**: `重要性 = ∂e/∂x_i`(损失对像素的偏导);画成图即显著图
    - **SmoothGrad**: 加多次噪声取平均显著图,降噪
    - **积分梯度**: 解决梯度饱和(大象鼻子长度--超长后梯度为 0,但实际重要)
  - 关键: PASCAL VOC 案例显示机器靠图片左下角"马"字识别马,非看马本身--XAI 揭露作弊

- **局部解释 - 探针 (Probing) (LeeDL)**
  - When to use: 看网络某层学到什么
  - How:
    - 训分类器(探针)从某层嵌入预测属性
    - **POS 探针**: 嵌入 -> 词性分类器,准 = 嵌入含词性信息
    - **NER 探针**: 嵌入 -> 命名实体分类器
    - **TTS 探针**: 嵌入 -> 语音合成,看能否还原讲述者特征(不能 = 已去除)
  - 警告: 探针分类器太弱会误判(可能嵌入有信息但探针没训好)
  - 关键: 用辅助任务验证嵌入含什么信息

- **全局解释 (LeeDL)**
  - When to use: 模型"心目中"某类长什么样
  - How:
    - 找 X* 最大化某滤波器/类别的激活: `X* = argmax_X Σ a_ij`(梯度上升)
    - **问题**: 找出的 X* 像噪声(对抗攻击原理)
    - **加正则**: `X* = argmax_X y_i + R(X)`,R(X) = 像素平方和(抑制极端值)、梯度平方和(平滑)
    - **生成器约束**: `X = G(z)`,找 `z* = argmax_z y_i`,G 是 GAN/VAE 生成器,保证 X 像真实图
  - 关键: 全局解释需强先验(生成器)才让人认可

- **LIME 局部可解释模型无关解释 (LeeDL)**
  - When to use: 用简单模型模仿黑盒局部行为
  - How:
    - 黑盒全局不可用线性模仿,但局部可
    - 在样本邻域采样,用线性模型拟合黑盒输出
    - 分析线性模型系数 = 局部特征重要性
  - 关键: 全局复杂、局部简单,用局部线性解释

## Key Concepts

- **网络压缩 (network compression)**: 减少模型大小/计算
- **边缘设备 (edge device)**: 手表、自驾车等资源受限设备
- **网络剪枝 (network pruning)**: 去冗余参数
- **权重剪枝**: 以参数为单位,形状不规则
- **神经元剪枝**: 以神经元为单位,形状规则
- **彩票假说 (lottery ticket hypothesis)**: 大网络含易训子网络
- **知识蒸馏 (knowledge distillation)**: 教师教学生
- **教师/学生网络 (teacher/student)**: 大/小网络
- **软标签 (soft label)**: 教师输出的概率分布
- **温度 T (temperature)**: Softmax 平滑参数
- **参数量化 (parameter quantization)**: 降低参数精度
- **权重聚类 (weight clustering)**: 分群代表参数
- **哈夫曼编码 (Huffman encoding)**: 不等长编码
- **二值网络 (binary network)**: 权重只 ±1
- **BinaryConnect**: 二值连接方法
- **深度可分离卷积**: 深度卷积 + 点卷积
- **深度卷积 (depthwise)**: 每通道一滤波器
- **点卷积 (pointwise)**: 1×1 卷积
- **低秩近似**: W = U·V 拆解
- **动态计算**: 网络按资源调深度/宽度
- **动态深度/宽度**: 早停或多宽度配置
- **XAI**: 可解释性 AI
- **局部解释 (local explanation)**: 为何这张图是 X
- **全局解释 (global explanation)**: 模型心中 X 长什么样
- **显著图 (saliency map)**: 像素重要性图
- **SmoothGrad**: 加噪声平均降噪
- **积分梯度 (integrated gradients)**: 解决梯度饱和
- **探针 (probing)**: 用辅助分类器探嵌入信息
- **LIME**: 局部线性模型解释黑盒
- **生成器约束**: GAN/VAE 保证 X* 真实

## Mental Models

- **大网络 = 彩票包牌**: 多子网络 = 多彩票,中奖率高
- **剪枝后微调 = 复原**: 小损伤可微调恢复,大损伤难
- **权重剪枝 = 不规则网络**: GPU 难加速,实际不快
- **神经元剪枝 = 规则网络**: 易实现,GPU 加速
- **教师 = 软标签信息多**: 1 像 7 比 1 是 1 信息多
- **温度 T = 分布平滑度**: T 大平滑,T 小集中,T 无穷都一样
- **深度可分离 = 空间与通道分离**: 深度卷积管空间,点卷积管通道
- **低秩近似 = 矩阵分解**: W = U·V,K 小则参数少
- **动态计算 = 一网多用**: 一个网络替代多个不同大小网络
- **局部解释 = 这张图为何是猫**: 针对特定输入
- **全局解释 = 模型心中猫长啥样**: 针对模型参数
- **显著图 = 像素重要性热力图**: 梯度大 = 重要
- **梯度饱和 = 梯度失效**: 大象鼻子超长后梯度 0,但实际重要
- **探针 = 旁敲侧击**: 用辅助任务反推嵌入内容
- **全局解释噪声 = 对抗攻击同源**: 找最大激活 X* 像噪声
- **生成器约束 = 强先验**: 保证 X* 像真实图
- **LIME = 局部线性近似**: 黑盒全局复杂,局部可线性

## Anti-patterns

- **权重剪枝期望 GPU 加速**: 形状不规则,实际更慢。**用神经元剪枝**
- **一次剪大量参数**: 伤太大,微调难恢复。**每次剪 ~10%**
- **直接训小网络代替剪枝**: 难达到剪枝效果。**先训大再剪**
- **剪枝后随机重训**: 彩票假说失效。**用原初始参数**
- **知识蒸馏用硬标签**: 失去软标签信息。**用教师软标签 + 温度**
- **温度 T 太大**: 分布太平,学不到东西。**T 典型 2-20**
- **参数量化精度过低不评估**: 性能可能掉。**16/8 位几乎无损,1 位需评估**
- **权重聚类训练后做**: 参数差太大。**训练时加聚类约束**
- **深度可分离卷积不用**: 错过参数压缩。**MobileNet 等标配**
- **动态计算用多网络**: 浪费存储。**一个网络动态调整**
- **显著图直接信梯度**: 梯度饱和误导。**用积分梯度或 SmoothGrad**
- **探针分类器太弱**: 误判嵌入无信息。**用强分类器**
- **全局解释不加正则**: X* 像噪声。**加 R(X) 或用生成器**
- **期望 XAI 揭示模型全部**: 不现实。**目标是人可接受的解释**

## Code Examples

**网络剪枝(神经元剪枝)**:

```python
import torch
import torch.nn as nn

def prune_neurons(model, layer_idx, prune_ratio=0.3):
    """按神经元输出绝对值均值剪枝"""
    layer = model.layers[layer_idx]
    # 评估每个神经元重要性(用一批数据统计输出)
    activations = []
    with torch.no_grad():
        for x, _ in dataloader:
            acts = layer(x)
            activations.append(acts.mean(dim=0))  # 每神经元平均激活
    importance = torch.stack(activations).mean(dim=0)
    
    # 保留 top-(1-prune_ratio) 的神经元
    n_keep = int(len(importance) * (1 - prune_ratio))
    keep_idx = torch.topk(importance, n_keep).indices
    
    # 构建新层(维度减小)
    new_layer = nn.Linear(layer.in_features, n_keep)
    new_layer.weight.data = layer.weight.data[keep_idx]
    new_layer.bias.data = layer.bias.data[keep_idx]
    model.layers[layer_idx] = new_layer
    return model, keep_idx

# 剪枝训练循环
model = train_large_model()
for iter in range(prune_iterations):
    model, keep_idx = prune_neurons(model, layer_idx=-2, prune_ratio=0.1)
    # 微调
    for epoch in range(finetune_epochs):
        for x, y in dataloader:
            loss = nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**知识蒸馏**:

```python
class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.7):
        super().__init__()
        self.T = T
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        # 软标签损失(KL 散度,带温度)
        soft_loss = nn.functional.kl_div(
            nn.functional.log_softmax(student_logits / self.T, dim=1),
            nn.functional.softmax(teacher_logits / self.T, dim=1),
            reduction='batchmean') * (self.T ** 2)
        # 硬标签损失
        hard_loss = nn.functional.cross_entropy(student_logits, labels)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

# 训练
teacher = load_pretrained_teacher().eval()
student = StudentModel()
criterion = DistillationLoss(T=4.0, alpha=0.7)

for x, y in dataloader:
    with torch.no_grad():
        teacher_logits = teacher(x)
    student_logits = student(x)
    loss = criterion(student_logits, teacher_logits, y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```
- **What it demonstrates**: 知识蒸馏 = 软标签(KL)+ 硬标签(CE),温度 T 平滑分布

**深度可分离卷积(MobileNet 核心)**:

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        # 深度卷积: 每通道一个滤波器(groups=in_ch)
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, 
                                    padding=kernel_size//2, groups=in_ch)
        # 点卷积: 1x1 改变通道数
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# 对比标准卷积参数量
in_ch, out_ch, k = 64, 256, 3
standard = in_ch * out_ch * k * k  # 147,456
separable = in_ch * k * k + in_ch * out_ch  # 576 + 16,384 = 16,960
print(f"压缩比: {standard / separable:.1f}x")  # ~8.7x
```

**显著图(梯度法)**:

```python
def saliency_map(model, x, target_class):
    """计算每个像素的梯度重要性"""
    x.requires_grad_(True)
    output = model(x)
    score = output[0, target_class]
    
    model.zero_grad()
    score.backward()
    
    # 取绝对值最大通道
    saliency = x.grad.abs().max(dim=1)[0]  # (B, H, W)
    return saliency

# SmoothGrad
def smooth_grad(model, x, target_class, n_samples=50, noise_level=0.1):
    """加噪声平均降噪"""
    saliencies = []
    for _ in range(n_samples):
        noise = torch.randn_like(x) * noise_level
        saliencies.append(saliency_map(model, x + noise, target_class))
    return torch.stack(saliencies).mean(dim=0)
```

**全局解释(找滤波器最敏感的图片)**:

```python
def find_filter_activation(model, layer_idx, filter_idx, 
                            image_shape=(1, 224, 224), n_iter=100, lr=0.1):
    """找让指定滤波器激活最大的图片 X*"""
    X = torch.randn(image_shape, requires_grad=True)  # 随机初始化
    
    for _ in range(n_iter):
        # 前向到目标层
        h = X.unsqueeze(0)
        for i, layer in enumerate(model.features):
            h = layer(h)
            if i == layer_idx:
                activation = h[0, filter_idx]  # 该滤波器的特征图
                break
        
        # 最大化激活 + 正则(像素值 + 梯度平滑)
        loss = activation.mean() - 1e-3 * X.pow(2).mean() - 1e-3 * X.grad.abs().mean() if X.grad else activation.mean()
        loss = activation.mean() - 1e-3 * X.pow(2).mean()
        
        if X.grad is not None:
            X.grad.zero_()
        loss.backward()
        X.data += lr * X.grad  # 梯度上升
        X.grad.zero_()
    
    return X.detach()
```

## Reference Tables

**网络压缩五技术对比**:

| 技术 | 压缩比 | 实现难度 | 性能损失 | 互斥? |
|---|---|---|---|---|
| 网络剪枝 | 95% 参数可剪 | 中 | 1-2% | 否 |
| 知识蒸馏 | 10-100× | 中 | 小 | 否 |
| 参数量化 | 8-32× | 易 | 极小(16/8位) | 否 |
| 架构设计 | 4-9× | 中 | 小 | 否 |
| 动态计算 | 灵活 | 高 | 无(按需) | 否 |

**剪枝单位对比**:

| 单位 | 形状 | GPU 加速 | 实现 | 推荐 |
|---|---|---|---|---|
| 权重 | 不规则 | **难** | 难 | 不推荐 |
| 神经元 | 规则 | 易 | 易 | **推荐** |

**知识蒸馏温度 T 影响**:

| T | 分布 | 效果 |
|---|---|---|
| 1 | 集中(原 Softmax) | 软标签≈硬标签,无效 |
| 2-20 | 平滑 | 信息多,推荐 |
| ∞ | 均匀 | 学不到东西 |

**深度可分离卷积参数对比**:

| 核大小 k | 标准卷积 | 深度可分离 | 压缩比 |
|---|---|---|---|
| 1×1 | k×k×I×O | I×O + I×O | 1×(无效) |
| 2×2 | 4IO | 4I + IO | ~4× |
| 3×3 | 9IO | 9I + IO | ~9× |
| 5×5 | 25IO | 25I + IO | ~25× |

**局部解释方法对比**:

| 方法 | 原理 | 优点 | 缺点 |
|---|---|---|---|
| 遮挡法 | 灰块滑过看输出变 | 直观 | 慢 |
| 梯度法 | ∂e/∂x | 快 | 噪声多 |
| SmoothGrad | 加噪平均 | 降噪 | 慢 |
| 积分梯度 | 沿路径积分 | 解决饱和 | 复杂 |

**XAI 两大类**:

| 类型 | 问题 | 方法 |
|---|---|---|
| 局部解释 | 为何这张图是猫? | 显著图、探针、LIME |
| 全局解释 | 模型心中猫长啥样? | 找 X* + 正则 + 生成器 |

## Worked Example

**为什么大网络剪枝比直接训小网络好?** -- 彩票假说案例。

**实验**:
1. 训练大网络(随机初始化 θ⁰) -> 得 θ^*
2. 剪枝得小网络(参数子集)
3. **情况 A**: 小网络用 θ⁰ 对应子集初始化 -> 训起来
4. **情况 B**: 小网络随机重新初始化 -> 训不起来

**彩票假说解释**:
- 大网络 = 多个子网络的组合(包牌)
- θ⁰ 中含"幸运子网络"--初始参数正好让它易训
- 剪枝留下的是这个幸运子网络
- 情况 A: 用了幸运初始化,训起来
- 情况 B: 失去幸运初始化,训不起来

**Deconstructing Lottery Tickets 发现**:
- 只保留正负号(绝对值都用 α 取代):训起来
- 改变正负号:训不起来
- **结论**: 初始化的**正负号**是关键,绝对值不重要

**反驳观点**(Rethinking the Value of Network Pruning):
- 学习率大时观察不到彩票现象
- 非结构化(权重)剪枝才有,结构化(神经元)剪枝无
- 彩票假说适用范围有限

**关键洞察**:
1. **大网络优势**: 含多个子网络,中奖率高
2. **剪枝保留幸运初始化**: 不只是参数子集,还有初始值
3. **正负号决定方向**: 梯度下降方向由正负号决定
4. **Supermask**: 甚至不训练,直接剪枝随机网络可得分类器(米开朗基罗"塑像在石头里")

**知识蒸馏案例**:
- 教师: 1 -> 0.7, 7 -> 0.2, 9 -> 0.1(软标签)
- 学生学: 1 与 7 有点像(信息比"1 是 1"多)
- **零样本学习**: 学生从未见过 7,但教师说"1 像 7,7 像 9",学生可推断 7 的特征

**显著图揭露作弊**:
- PASCAL VOC 2007: 马的图片,机器识别为"马"
- 显著图显示: 机器看的是左下角"马"字(来自网站水印),不是马本身
- **XAI 价值**: 揭露模型靠无关特征作弊,非真理解

**全局解释的噪声问题**:
- 直接找 `X* = argmax y_i`: X* 像随机噪声(对抗攻击同源)
- 机器不需要"像猫的图"才能输出"猫"--加噪声即可
- **加正则**: R(X) = 像素平方和 + 梯度平方和 -> X* 更规则
- **生成器约束**: X = G(z),找 z* -> X* 像真实图(因 G 只生成真实图)

## Key Takeaways

1. 网络压缩让大模型适配边缘设备(降低延迟、保护隐私)
2. **网络剪枝**: 95% 参数可剪,准确率仅掉 1-2%;**神经元剪枝**比权重剪枝易加速
3. **彩票假说**: 大网络含幸运子网络,剪枝保留其初始化;正负号是关键
4. **知识蒸馏**: 教师软标签比硬标签信息多;Softmax 加温度 T 平滑分布
5. **参数量化**: 16/8 位无损,权重聚类 + 哈夫曼可压到 1 位(二值网络)
6. **深度可分离卷积**: 深度卷积(空间)+ 点卷积(通道),参数降至 1/k²
7. **动态计算**: 一个网络按资源调深度/宽度,替代多网络
8. 五种技术**非互斥**,可组合使用
9. XAI 必要性: 银行/医疗/法律/自动驾驶需决策理由;好解释 = 人可接受
10. **局部解释**: 显著图(梯度法)、SmoothGrad、积分梯度、探针
11. **全局解释**: 找最大激活 X*,需加正则或生成器约束避免噪声
12. **LIME**: 用线性模型局部模仿黑盒,分析局部特征重要性

## Connects To

- **Ch 6**: CNN 压缩用深度可分离卷积(MobileNet)
- **Ch 18**: 对抗攻击的全局解释噪声同源(找最大激活 X* 像噪声)
- **Ch 17**: BERT/GPT 等大模型是压缩对象;知识蒸馏常用 BERT 教小 BERT
- **Ch 15**: 生成器约束用 GAN/VAE
- **Ch 14**: 探针常用于分析 BERT 嵌入含什么信息
