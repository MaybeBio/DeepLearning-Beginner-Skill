---
name: dl-skill
description: 深度学习新手入门知识库 — 整合李宏毅(LeeDL)和李沐(d2l)两本教材,覆盖从ML基础到ChatGPT/扩散/强化学习/视觉RL的24章。提供框架速查、代码模板、故障决策树和概念检索
---

# DeepLearning-Beginner-Skill

24 章新手入门深度学习知识库。每章 = Core Idea + Framework + 代码 + 反模式 + 实例 + 知识连接。可用关键词检索概念、按任务找模型选型、故障调试查决策树。

## Core Frameworks

这 7 个框架贯穿全书,是所有技术的基础:

- **ML = 找函数 (LeeDL 3-Step)**: ①定义含未知参数的函数(模型架构);②定义损失(衡量好坏);③梯度下降最小化损失。这个框架适用于监督/自监督/RL/元学习所有范式
- **训练 = 前向 + 反向 + 更新**: `y_pred = model(x)` → `loss(y_pred, y)` → `loss.backward()` → `optimizer.step()`。关键细节:每一批前 `optimizer.zero_grad()`
- **预训练 + 微调**: 海量无标注数据自监督预训练(学通用表示)→ 少量标注微调(适配任务)。BERT/GPT/Stable Diffusion 都靠这个范式。比从头训少用 100× 数据
- **Encoder-Decoder**: 输入编码成隐表示,解码成目标输出。CNN(Vision)→Seq2Seq(NLP)→VAE(生成)→Transformer→扩散 都有此影子
- **注意力 = 加权汇总**: Q·K^T 算分数 → Softmax → 加权 V。"重要内容多看,不重要的少看"。自注意(Q=K=V)是 Transformer 的核心
- **GAN 对抗博弈**: G(生成器)和 D(判别器)交替训练,G 骗 D,D 抓 G。延伸到领域对抗训练、Circuit GAN、RL Actor-Critic
- **训练故障决策树**: Loss 不降→检查梯度(临界点/爆炸/消失);过拟合→正则化/数据增强/早停;欠拟合→加模型复杂度/改特征

## Chapter Index

| # | 章节 | 一句话 |
|---|---|---|
| 1 | [机器学习基础](chapters/ch01-机器学习基础.md) | ML三步骤、任务类型(分类/回归/聚类)、过拟合vs欠拟合 |
| 2 | [数学预备](chapters/ch02-数学预备.md) | 线性代数(矩阵乘/转置/逆)、微积分(梯度/链式法则)、概率(分布/条件) |
| 3 | [线性模型与从零实现](chapters/ch03-线性模型与从零实现.md) | 线性回归(MSE闭式解)、Softmax多分类,从零手写+d2l实现 |
| 4 | [多层感知机](chapters/ch04-多层感知机.md) | 隐藏层+非线性激活(ReLU/Sigmoid),MLP万能逼近,GPU加速 |
| 5 | [深度学习计算](chapters/ch05-深度学习计算.md) | autograd(自动求导)、nn.Module(参数管理)、DataLoader(数据迭代) |
| 6 | [实践方法论](chapters/ch06-实践方法论.md) | 训练流程、故障诊断(Loss曲线分析)、超参搜索策略、验证集设计 |
| 7 | [优化算法](chapters/ch07-优化算法.md) | SGD+Momentum+Adam三件套、学习率调度(Warmup+余弦)、梯度裁剪 |
| 8 | [正则化与泛化](chapters/ch08-正则化与泛化.md) | Weight Decay/Dropout/BN/数据增强/Early Stopping,五件套防过拟合 |
| 9 | [卷积神经网络基础](chapters/ch09-卷积神经网络基础.md) | 卷积核(局部连接+参数共享)、池化(降采样)、多通道、BN作用 |
| 10 | [现代CNN架构](chapters/ch10-现代CNN架构.md) | LeNet→AlexNet→VGG→NiN→GoogLeNet→ResNet→DenseNet,深度+残差 |
| 11 | [循环神经网络](chapters/ch11-循环神经网络.md) | RNN/LSTM(3门控)/GRU(2门控)、BPTT、梯度消失/爆炸、Seq2Seq |
| 12 | [注意力机制与Transformer](chapters/ch12-注意力机制与Transformer.md) | Self-Attention(QKV)、多头注意、位置编码、编码器+解码器、BERT/GPT |
| 13 | [计算机视觉](chapters/ch13-计算机视觉.md) | 数据增强、微调/迁移学习、目标检测(锚框/NMS/SSD/R-CNN)、语义分割(FCN/U-Net) |
| 14 | [自然语言处理](chapters/ch14-自然语言处理.md) | word2vec(Skip-Gram/CBOW)、负采样、GloVe/fastText/BPE、BERT微调 |
| 15 | [生成模型](chapters/ch15-生成模型.md) | GAN(G+D对抗)、WGAN-GP、条件GAN、CycleGAN、IS/FID评估、模式崩塌 |
| 16 | [扩散模型](chapters/ch16-扩散模型.md) | DDPM(前向加噪+逆向去噪)、噪声预测、潜在扩散(Stable Diffusion)、CFG |
| 17 | [自监督学习与表示学习](chapters/ch17-自监督学习与表示学习.md) | BERT(MLM填空)、GPT(自回归)、自编码器、VQ-VAE、特征解耦、多语言BERT |
| 18 | [对抗攻击与模型鲁棒性](chapters/ch18-对抗攻击与模型鲁棒性.md) | FGSM/I-FGSM、白盒vs黑盒(代理网络)、防御(模糊化/随机化/对抗训练) |
| 19 | [迁移学习](chapters/ch19-迁移学习.md) | 领域偏移、领域自适应(领域对抗训练)、领域泛化、微调策略、TTT |
| 20 | [强化学习](chapters/ch20-强化学习.md) | 策略梯度、4版A(即时/累积/折扣/减基线)、Actor-Critic、MC vs TD、探索 |
| 21 | [元学习与终身学习](chapters/ch21-元学习与终身学习.md) | MAML(学初始化)、NAS、N-way K-shot、灾难性遗忘、EWC(守卫bᵢ)、GEM |
| 22 | [网络压缩与可解释性AI](chapters/ch22-网络压缩与可解释性AI.md) | 剪枝/蒸馏/量化/深度可分离/动态计算;显著图/SmoothGrad/探针/全局解释/LIME |
| 23 | [ChatGPT与大语言模型](chapters/ch23-ChatGPT与大语言模型.md) | GPT演进、预训练+监督微调+RLHF三阶段、ICL、多语言迁移、提示工程 |
| 24 | [MineCraft纯视觉RL案例](chapters/ch24-MineCraft纯视觉强化学习案例.md) | LS-Imagine(长短期世界模型)、功用性图、跳跃转换、AC在混合想象序列上 |

## Topic Index

按任务/概念快速定位:

| 要找什么 | 去第几章 |
|---|---|
| 训练不收敛/Loss曲线异常 | Ch 6, 7, 8 |
| 过拟合了怎么办 | Ch 8 |
| 选什么优化器/学习率 | Ch 7 |
| CNN 架构怎么选 | Ch 9, 10 |
| 处理文本/序列数据 | Ch 11, 12, 14 |
| BERT 怎么用 | Ch 14, 17 |
| GPT/ChatGPT 原理 | Ch 17, 23 |
| 图像分类/检测/分割 | Ch 9, 10, 13 |
| 生成图片 | Ch 15, 16 |
| 文生图(Stable Diffusion) | Ch 16 |
| 模型太大要压缩 | Ch 22 |
| 模型决策需解释 | Ch 22 |
| 新任务数据太少 | Ch 13, 19, 21 |
| 玩游戏/AI决策 | Ch 20, 24 |
| 对抗样本攻击与防御 | Ch 18 |
| 多任务持续学习 | Ch 21 |
| 数学基础忘了 | Ch 2 |

## Supporting Files

- **[glossary.md](glossary.md)** — ~100个核心术语的字母排序定义,每术语一行带归属章节
- **[patterns.md](patterns.md)** — 25个可复用的设计模式/技术方案,每种含适用场景+怎么做+注意事项
- **[cheatsheet.md](cheatsheet.md)** — 速查表:训练故障决策树、模型选型、损失函数、优化器默认值、预训练模型选择

## Scope & Limits

**覆盖范围**: 深度学习从入门到进阶的完整知识体系:
- 基础: ML概念、数学预备、线性模型、MLP、计算框架(PyTorch)
- 实践: 训练方法论、优化、正则化、超参调优
- 架构: CNN、RNN/LSTM/GRU、Transformer、ResNet
- 应用: CV(检测/分割)、NLP(word2vec/BERT/GPT)、生成(GAN/扩散)
- 进阶: 自监督、RL、元学习、迁移学习、对抗攻击、网络压缩、XAI、ChatGPT/LLM、视觉RL

**不在范围内**: 图神经网络(GNN)、3D视觉、推荐系统、强化学习的详细理论(Sutton & Barto水平)、多模态(除MineCLIP外)、分布式训练系统

**来源**: 李宏毅深度学习教程v1.2.4(LeeDL_Tutorial) + 李沐动手学深度学习PyTorch 2.0.0(d2l-zh-pytorch)。两书重叠内容融合为统一章节,各自独有的进阶主题作为独立章节保留

**使用方式**: 先查 cheatsheet 快速定位问题/选型,再用 patterns 找技术方案,然后去对应章节深入细节,用 glossary 查术语定义,章节末尾的 Connects To 帮助跳转相关知识
