# Cheatsheet

## 训练故障决策树

```
Loss不降?
├── 梯度为零? → 临界点(鞍点/局部极小):加动量/Adam/调学习率
├── 梯度爆炸/消失? → BatchNorm/ResNet改残差/换激活/Gradient Clipping
└── 学习率不对? → Warmup + 余弦衰减;先小网格搜索

过拟合?(训练好/验证差)
├── 数据不够 → 数据增强/更多数据
├── 模型太大 → Dropout/减小网络/Weight Decay(L2)
├── 训太久 → Early Stopping
└── BN统计量漂移 → 验证时用训练统计量

欠拟合?(训练/验证都差)
├── 模型太小 → 加层/加宽
├── 训不够 → 增epoch/调学习率
└── 特征不够 → 加特征/换更强架构
```

## 模型选型速查

| 任务 | 推荐架构 | 备选 |
|---|---|---|
| 图像分类 | ResNet/ViT | EfficientNet/ConvNeXt |
| 目标检测 | Faster R-CNN/YOLO | SSD/DETR |
| 语义分割 | U-Net/DeepLab | FCN/SegFormer |
| 文本分类 | BERT/RoBERTa | fastText/DistilBERT |
| 文本生成 | GPT/LLaMA | T5/BART |
| 序列标注 | BERT+CRF | BiLSTM+CRF |
| 机器翻译 | Transformer | T5/BART |
| 图像生成 | Stable Diffusion | GAN/VAE |
| 语音识别 | Conformer/Whisper | DeepSpeech |

## 常用损失函数

| 任务 | 损失 |
|---|---|
| 二分类 | BCE(二元交叉熵) |
| 多分类 | CrossEntropy(Softmax+CE) |
| 回归 | MSE(均方误差) / MAE(绝对) |
| GAN判别器 | BCE(真=1/假=0) |
| GAN生成器 | BCE(骗D输出1) |
| WGAN | D(fake)-D(real)+GP |
| 扩散 | MSE(ε_pred, ε) |
| 自编码器 | MSE/BCE(重构误差) |
| RL策略梯度 | ΣA_n·e_n(加权交叉熵) |
| 知识蒸馏 | α·KL(软)+ (1-α)·CE(硬) |

## 优化器选择

| 场景 | 推荐 | 学习率起点 |
|---|---|---|
| 通用/NLP | Adam/AdamW | 1e-4 ~ 3e-4 |
| CV | SGD + Momentum | 0.01 ~ 0.1 |
| GAN | Adam β1=0.5 | 1e-4 ~ 2e-4 |
| 微调 | Adam/AdamW | 1e-5 ~ 5e-5 |
| 大批量 | LAMB/LARS | 按批量缩放 |

## 核心超参默认值

| 超参 | 默认 | 调参范围 |
|---|---|---|
| 学习率 | 1e-3 | 1e-5 ~ 1e-1 |
| 批大小 | 32/64 | 16 ~ 512 |
| Weight Decay | 1e-4 | 1e-6 ~ 1e-1 |
| Dropout率 | 0.5 | 0.1 ~ 0.7 |
| Momentum | 0.9 | 0.8 ~ 0.99 |
| 梯度裁剪 | 1.0 | 0.1 ~ 10 |
| Warmup步数 | 4000 | 500 ~ 10000 |

## 预训练模型选型

| 领域 | 首选 | 轻量替代 |
|---|---|---|
| NLP理解 | BERT-base/RoBERTa | DistilBERT/ALBERT |
| NLP生成 | GPT-3.5/4 | GPT-2/LLaMA-7B |
| CV分类 | ViT/ResNet-50 | MobileNet/EfficientNet-B0 |
| CV生成 | Stable Diffusion | GAN |
| 语音 | Whisper | Wav2Vec2 |
| 多语言 | XLM-R | mBERT |

## 数据量-策略匹配

| 标注数据 | 策略 |
|---|---|
| 0 | 自监督预训练(SSL) |
| <1K/类 | 小样本学习/元学习(MAML) |
| 1K~10K | 预训练+微调(冻底层) |
| 10K~100K | 预训练+微调(解冻全部) |
| >100K | 可尝试从头训练 |
| >1M | 从头训练可能超预训练 |

## 经典BERT vs GPT

| 维度 | BERT | GPT |
|---|---|---|
| 架构 | Transformer编码器 | Transformer解码器 |
| 方向 | 双向 | 单向(自回归) |
| 任务 | MLM(填空) | 下一词预测 |
| 适合 | NLU(分类/QA/NER) | NLG(对话/写作) |
| 输入 | [CLS]句[SEP] | 前缀文本 |
| 微调 | 加分类头 | prompt/in-context |

## 攻击与防御速查

| 攻击 | 信息 | 迭代 | 防御 | 类型 |
|---|---|---|---|---|
| FGSM | 白盒 | 1次 | 对抗训练 | 主动 |
| I-FGSM/PGD | 白盒 | 多次 | 对抗训练 | 主动 |
| 黑盒(代理) | 查询 | 可变 | 模糊化/压缩 | 被动 |
| 通用扰动 | 白盒 | 多次 | 随机化防御 | 被动/主动 |

## GAN vs VAE vs 扩散

| 维度 | GAN | VAE | 扩散 |
|---|---|---|---|
| 生成质量 | 高 | 中(模糊) | **极高** |
| 训练稳定性 | 难(崩塌) | 简单 | **稳定** |
| 多样性 | 易崩塌 | 好 | 好 |
| 速度 | **快**(1步) | 快 | 慢(多步) |
| 可控性 | 难 | 中 | **易**(CFG) |
