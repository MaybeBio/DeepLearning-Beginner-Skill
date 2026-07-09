# Chapter 12: 注意力机制与Transformer

## Core Idea
RNN 长依赖难学、不能并行。**注意力机制**用 Q(query)、K(key)、V(value) 三元组: 用 Q 与所有 K 算相似度得权重,加权 V 得输出--让模型"看哪里"。**自注意力**让序列中每个位置直接关注所有其他位置,任意两步距离为 1。**Transformer = 自注意力 + 多头 + 位置编码 + 残差 + LayerNorm + FFN**,完全无 RNN/CNN,可并行,长依赖天然友好,2017 年后统治 NLP 并进军 CV/语音。

## Frameworks Introduced

- **Q/K/V 框架 (d2l)**
  - When to use: 任何注意力机制的标准抽象
  - How: 给定 query q,对每个值 v_i 配一个 key k_i,用相似度函数 `a(q, k_i)` 算权重,softmax 后加权 V:`attention(q, K, V) = Σ softmax(a(q, k_i)) · v_i`
  - 直觉: 查询像搜索词,key 像索引,value 是返回内容

- **注意力评分函数 (d2l)**
  - When to use: 选 a(q, k) 的具体形式
  - How:
    - **加性注意力** `a(q,k) = vᵀ·tanh(W_q·q + W_k·k)` -- 通用,Q/K 维度可不同
    - **缩放点积注意力** `a(q,k) = (qᵀk)/√d` -- 主流(Transformer 用),要求 Q/K 同维度,可矩阵并行

- **自注意力 (LeeDL Ch6 + d2l)**
  - When to use: 序列内任意两元素要交互(CNN 局部、RNN 慢)
  - How: 把序列 X 通过 3 个线性变换投影成 Q=XW_Q、K=XW_K、V=XW_V,然后 `Attention(Q,K,V) = softmax(QKᵀ/√d) · V`。**输出与输入同长度,每位置都看了整个序列**
  - 关键: 任意两位置距离 = 1,长依赖直接学

- **多头注意力 (d2l)**
  - When to use: 让模型从不同表示子空间关注不同位置
  - How: 把 Q/K/V 拆成 h 份(每份 d/h 维),各做一次注意力,结果拼接再投影。**多组并行,各组学不同模式**

- **位置编码 (d2l)**
  - When to use: 自注意力本身无位置信息,需注入
  - How: 给每个位置加一个固定/可学的位置向量 `X = X + P`,P 用正弦/余弦函数或可学嵌入。常用 sin/cos 不同频率

- **Transformer 架构 (LeeDL Ch7 + d2l)**
  - When to use: 序列到序列任务(翻译、摘要、聊天)
  - How:
    - **编码器**: N 个块,每块 = 多头自注意力 + FFN,各加残差 + LayerNorm
    - **解码器**: N 个块,每块 = 掩蔽自注意力 + 编码器-解码器注意力 + FFN
    - **掩蔽**: 解码时未来位置 mask 掉,保证自回归
    - **FFN**: 两层 MLP,中间维度通常 4× 模型维度

- **LayerNorm vs BatchNorm (d2l)**
  - When to use: 序列/可变长度数据用 LayerNorm
  - How: LayerNorm 沿特征维归一化(每个样本独立);BatchNorm 沿 batch 维归一化(每特征独立)。**LayerNorm 不依赖 batch size,适合序列和推理**

- **序列到序列的应用 (LeeDL Ch7)**
  - When to use: 翻译、摘要、QA、聊天
  - How: 编码器读输入序列得表示,解码器自回归生成输出。**输出长度由机器决定**(用 <EOS> token 终止)

## Key Concepts

- **查询 (Query Q)**: 主动询问"我要找什么"
- **键 (Key K)**: 被动索引"我能被怎么匹配"
- **值 (Value V)**: 实际内容
- **注意力权重**: softmax(评分) 后的归一化权重
- **缩放点积注意力**: `softmax(QKᵀ/√d)·V`,Transformer 默认
- **自注意力 (self-attention)**: Q/K/V 都来自同一序列
- **多头注意力 (multi-head)**: 多组注意力并行,捕捉不同子空间
- **位置编码 (positional encoding)**: 注入位置信息
- **掩蔽 (masking)**: 屏蔽未来位置(解码)或 padding
- **编码器-解码器注意力**: 解码器 Q,编码器 K/V -- 跨序列关注
- **前馈网络 (FFN)**: 两层 MLP,逐位置应用
- **LayerNorm (LN)**: 沿特征维归一化,适合序列
- **自回归 (autoregressive)**: 解码时逐步生成,每步用前一步输出
- **teacher forcing**: 训练时用真实前缀,推理时用模型输出
- **seq2seq**: 序列到序列模型
- **EOS (end of sequence)**: 终止符,告诉解码器停止

## Mental Models

- **注意力 = 加权平均**: 用 Q 与 K 的相似度当权重,对 V 加权平均
- **自注意力 = 序列内部"互相看"**: 每个位置问"我应该关注序列里哪些位置?"
- **多头 = 多视角**: 一组头看语法,一组头看语义,一组头看指代...
- **位置编码 = 给注意力补"顺序感"**: 自注意力本身没有位置概念,必须显式注入
- **Transformer = "全连接但权重可学"**: 每个位置都能关注所有位置(像全连接),但权重由 Q/K 相似度算(可学),不是固定参数
- **Transformer 可并行**: 不像 RNN 必须按时间步算,自注意力整个序列一次算完
- **LayerNorm 是序列的 BN**: 不依赖 batch,推理与训练一致

## Anti-patterns

- **注意力不除 √d**: d 大时点积大,softmax 进饱和区,梯度消失。**必须 `QKᵀ/√d`**
- **解码器不掩蔽**: 训练时解码器看到未来 token,推理时崩。**必须 mask 成下三角**
- **位置编码用可学但数据少**: 易过拟合;小数据用 sin/cos 固定编码
- **多头维度不整除**: d_model 必须能被 num_heads 整除
- **编码器-解码器注意力搞反**: 解码器是 Q(主动问),编码器是 K/V(被动答)。搞反会维度错
- **FFN 没有**: FFN 提供逐位置的非线性,光注意力不够
- **残差 + LN 顺序错**: 标准 Pre-LN(先 LN 再子层)或 Post-LN(先子层再 LN)选一种,不要乱
- **不调 warmup**: Transformer 初期梯度大,必须 warmup 否则发散

## Code Examples

**缩放点积注意力**:

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, num_heads, seq_q, d_k)
    K, V: (batch, num_heads, seq_kv, d_k)
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)  # 缩放!
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    return attn @ V, attn
```

**多头注意力**:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch = Q.size(0)
        # 投影 + 拆头: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        Q = self.W_q(Q).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 注意力
        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        # 合并头: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.num_heads * self.d_k)
        return self.W_o(out)
```

**位置编码 (sinusoidal)**:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

**Transformer 编码器块**:

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-LN: 先归一化再子层
        x = x + self.dropout(self.attn(self.ln1(x), self.ln1(x), self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
```
- **What it demonstrates**: 标准编码器块 = 多头自注意力 + FFN,各加残差 + LayerNorm。**注意自注意力 Q/K/V 都是同一个 x**

**PyTorch 内置 Transformer**:

```python
transformer = nn.Transformer(
    d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
    dim_feedforward=2048, dropout=0.1, batch_first=True
)
# src: (batch, src_len, d_model), tgt: (batch, tgt_len, d_model)
out = transformer(src, tgt, src_mask=..., tgt_mask=...)
```

## Reference Tables

**注意力变体对照**:

| 类型 | 评分函数 | 适用 |
|---|---|---|
| 加性注意力 | `vᵀ·tanh(W_q·q + W_k·k)` | Q/K 维度不同 |
| 点积注意力 | `qᵀk` | Q/K 同维度 |
| 缩放点积 | `qᵀk/√d` | Transformer 默认 |
| 双线性 | `qᵀW·k` | 部分场景 |

**Transformer 超参对照**:

| 超参 | 含义 | 典型值 |
|---|---|---|
| d_model | 模型维度 | 512 / 768 / 1024 |
| num_heads | 头数 | 8 / 12 / 16 |
| d_k = d_model / h | 每头维度 | 64 |
| d_ff | FFN 中间维度 | 4 × d_model |
| num_layers | 编码器/解码器层数 | 6 / 12 / 24 |
| dropout | 正则强度 | 0.1 |

**LayerNorm vs BatchNorm**:

| 维度 | BatchNorm | LayerNorm |
|---|---|---|
| 归一化方向 | 沿 batch 维 | 沿特征维 |
| 依赖 batch? | 是 | 否 |
| 训练/推理差异 | 有 | 无 |
| 适合序列? | 不适合 | 适合 |
| 适合 CV? | 是 | 也可(BN 更常见) |

## Worked Example

**为什么缩放点积要除 √d?** -- 数值稳定性推导。

设 q, k ∈ ℝ^d,各分量独立,均值 0、方差 1。点积 `q·k = Σᵢ qᵢ·kᵢ`,期望 0,方差:
$$\text{Var}(q·k) = \sum_i \text{Var}(q_i k_i) = \sum_i E[q_i^2]·E[k_i^2] = d$$

所以点积的标准差 ≈ √d。d=64 时点积标准差 8,softmax 进入饱和区(大值压小值几乎为 0)。

**除以 √d 把方差拉回 1**:
$$\text{Var}\left(\frac{q·k}{\sqrt{d}}\right) = 1$$

**关键洞察**:
1. 不缩放时 d 越大,softmax 越接近 one-hot(只关注最大值)
2. 梯度几乎全集中在最大值上,其他位置梯度→0(消失)
3. 除 √d 把数值稳定在 softmax 的"工作区"
4. 这就是 Transformer 论文为什么写 `softmax(QKᵀ/√d_k)` 而非 `softmax(QKᵀ)`

## Key Takeaways

1. 注意力机制 = Q/K/V 三元组,用 Q-K 相似度加权 V
2. 自注意力让序列中任意两位置直接交互,距离恒为 1
3. 缩放点积注意力 `softmax(QKᵀ/√d)V` 是 Transformer 默认
4. 多头注意力从不同子空间并行关注,类似多个独立的"注意力专家"
5. 位置编码注入位置信息(自注意力本身无序)
6. Transformer = 多头注意力 + FFN + 残差 + LayerNorm,可并行、长依赖友好
7. LayerNorm 适合序列(不依赖 batch size);Transformer 必须 warmup + AdamW
8. 编码器-解码器注意力做跨序列关注;解码器要 mask 未来位置

## Connects To

- **Ch 11**: Transformer 取代 RNN 处理序列,解决长依赖和并行问题
- **Ch 04**: 残差连接思想来自 ResNet,Transformer 也是深网标配
- **Ch 08**: LayerNorm 是 BatchNorm 在序列上的对应
- **Ch 14**: BERT/GPT 都是 Transformer 架构
- **Ch 23**: ChatGPT/大语言模型的核心就是 Transformer 堆到极大
