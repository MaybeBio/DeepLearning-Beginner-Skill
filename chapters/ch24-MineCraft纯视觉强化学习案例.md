# Chapter 24: MineCraft 纯视觉强化学习案例 (LS-Imagine)

## Core Idea
**LS-Imagine** (ICLR 2025 Oral) 用纯视觉观测玩 MineCraft,单卡 3090 训练,不开外挂不使用特权信息。核心创新: 构建**长短期世界模型 (Long Short-Term World Model)**,既能做**即时状态转换**(相邻步),又能做**跳跃式状态转换**(直接跳到目标相关未来状态),避免"短视"问题。**功用性图 (Affordance Map)** 是关键--对单帧图像做滑动窗口放大模拟探索过程,用 MineCLIP 评估与任务描述的相关性,生成任务相关区域热力图。功用性图驱动**内在奖励**(激励靠近目标)和**跳跃标志**(功用性图峰度高 -> 目标远 -> 用跳跃转换)。在 5 个 MineCraft 任务上全面超越 DreamerV3/VPT/STEVE-1 等方法,"收集原木"成功率 80.63%(DreamerV3 53.33%)。

## Frameworks Introduced

- **功用性图 (Affordance Map) (LS-Imagine)**
  - When to use: 无目标位置先验,纯视觉定位目标
  - How:
    1. 滑动边界框从左到右、从上到下扫描整张观察图
    2. 每位置从原图裁剪出 16 张逐渐放大的图像(模拟靠近该区域的视觉变化)
    3. 用预训练 MineCLIP 计算放大序列与任务描述的**相关性**
    4. 融合所有位置相关性值 -> 完整功用性图(热力图)
  - **加速**: 用滑动窗口标注的数据训**多模态 Swin-UNet**,每步直接生成功用性图
  - 关键: 即使目标被遮挡/不可见(矿石在地下、村庄被山挡),功用性图仍指示探索方向

- **功用性驱动内在奖励 (Affordance-driven Intrinsic Reward) (LS-Imagine)**
  - When to use: 激励智能体靠近目标
  - How:
    - 功用性图 M_t 与同尺寸**二维高斯矩阵**(中心高四周低)逐元素相乘取均值
    - 含义: 目标在视角中心时奖励大--鼓励对齐目标到视野中心
  - 关键: 内在奖励 = 任务相关的探索奖励,不依赖外部环境奖励

- **跳跃标志 (Jumping Flag) (LS-Imagine)**
  - When to use: 判断何时用跳跃式状态转换
  - How:
    - 计算功用性图**峰度 (kurtosis)**
    - 高价值区域**高度集中**(目标远且小) -> 峰度飙升 -> 启用跳跃转换
    - 目标近(大)时无跳转,正常逐步接近
  - 关键: 远距离目标用跳跃一步到位,近距离逐步接近--自适应

- **长短期世界模型 (Long Short-Term World Model) (LS-Imagine)**
  - When to use: 开放式世界的有模型 RL
  - How:
    - **短期转换分支**: 输入当前状态+动作 -> 预测下一相邻状态(标准)
    - **长期转换分支**: 输入当前状态 -> 直接预测目标相关未来状态 s_{t+H}(跳跃)
    - **跳跃预测器 (Jump predictor)**: 根据当前状态决定用短期还是长期转换
    - **间隔预测器 (Interval predictor)**: 估计跳跃前后的时间步数 Δ̂_t 和累计折扣奖励 Ĝ_t
    - **编码器**: 融合功用性图 M_t 提供目标先验
  - 训练数据:
    - 短期: 相邻时间步样本对(真实交互)
    - 长期: 根据功用性图建模的跨越时间间隔样本对
  - 关键: 跳跃式转换绕过中间状态,一步到目标--解决长期探索"短视"

- **长短期想象行为学习 (LS-Imagine Actor-Critic)**
  - When to use: 在混合序列上优化策略
  - How:
    - 从当前状态出发,动态选择短期/长期转换,生成**想象序列**(长度 L)
    - 每条想象序列 = 即时转换 + 跳跃转换的混合
    - 改进的 **bootstrap λ-returns** 结合长短期想象:
      ```
      R_t^λ = ĉ_t { Ĝ_{t+1} + γ^{Δ̂_{t+1}}[(1-λ)v_ψ(ŝ_{t+1}) + λR_{t+1}^λ] }
      ```
    - 用 Actor-Critic 算法优化策略
  - 关键: 跳跃转换直接估计长期回报,而非逐步累积--改善长期规划

- **MineCLIP 预训练模型 (LS-Imagine)**
  - When to use: 评估视频与任务文本的相关性
  - How: MineCLIP 是 Minecraft 视频-语言预训练模型,类似 CLIP
    - 输入: 视频帧序列 + 任务描述
    - 输出: 相关性分数
  - 关键: 大量专家示范视频预训练,即使目标不可见也能提供有效探索指导

## Key Concepts

- **LS-Imagine**: 长短期想象开放世界 RL
- **功用性图 (affordance map)**: 任务相关区域热力图
- **功用性驱动内在奖励**: 激励靠近目标的内在奖励
- **跳跃式状态转换 (jump state transition)**: 一步到目标未来状态
- **即时状态转换 (instant transition)**: 标准相邻步预测
- **跳跃标志 (jumping flag)**: 峰度判断是否用跳跃
- **长短期世界模型**: 短期 + 长期双分支
- **跳跃预测器 (jump predictor)**: 选择转换类型
- **间隔预测器 (interval predictor)**: 估计跳跃步数和折扣奖励
- **短期想象序列**: 逐步想象的隐状态序列
- **长短期想象序列**: 即时 + 跳跃混合序列
- **bootstrap λ-returns**: 改进的 TD(λ) 回报估计
- **MineCLIP**: Minecraft 视频语言预训练模型
- **Swin-UNet**: 多模态 U-Net 生成功用性图
- **峰度 (kurtosis)**: 衡量分布集中度,判断目标距离
- **Minecraft**: 典型开放世界游戏
- **视觉强化学习 (VRL)**: 纯粹从像素学习策略
- **有模型 RL (MBRL)**: 学习世界模型辅助决策
- **DreamerV3**: Dreamer 系列世界模型 baseline
- **VPT**: Video PreTraining,MineCraft 行为克隆 baseline
- **STEVE-1**: MineCraft 文生动作模型
- **DECKARD**: 无模型 RL MineCraft 方法

## Mental Models

- **功用性图 = 探索热力图**: 类似显著图,但指示"去哪里探索",不是"哪里重要"
- **滑动窗口放大 = 虚拟探索**: 模拟走近目标区域的视频序列
- **跳跃转换 = 虫洞**: 绕过中间状态,一步到远处--像科幻的虫洞跳跃
- **峰度 = 目标距离**: 功用性图越集中(峰度越高),目标越远,越需要跳跃
- **长期/短期 = 战略/战术**: 跳跃是战略(去哪里),即时是战术(怎么走)
- **MineCLIP = 常识**: 从海量专家视频学到的"玩游戏的本能判断"
- **Ĝ = 累计奖励快进**: 跳过中间步,直接估计一段路径的总奖励
- **混合想象 = 快慢结合**: 慢走(短期) + 快跳(长期)在同一个想象序列里

## Anti-patterns

- **只用短期想象**: 短视,看不到远处目标。**加跳跃式转换**
- **只用跳跃不用短期**: 近距离操作不精确。**自适应选择(峰度)**
- **滑动窗口直接实时算功用性图**: 太慢。**用 Swin-UNet 加速**
- **目标被遮挡时放弃功用性图**: 仍可工作。**MineCLIP 从大量专家视频学到常识**
- **世界模型不分长短:期分支**: 跳跃会破坏模型一致性。**两分支独立**
- **忽略功用性图作为编码器先验**: 减少目标指引。**功用性图送入编码器**
- **所有任务手写奖励**: 泛化差。**用功用性驱动内在奖励(零样本)**
- **只比较标准 VRL baseline**: DreamerV3 仍是强 baseline。**对比 VPT/STEVE-1 等**

## Code Examples

**功用性图生成(Swin-UNet 快速版)**:

```python
class AffordanceUNet(nn.Module):
    """多模态 Swin-UNet 生成功用性图
    输入: 视觉观测 + 语言指令
    输出: 功用性图(单通道热力图)
    """
    def __init__(self, img_size=224, text_dim=512):
        super().__init__()
        self.swin_unet = SwinUNet(img_size=img_size, in_ch=3, out_ch=1)
        self.text_proj = nn.Linear(text_dim, 256)  # 语言嵌入投影
        self.text_cross = CrossAttention(256)
    
    def forward(self, obs_img, text_emb):
        # obs_img: (B, 3, H, W), text_emb: (B, 512, text_dim)
        feat = self.swin_unet.encode(obs_img)
        # 交叉注意力融合文本
        text_feat = self.text_proj(text_emb[:, 0])  # (B, 256)
        feat = self.text_cross(feat, text_feat)
        affordance = self.swin_unet.decode(feat)  # (B, 1, H, W)
        return affordance.sigmoid()

# 训练数据由滑动窗口 + MineCLIP 标注(离线预生成)
aff_unet = AffordanceUNet()
for obs_batch, text_batch, afford_gt in affordance_dataloader:
    pred = aff_unet(obs_batch, text_batch)
    loss = F.mse_loss(pred, afford_gt)
    loss.backward(); optimizer.step()
```

**长短期世界模型**:

```python
class LSWorldModel(nn.Module):
    def __init__(self, latent_dim=256, action_dim=10, hidden=512):
        super().__init__()
        # 编码器(含功用性图输入)
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + 1, 64, 3, stride=2), nn.ReLU(),  # RGB + affordance
            nn.Conv2d(64, 128, 3, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(128 * 55 * 55, latent_dim)
        )
        # 短期转换(即时)
        self.short_trans = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
        # 长期转换(跳跃)
        self.long_trans = nn.Linear(latent_dim, latent_dim)  # 直接跳跃
        # 跳跃预测器
        self.jump_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()  # ŷ_t: 是否跳跃
        )
        # 间隔预测器(跳跃时用)
        self.interval_predictor = nn.Linear(latent_dim, 2)  # (Δ̂, Ĝ)
        
        # 奖励/继续预测器
        self.reward_head = nn.Linear(latent_dim, 1)
        self.continue_head = nn.Linear(latent_dim, 1)
    
    def forward(self, obs, action, affordance, mode='dynamic'):
        """mode: 'short', 'long', 'dynamic'"""
        s = self.encoder(torch.cat([obs, affordance.unsqueeze(1)], dim=1))
        
        if mode == 'dynamic':
            jump_prob = self.jump_predictor(s)
            jump_flag = (jump_prob > 0.5).float()
            # 选择短期或长期转换
            s_next = (1 - jump_flag) * self.short_trans(torch.cat([s, action], dim=1))
            s_next += jump_flag * self.long_trans(s)
            delta_G = self.interval_predictor(s)  # (Δ̂, Ĝ)
        elif mode == 'short':
            s_next = self.short_trans(torch.cat([s, action], dim=1))
            delta_G = None
        else:  # 'long'
            s_next = self.long_trans(s)
            delta_G = self.interval_predictor(s)
        
        r = self.reward_head(s_next)
        cont = self.continue_head(s_next).sigmoid()
        return s_next, r, cont, delta_G
```

**长短期想象 + Actor-Critic**:

```python
def imagine_rollout(world_model, actor, critic, s_0, L=16, gamma=0.99, lam=0.95):
    """生成长短期混合想象序列,计算 λ-returns"""
    s_t = s_0
    rewards, continues, values, deltas, gammas = [], [], [], [], []
    
    for t in range(L):
        # 动态选择转换类型
        jump_prob = world_model.jump_predictor(s_t)
        jump_flag = (jump_prob > 0.5).float()
        
        action = actor(s_t).sample()  # 策略采样
        s_next, r, cont, delta_G = world_model.forward_dynamic(s_t, action)
        
        if jump_flag and delta_G is not None:
            # 跳跃转换: 直接用 Ĝ 作为这段的累积奖励
            effective_r = delta_G[:, 1]  # Ĝ
            effective_gamma = gamma ** delta_G[:, 0]  # γ^Δ̂
        else:
            # 短期转换: 正常单步
            effective_r = r.squeeze(-1)
            effective_gamma = gamma
        
        value = critic(s_t)
        
        rewards.append(effective_r)
        continues.append(cont.squeeze(-1))
        values.append(value)
        gammas.append(effective_gamma)
        
        s_t = s_next
    
    # Bootstrap λ-returns
    R = critic(s_t)
    returns = []
    for t in reversed(range(L)):
        R = rewards[t] + continues[t] * gammas[t] * (
            (1 - lam) * values[t] + lam * R
        )
        returns.insert(0, R)
    return returns, values
```

## Reference Tables

**LS-Imagine vs 其他方法(5 任务成功率)**:

| 模型 | 收集原木 | 收集水 | 采集沙子 | 剪羊毛 | 开采铁矿石 |
|---|---|---|---|---|---|
| VPT | 6.97% | 0.61% | 12.99% | 1.94% | 0.00% |
| STEVE-1 | 57.00% | 6.00% | 37.00% | 3.00% | 0.00% |
| PTGM | 41.86% | 2.78% | 17.71% | 21.54% | 15.14% |
| Director | 8.67% | 20.90% | 36.36% | 1.27% | 7.82% |
| DreamerV3 | 53.33% | 55.72% | 59.88% | 25.13% | 16.79% |
| **LS-Imagine** | **80.63%** | **77.31%** | **62.68%** | **54.28%** | **20.28%** |

**Minecraft VRL 方法分类**:

| 方法 | 类型 | 观测 | 世界模型 | 探索 |
|---|---|---|---|---|
| VPT | 行为克隆 | 视觉 | 无 | 模仿 |
| STEVE-1 | 文到动作 | 视觉 | 无 | 模仿 |
| DreamerV3 | MBRL | 视觉 | 短期 | 内在奖励 |
| DECKARD | MFRL | 视觉 | 无 | 试错 |
| **LS-Imagine** | **MBRL** | **视觉+功用性图** | **长短期** | **功用性驱动** |

**世界模型转换类型对比**:

| 类型 | 步数 | 状态预测 | 奖励估计 | 何时用 |
|---|---|---|---|---|
| 短期(即时) | 1 步 | 精确 | 单步 r | 目标近,精细操作 |
| 长期(跳跃) | H 步 | 近似(目标状态) | Ĝ(累积) | 目标远,峰度高 |

**功用性图计算二阶段**:

| 阶段 | 方法 | 速度 | 用途 |
|---|---|---|---|
| 离线标注 | 滑动窗口+MineCLIP | 慢 | 生成训练数据 |
| 在线推理 | 多模态 Swin-UNet | 快(实时) | 每步生成功用性图 |

## Worked Example

**为什么 DreamerV3 在 MineCraft 中"短视"?** -- 收集原木任务案例。

**任务**: 在 Minecraft 平原上"砍一棵树"获取原木。智能体出生在随机位置,树木可能很远。

**DreamerV3 方法**:
1. 世界模型: 输入当前观测+动作,预测下一帧(短期 1 步)
2. 在世界模型生成的短序列上优化策略
3. **问题**: 想象序列长度有限(典型 15-50 步)
   - 树在 200 步外时,短序列想象中看不到树
   - 看不到树 = 没有砍树奖励 = 智能体随机游走
   - 只能靠随机探索偶然碰到树,样本效率极低

**LS-Imagine 方法**:
1. **功用性图生成**: 
   - 滑动窗口放大模拟"走向各处"的视觉变化
   - MineCLIP 评估: 哪个方向"像在砍树"(即使树被遮挡)
   - 生成热力图,高亮树木可能方向
2. **跳跃标志**: 功用性图峰度高(目标区域小且集中) -> 目标远
3. **跳跃式状态转换**: 
   - 世界模型直接预测"走到树旁边"时的状态(跳跃 200 步)
   - 间隔预测器估计: Δ̂ ≈ 200, Ĝ ≈ 砍树奖励
4. **混合想象**: 
   - 跳跃到树旁 -> 短期逐步靠近 -> 砍树
   - 在想象序列中已经看到奖励,优化策略有方向

**实验数据**(收集原木,1000 步内):
- DreamerV3: 成功率 53.33%, 平均步数 711
- STEVE-1(行为克隆): 成功率 57.00%, 752 步
- **LS-Imagine: 成功率 80.63%, 平均步数 503**

**为什么目标被遮挡仍能工作?**
- 寻路村庄: 村庄被山挡住,但功用性图指示"向右走"或"向左上山"(MineCLIP 从专家视频学到"村庄常出现在开阔处")
- 挖矿: 矿石在地下,不可见,但功用性图指示"向山体内挖"(MineCLIP 学到"矿石在石头里")
- **关键**: MineCLIP 预训练视频含有"从无到有发现目标"的过程

**为什么 LS-Imagine 在开采铁矿石上优势不大?**
- 铁矿石成功率: LS-Imagine 20.28% vs DreamerV3 16.79%(差距小)
- 原因: 铁矿石在地下深处,需多步挖掘,即使跳跃到矿石上方,仍需精确挖掘
- 开采铁矿石更像密集操作而非探索--LS-Imagine 最大优势在**探索密集型任务**
- 上限被 Minecraft 的硬核挖掘机制限制

**关键洞察**:
1. **核心矛盾**: MBRL 的短想象 vs 开放世界的长距离--LS-Imagine 用跳跃解决
2. **功用性图 = 零样本探索指导**: 不需要任务特定的奖励设计
3. **自适应跳跃**: 目标远用跳跃,目标近用短期--不是一刀切
4. **预训练知识注入**: MineCLIP 预训练知识让功用性图即使在遮挡下也能工作
5. **单卡 3090 可行**: 相比需要万卡的方法(VPT),LS-Imagine 成本低
6. **长短期 = 分层**: 战略(跳跃去哪) + 战术(逐步怎么走)在同一框架

## Key Takeaways

1. LS-Imagine 用**纯视觉**玩 MineCraft,单卡 3090,不开外挂--视觉 RL 里程碑
2. 核心矛盾: 标准世界模型短想象 vs 开放世界长距离探索
3. **功用性图 (Affordance Map)**: 滑动窗口放大 + MineCLIP 相关性 = 任务热力图
4. **功用性驱动内在奖励**: 激励智能体靠近目标,无需任务特定奖励设计
5. **跳跃标志**(峰度): 功用性图越集中 = 目标越远 = 用跳跃转换
6. **长短期世界模型**: 短期转换(即时 1 步) + 长期转换(跳跃 H 步),双分支
7. **跳跃预测器** + **间隔预测器**: 自适应选择转换类型 + 估计 Δ̂ 和 Ĝ
8. **混合想象序列**: Actor-Critic 在即时+跳跃混合序列上优化
9. 改进的 **bootstrap λ-returns**: 结合跳跃转换的长期回报估计
10. 5 个 MineCraft 任务全面超 DreamerV3/VPT/STEVE-1,探索密集型任务优势巨大
11. 功用性图即使目标被遮挡(矿石地下、村庄被山挡)仍指引方向
12. MineCLIP 预训练知识让功用性图有"常识",非纯粹视觉检测

## Connects To

- **Ch 20**: Actor-Critic 架构是 LS-Imagine 的行为学习基础
- **Ch 13**: 视觉 RL 是 CV(图像观测) + RL(决策) 的结合
- **Ch 16**: 世界模型"想象未来"类似扩散模型从噪声逐步去噪
- **Ch 23**: MineCLIP 是预训练模型(类似 CLIP/BERT),知识迁移到 RL
- **Ch 17**: MineCLIP 是自监督/对比学习训练的视频-语言模型
- **Ch 14**: MineCLIP 文本任务描述是 NLP 在 RL 中的应用
