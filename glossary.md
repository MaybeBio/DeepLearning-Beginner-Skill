# Glossary

**激活函数 (activation function)** — 引入非线性的函数(ReLU/Sigmoid/GELU) (Ch 6)

**Actor-Critic** — 策略网络(Actor)+ 价值网络(Critic)联合训练的RL方法 (Ch 20)

**优势函数 (advantage A)** — 动作比平均水平好多少:`A = Q(s,a) - V(s)` (Ch 20)

**对抗攻击 (adversarial attack)** — 加微小噪声骗网络(白盒/黑盒,FGSM) (Ch 18)

**对抗训练 (adversarial training)** — 用对抗样本数据增强训练,提升鲁棒性 (Ch 18)

**功用性图 (affordance map)** — 任务相关区域热力图,驱动视觉RL探索 (Ch 24)

**代理网络 (proxy network)** — 模仿黑盒目标的替代模型,用于黑盒攻击 (Ch 18)

**自编码器 (autoencoder)** — 编码器-解码器 + 重构损失,无监督学特征 (Ch 17)

**自回归 (autoregressive)** — 逐词生成,每步输出作为下一步输入(GPT核心) (Ch 23)

**批量归一化 (Batch Normalization)** — 层间标准化,加速训练,减少内部协变量偏移 (Ch 7)

**BERT** — Transformer编码器 + MLM(填空),双向,适合理解任务 (Ch 17)

**瓶颈 (bottleneck)** — 自编码器中低维中间层,强迫压缩表示 (Ch 17)

**灾难性遗忘 (catastrophic forgetting)** — 学新任务后旧任务表现暴跌 (Ch 21)

**ChatGPT** — GPT + 监督微调 + RLHF 训练的对话大模型 (Ch 23)

**分类 (classification)** — 输出类别标签(二分类/多分类/多标签) (Ch 1)

**CNN** — 卷积神经网络,参数共享+局部连接,图像处理标配 (Ch 6)

**上下文/语境化嵌入 (contextualized embedding)** — 同词不同上下文不同向量(BERT核心) (Ch 17)

**卷积层 (convolutional layer)** — 滤波器滑过输入提取局部特征 (Ch 6)

**代价/损失函数 (cost/loss function)** — 衡量预测与真实差距(交叉熵/MSE) (Ch 5)

**临界点 (critical point)** — 梯度为零处(局部极小/极大/鞍点) (Ch 8)

**数据增强 (data augmentation)** — 旋转/翻转/裁剪增广训练数据 (Ch 13)

**DDPM** — 加噪-去噪扩散生成模型,Stable Diffusion 基础 (Ch 16)

**判别器 (discriminator D)** — GAN中区分真假数据的网络 (Ch 15)

**领域自适应 (domain adaptation)** — 源领域训的模型适配目标领域 (Ch 19)

**领域泛化 (domain generalization)** — 训练多领域数据,泛化到未知领域 (Ch 19)

**领域偏移 (domain shift)** — 训练/测试分布不同导致性能下降 (Ch 19)

**DQN** — 深度Q网络,用Critic直接选动作的RL方法 (Ch 20)

**Dropout** — 随机删神经元防过拟合,训练时随机失活 (Ch 7)

**动态计算 (dynamic computation)** — 同一网络按资源调整深度/宽度 (Ch 22)

**嵌入 (embedding)** — 数据在低维空间的稠密向量表示 (Ch 17)

**探索 (exploration)** — RL中主动尝试新动作(否则好坏未知) (Ch 20)

**FGSM (Fast Gradient Sign Method)** — 快速梯度符号法,一次迭代生成对抗样本 (Ch 18)

**FID (Frechet Inception Distance)** — 衡量生成图与真实图分布距离(越小越好) (Ch 15)

**微调 (fine-tuning)** — 预训练模型在目标任务上用少量数据继续训练 (Ch 13, 17, 19)

**基石模型 (foundation model)** — 预训练自监督模型,可微调到各种下游任务 (Ch 23)

**GAN (Generative Adversarial Network)** — G(生成器) + D(判别器)对抗训练 (Ch 15)

**GPT** — Transformer解码器 + 自回归,适合文本生成 (Ch 17, 23)

**梯度下降 (gradient descent)** — 沿负梯度方向更新参数最小化损失 (Ch 5)

**梯度消失/爆炸 (vanishing/exploding gradient)** — RNN深时梯度趋零或无穷 (Ch 11)

**GPU** — 并行矩阵乘法加速深度学习训练推理 (Ch 4)

**GRU** — 门控循环单元,简化版LSTM(更新门+重置门) (Ch 11)

**幻觉 (hallucination)** — LLM生成看似合理但事实错误的内容 (Ch 23)

**超参数 (hyperparameter)** — 人工设定的参数(学习率/批大小/层数) (Ch 5)

**Inception Score (IS)** — 衡量GAN生成质量+多样性(越大越好) (Ch 15)

**In-context learning** — GPT通过prompt上下文学任务,无梯度更新 (Ch 17, 23)

**知识蒸馏 (knowledge distillation)** — 教师网络(大)教学生网络(小) (Ch 22)

**学习率 (learning rate)** — 梯度下降步长,过大震荡过小收敛慢 (Ch 5, 8)

**LIME** — 局部线性模型解释黑盒,在样本邻域拟合线性 (Ch 22)

**Lipschitz** — 梯度范数≤1,保证WGAN收敛的判别器平滑性 (Ch 15)

**LLM (大语言模型)** — 大规模自回归语言模型(GPT/ChatGPT) (Ch 23)

**LSTM** — 长短期记忆网络,三控门(输入/遗忘/输出)解决长期依赖 (Ch 11)

**MAML** — 模型不可知元学习,学好的初始化参数 (Ch 21)

**MLM (掩蔽语言模型)** — BERT填空任务,15%词元掩蔽后预测 (Ch 17)

**模式崩塌 (mode collapse)** — GAN生成器只会几种输出,丧失多样性 (Ch 15)

**动量 (momentum)** — 保留历史梯度方向,加速收敛+跳过鞍点 (Ch 8)

**多任务学习 (multi-task learning)** — 多个任务数据一起训,正则是终身学习上界 (Ch 21)

**NAS (神经网络架构搜索)** — 自动搜索最优网络架构(RL/进化/DARTS) (Ch 21)

**噪声预测器 (noise predictor)** — DDPM核心:预测图里加了什么噪声 (Ch 16)

**N-way K-shot** — N类每类K样例的小样本学习任务格式 (Ch 21)

**过拟合 (overfitting)** — 训练集好测试集差,模型记噪声非规律 (Ch 7)

**池化层 (pooling layer)** — 降采样,减小特征图尺寸,提鲁棒性(最大/平均) (Ch 6)

**预训练 (pre-training)** — 海量无标注数据自监督学通用表示 (Ch 17, 23)

**探针 (probing)** — 用辅助分类器探查网络某层嵌入含什么信息 (Ch 22)

**Rainbow** — DQN 7种改进合集(Double/Dueling/Prioritized/Multi-step等) (Ch 20)

**强化学习 (RL)** — agent与环境互动获取奖励,学最大化累积回报 (Ch 20)

**RLHF (Reinforcement Learning from Human Feedback)** — 人类反馈强化学习,ChatGPT训练阶段三 (Ch 23)

**回归 (regression)** — 输出连续数值(房价/温度预测) (Ch 1)

**正则化 (regularization)** — 防过拟合技术(Dropout/BN/weight decay/early stop等) (Ch 7)

**ResNet** — 残差网络,跳跃连接解决深层网络退化 (Ch 9)

**RNN** — 循环神经网络,处理序列数据,隐藏状态传递历史信息 (Ch 11)

**显著图 (saliency map)** — 像素重要性热力图,∂e/∂x 计算 (Ch 22)

**自注意力 (self-attention)** — 序列内元素两两交互,全局依赖,Transformer核心 (Ch 12)

**自监督学习 (SSL)** — 数据自身构造标签(无人工标注)预训练模型 (Ch 17)

**Seq2Seq** — 编码器-解码器结构,输入序列映射到输出序列 (Ch 11)

**Skip-Gram** — word2vec之一,用中心词预测周围词 (Ch 14)

**Softmax** — 指数归一化,分类输出概率分布;温度T平滑 (Ch 5, 22)

**Stable Diffusion** — 潜在扩散模型,在VAE潜在空间做扩散(降计算量) (Ch 16)

**TD (时序差分)** — RL Critic训练:单步`V(s_t)=r_t+γV(s_{t+1})` (Ch 20)

**Transformer** — 纯注意力结构(编码器+解码器),并行处理,现代NLP基石 (Ch 12)

**迁移学习 (transfer learning)** — 源任务知识迁移到目标任务 (Ch 19)

**VQ-VAE** — 离散码本嵌入,可学出基本单元(音素/笔画) (Ch 17)

**WGAN** — Wasserstein GAN,用推土机距离替代JS散度 (Ch 15)

**word2vec** — 浅层窗口预测学词嵌入(Skip-Gram/CBOW) (Ch 14)

**XAI (可解释性AI)** — 解释模型决策理由:局部/全局,LIME/显著图/探针 (Ch 22)
