### Introduction
github上的开源项目[Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)聚焦于自动驾驶中的planning过程，设计了基于diffusion-transformer模型的planner，发现在闭环任务上表现良好，不依赖于基于规则的规划器
### Diffusion Model
#### Principle
扩散模型，又称扩散概率模型 (Diffusion Probabilistic Models, DPMs)，主要分两个过程：
1.前向扩散过程 (Forward Diffusion Process)：基于Markov Model，每一个时间步$t$,基于分布$p(x_t|x_{t-1})$生成$x_t$,并且随着$t$增大，概率分布越接近高斯分布$N(0,I)$
2.反向去噪过程 (Reverse Denoising Process)：逆转前向扩散过程，每个时间步$t$,基于可学习的带参数$\theta$分布$p_ \theta (x_{t-1}|x_t)$生成$x_{t-1}$,
模型训练过程：
损失函数：$L = E_{t \sim [1, T], x_0 \sim q(x_0), \epsilon \sim N(0, I)} [||\epsilon - \epsilon_\theta(x_t, t)||^2]$，其中$\epsilon_\theta(x_t,t)$是模型预测噪声，$\epsilon$是真实噪声
条件信息：可以将条件信息嵌入噪声预测器，作为引导机制
模型生成过程：
1.采集噪声：从高斯噪声分布中随机采集样本$x_t$
2.逐步去噪：根据训练出的$\epsilon_\theta(x_t,t)$估计噪声，再逐步还原为初始样本$x_0$
#### Advantages
论文核心亮点就是把扩散模型能处理复杂概率分布且具有引导机制的特性迁移到planning任务，能处理自车信息和其他车辆信息，并通过引导机制让planning更与人类习惯相似
### Model architecture of Diffusion Planner
<img src="res\diffusion_planner1.png" alt="核心架构" width="800" />

#### MLP-Mixer
用于从多维度信息提取关键特征信息，先将输入数据分割为相同的patches，再进入两个过程：
1.Channel-Mixing MLP:作用在特征维度，每个patch内部信息进行MLP
2.Token-Mixing MLP:作用在空间维度，在相同空间位置上的所有特征进行MLP
数学过程：
记输入数据$S$, $S=S+{MLP(S^T)}^T$, $S=S+MLP(S)$
#### Encorder
这部分输入邻居车辆、车道、静态物品信息先分别通过embedding层，再分别通过MLP-Mixer,MLP-Mixer,MLP层，最后进入self-attention层进行多信息聚合，得到输出$Q_f$
#### navigation information
输入导航信息，通过MLP-Mixer输出$Q_n$
#### 主干网络 (Diffusion Transformer)
多个transformer块堆叠而成，主要有以下几个模块
1.多头自注意力 (Multi-Head Attention)：整合序列内部信息
2.前馈网络 (FFN):对输入非线性映射
3.自适应层归一化 (Adaptive Layer Norm - LN)：Scale Shift块将$Q_t+Q_n$注入transformer模块，即归一化层参数会被当前扩散时间步信息和导航信息影响
4.多头交叉注意力 (Multi-Head Cross-Attention - MHCA)：融合$Q_f$和当前时间步轨迹$x$
$x=x+MHCA(x,Q_f)$, $x=x+FFN(x)$
所以主干网络(DiT)完成的过程就是一个反向去噪的过程：输入当前时间步轨迹信息$x_t$,先进行多头自注意力，中间还会和输入的导航信息融合，再进行以轨迹信息$x$为$Q$,以$Q_f$为$K,V$的多头交叉注意力,最后得到还原出的去噪轨迹$x_{t-1}$
### Conclusion
1.闭环任务，也就是planner的设计还有待研究，可以结合不同的网络架构，发掘架构特性和闭环任务的相似之处
2.这里扩散模型的引导机制能引导规划轨迹成为喜爱的样子，落地时可以考虑加一个在人类驾驶时实时采集数据并分析的网络，然后开启自驾时引导规划更符合车主驾驶习惯
