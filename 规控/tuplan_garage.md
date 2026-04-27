### Introduction
github上的开源项目[tuplan_garage](https://github.com/autonomousvision/nuplan_garage)主要解决了了基于学习和基于规则两种规划器在nuplan数据集上评估结果的利弊，发现基于学习的规划器在开环任务上表现更好，而基于规则的规划器在闭环任务上表现更好，所以该项目提出了混合规划器，解决了这一平衡问题
### Dataset
训练和仿真评估都在nuplan数据集上进行<br>
nuplan数据集主要用于自动驾驶中轨迹规划，总共有1300小时的真实驾驶数据，主要包括地图数据、场景数据(传感器数据和标注)，每个场景通常为15秒的模拟片段
#### Stucture
nuplan/  
├── maps/               # 高精地图数据（各城市的车道、交通设施等）  
├── scenes/             # 场景数据（传感器数据+标注）  
│   ├── train/          # 训练集场景（15万+）  
│   ├── val/            # 验证集场景（1.4万+）  
│   └── test/           # 测试集场景（1.4万+，标注不公开）  
├── logs/               # 原始采集日志（传感器校准、车辆状态等）  
└── metrics/            # 评估指标计算工具及参考结果  
#### Data Types
1.传感器数据 sensors：摄像头、雷达的探测数据，以及自车运动参数<br>
2.场景标注 annotations：agents的类型、标注内容(主要是3D边框、速度、加速度、轨迹)，规划目标、交通规则<br>
3.高清地图 maps：车道层，交通设施，可行驶区域<br>
4.元数据 metadata：场景信息(时间、地点、天气)，数据质量(传感器故障标记、场景难度等级)<br>
#### Evaluation Citeration
论文采用的是nuplan的三个评估指标：<br>
1.开环分数(Open-Loop Score, OLS)：衡量自车预测的准确性，预测未来长时间的轨迹，反映预测轨迹与人类驾驶轨迹的偏差，包括位移误差和航向误差<br>
2.闭环分数非反应式(Closed-Loop Score Non-Reactive, CLS-NR)：评估规划器实际驾驶性能，但agents的动作按原始片段回放。子分数包括速度限制遵守、时间到碰撞、进程、舒适度；惩罚包括有责任碰撞、驾驶区域违规等<br>
3.闭环分数反应式(Closed-Loop Score Reactive, CLS-R)：评估规划器实际驾驶性能，但agents通过一个智能规划器(文中的IDM)作出反应，更接近真实世界。子分数包括速度限制遵守、时间到碰撞、进程、舒适度；惩罚包括有责任碰撞、驾驶区域违规等<br>
### Models
#### PDM_Open(Predictive Driver Model Open)
输入：<br>
Centerline ($c$)：IDM提取的中心线<br>
Ego History ($h$)：过去两秒内自车信息，包含位置、速度、加速度<br>
c和h都被线性投影到512维向量，再连接起来<br>
网络(MLP-$\varphi_{Open}$)：采用两个512维隐藏层的MLP<br>
输出：<br>
$w_{Open}$:未来路径点，预测8秒轨迹，两点间隔0.5秒,$w_{Open}=\varphi_{Open}(c,h)$<br>
#### PDM_Closed(Predictive Driver Model Closed)
模型输入当前场景信息(当前的车辆状态、地图信息、其他智能体状态以及全局路线信息),输出预测轨迹<br>
核心思想借鉴了MPC(模型控制预测)，是一个基于规则的规划器，核心步骤为Select Centerline (选择中心线)、Forecast the Environment (预测环境)、Creates Varying Trajectory Proposals (创建不同的轨迹提议)、Simulated (模拟)、Scored for Trajectory Selection (评分以选择轨迹)<br>
<img src="res\tuplan_garage1.png" alt="核心架构" width="600" />
#### PDM_Hybrid(Predictive Driver Model Hybrid)
模型包括PDM_Closed和PDM_Offset两大部分，PDM_Closed基于规则，PDM_Offset基于学习，所以模型是基于学习和规则的混合规划器<br>
1.PDM_Closed<br>
2.PDM_Offset：输入PDM_Closed模块产生的预测轨迹、中心线以及自车历史信息，输出修正后的轨迹,模型架构采用PDM_Open，发挥了PDM_Open在开环任务长时间轨迹规划的优势，只对长距离路径点进行修正<br>
总体来说，PDM_Closed是主要决策者，而PDM_Offset是一个修正器，主要修正长期路径规划<br>
### Conclusion
1.自动驾驶中规控部分主要分为规划层(Planning)和控制层(Control)，论文提出的PDM_Hybrid正是规划层部分，进行闭环任务时再加上控制器(如论文中LQR控制器)，而核心部分其实是规划层的设计<br>
2.规划层主要根据输入的场景信息和自车状态信息，预测规划轨迹<br>