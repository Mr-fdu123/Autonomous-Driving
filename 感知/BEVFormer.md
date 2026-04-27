### Introduction
github上的开源项目[BEVFormer](https://github.com/fundamentalvision/BEVFormer),聚焦于自动驾驶在感知层上的研究，提出了一种由多摄像头实时采集的多视角图像转化为实时的鸟瞰图的新架构，称之为BEVFormer<br>
### Dataset
论文采用了nuScenes数据集，包含了1000个场景，每个场景为约20秒的视频，总共约 1.4M 个3D边界框标注(有10个物体类别)。nuScenes数据集常用于自动驾驶中环境感知任务中,总体架构是[Scene 场景->Sample 样本帧->Sample Data 传感器数据->Sample Annotation 标注]的层级结构，具体为Scene -> Sample -> Sample Data + Sample Annotation<br>
数据集包含图像、激光雷达、雷达、GPS/IMU 数据时空对齐，支持跨传感器任务
#### Scene(scene.json)
token, name, description(如晴天、有行人横穿), frist_sample_token/last_sample_token, nbr_samples(包含的样本帧数量，约40-60帧)
#### Sample(sample.json)
token， timestamp(毫秒级)， prev/next(前后帧的token)， data(字典1，key为传感器类型(如 CAM_FRONT 前视相机), value为传感器数据token)
#### Sample Data(sample_data.json)
token，sample_token(关联的样本帧)， ego_pose_token(关联的自车位姿)， calibrated_sensor_token(传感器校准参数)，filename(数据文件路径，如samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927637525.jpg)，sensor_modality(传感器类型:camera/lidar激光雷达/radar毫米波雷达)，channel(传感器位置)
#### Sample Annotation(sample_annotation.json)
token，sample_token，instance_token(目标实例)，category_name(目标类别，如vehicle.car、pedestrian、cyclist)，translation(3D边界框中心坐标，米)，size(3D边界框长宽高)，rotation(3D旋转角，描述目标朝向)，velocity(目标速度)
#### Others
ego_pose.json：自车位姿（位置、旋转），用于将传感器数据转换到世界坐标系<br>
calibrated_sensor.json：传感器校准参数（如相机内参、激光雷达与自车的外参）<br>
instance.json：目标实例信息（同一物体在整个场景中的持续跟踪 ID）
#### Structure
v1.0-trainval/
├── samples/               
│   ├── CAM_FRONT/         
│   ├── LIDAR_TOP/        
│   ├── RADAR_FRONT/       
│   ...（其他传感器）
├── sweeps/               
├── maps/         
└── v1.0-trainval/        
    ├── scene.json
    ├── sample.json
    ├── sample_data.json
    ├── sample_annotation.json
    ...（其他元数据文件）
#### Evaluation Citeration
自动驾驶感知部分任务主要有3D object detection和map segmentation，采用的是nuScenes数据集的评估标准<br>
五大误差指标：mATE(平均平移误差)，mASE(平均尺度误差)，mAOE(平均方向误差)，mAVE(平均速度误差)，mAAE(平均属性误差)<br>
mAP(平均精度)：计算不同类别目标（如行人、车辆）在不同置信度阈值下的精度 - 召回率曲线下面积，再取均值，衡量检测的整体准确程度<br>
NDS(nuScene检测分数)：核心综合指标，通过对 mAP 及 5 个误差指标加权平均计算得出，能全面反映检测算法的综合性能，下面是论文中的计算公式<br>
$NDS = \frac{1}{10} \left[ 5 \times \text{mAP} + \sum_{mTP \in TP} \left( 1 - \min(1, mTP) \right) \right]$
### Creation
本论文的创新点主要集中在encorder架构的设计上和其他在训练上的创新<br>
#### Encorder
<img src="res\BEVFormer1.png" alt="核心架构" width="600" />
encorder主要组成有：BEV queries, temporal self-attention, spatial cross-attention, feed forward，temporal self-attention处理了BEV features的时序信息；spatial cross-attention的输入是多视角图像经过BackBone网络输出的features。输出的BEV features可用于3D object detection和map segmentation等下游任务

### Conclusion
BEVFormer是自动驾驶感知部分的典范，感知层主要流程：数据集的加载与预处理 -> 各类传感器原始数据的转换(主要转换成张量，可以加入位置、类别嵌入) -> Encorder(先用Resnet提取特征金字塔，再用时空transformer输出bev_features) -> Decorder(3D检测任务的探测头)，这也是主要的代码逻辑
