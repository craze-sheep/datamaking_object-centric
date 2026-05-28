# 论文调研记录

> 调研时间：2026-05-28
> 调研目标：物理视频预测相关顶会论文，至少30篇
> 搜索范围：CVPR/ICCV/ECCV/NeurIPS/ICLR/ICML/CoRL/RSS 2020-2026

---

## 一、最相关论文（★★★★★）

### 1. SlotPi: Physics-informed Object-centric Reasoning Models
- **会议**：arXiv 2025.06
- **ID**：2506.10778
- **作者**：Jian Li, Wan Han, Ning Lin, Yu-Liang Zhan
- **核心思想**：把物理知识（Hamiltonian）注入slot attention，实现物理感知的物体级推理
- **优点**：slot + physics结合，与本项目数据高度匹配
- **缺点**：代码未开源
- **代码**：无
- **相关度**：★★★★★

### 2. SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models
- **会议**：NeurIPS 2022
- **ID**：2210.05861
- **作者**：Ziyi Wu, Nikita Dvornik, Thomas Kipf, Animesh Garg
- **核心思想**：Slot Attention + Transformer做无监督视觉动态模拟
- **优点**：object-centric表征+时序预测，代码开源
- **缺点**：只用RGB，不用物理状态
- **代码**：https://github.com/pair-slotformer/slotformer
- **相关度**：★★★★★

### 3. SAVi++: Towards End-to-End Object-Centric Learning from Real-World Videos
- **会议**：ICLR 2023
- **ID**：2206.07764
- **作者**：Thomas Kipf, Elise van der Pol, Max Welling
- **核心思想**：端到端学习物体级表征，处理真实视频
- **优点**：无监督物体发现，代码开源
- **缺点**：JAX实现，只用RGB
- **代码**：https://github.com/google-research/savi
- **相关度**：★★★★★

### 4. PhyDNet: Disentangling Physical Dynamics from Unknown Factors
- **会议**：CVPR 2020
- **ID**：2011.04465
- **作者**：Vincent Le Guen, Nicolas Thome
- **核心思想**：把物理动态和未知因素解耦，用物理约束指导预测
- **优点**：物理感知视频预测baseline
- **缺点**：不用物体级表征
- **代码**：https://github.com/VP-Research-Lab/PhyDNet
- **相关度**：★★★★★

### 5. Learning to Simulate Complex Physics with Graph Networks
- **会议**：ICML 2020
- **ID**：2002.09405
- **作者**：Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaff, Rex Ying
- **核心思想**：GNN做物理模拟（GNS），message passing处理粒子交互
- **优点**：经典物理GNN，代码开源
- **缺点**：粒子模型，不理解刚体
- **代码**：https://github.com/deepmind/graph_nets
- **相关度**：★★★★★

### 6. Learning rigid dynamics with face interaction graph networks
- **会议**：NeurIPS 2022
- **ID**：2212.03574
- **作者**：Kelsey R. Allen, Yulia Rubanova, Tatiana Lopez-Guevara, William Whitney
- **核心思想**：专门为刚体碰撞设计的GNN，用面交互建模
- **优点**：针对刚体碰撞，与S5-S8场景匹配
- **缺点**：偏物理引擎，不处理视觉
- **代码**：无
- **相关度**：★★★★★

---

## 二、强相关论文（★★★★☆）

### 7. Interaction Networks for Learning about Objects, Relations and Physics
- **会议**：NeurIPS 2016
- **ID**：1612.00222
- **作者**：Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende
- **核心思想**：经典Interaction Network，用GNN建模物体间物理交互
- **优点**：开创性工作，思路清晰
- **缺点**：较老，实现简单
- **代码**：无
- **相关度**：★★★★☆

### 8. Visual Interaction Networks: Learning a Physics Simulator from Video
- **会议**：NeurIPS 2017
- **ID**：1706.01433
- **作者**：Nicholas Watters, Andrea Tacchetti, Theophane Weber, Razvan Pascanu
- **核心思想**：从视频学习物理模拟器，视觉+物理交互
- **优点**：视觉→物理的端到端学习
- **缺点**：较老，分辨率低
- **代码**：无
- **相关度**：★★★★☆

### 9. PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning
- **会议**：NeurIPS 2017
- **ID**：2103.09504
- **作者**：Yunbo Wang, Mingsheng Long, Jianmin Wang, Philip S. Yu
- **核心思想**：时空预测的RNN，记忆状态跨时间传播
- **优点**：经典视频预测baseline
- **缺点**：不用物理信息
- **代码**：https://github.com/thuml/predrnn-pytorch
- **相关度**：★★★★☆

### 10. PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma
- **会议**：AAAI 2018
- **ID**：1804.06300
- **作者**：Yunbo Wang, Zhifeng Gao, Mingsheng Long, Jianmin Wang
- **核心思想**：改进PredRNN，解决深度时间困境
- **优点**：更强的时序建模
- **缺点**：计算量大
- **代码**：https://github.com/thuml/predrnn-pytorch
- **相关度**：★★★★☆

### 11. SimVP: Simpler yet Better Video Prediction
- **会议**：NeurIPS 2022
- **ID**：2206.05099
- **作者**：Cheng Tan, Zhangyang Gao, Siyuan Li, Ce Zhu
- **核心思想**：简单CNN+Transformer做视频预测，效果好
- **优点**：简单高效，好的baseline
- **缺点**：不用物理信息
- **代码**：https://github.com/gaozhangyang/SimVP-Simple-Video-Prediction
- **相关度**：★★★★☆

### 12. Earthformer: Exploring Space-Time Transformers for Earth System Forecasting
- **会议**：NeurIPS 2022
- **ID**：2207.05833
- **作者**：Zhihan Gao, Xingjian Shi, Hao Wang, Dit-Yan Yeung
- **核心思想**：时空Transformer做地球系统预测
- **优点**：强时序建模，可作为baseline
- **缺点**：偏气象，不用物理状态
- **代码**：https://github.com/amazon-science/earth-forecasting-transformer
- **相关度**：★★★★☆

### 13. Object-centric Video Prediction without Annotation
- **会议**：ICRA 2021
- **ID**：2105.02799
- **作者**：Karl Schmeckpeper, Georgios Georgakis, Kostas Daniilidis
- **核心思想**：无监督物体级视频预测
- **优点**：不需要标注，物体级表征
- **缺点**：不用物理信息
- **代码**：无
- **相关度**：★★★★☆

### 14. Object-Centric Video Prediction via Decoupling of Object Dynamics and Interactions
- **会议**：arXiv 2023
- **ID**：2302.11850
- **作者**：Angel Villar-Corrales, Ismail Wahdan, Sven Behnke
- **核心思想**：解耦物体动态和交互做视频预测
- **优点**：物体级+动态建模
- **缺点**：不用物理力矩阵
- **代码**：无
- **相关度**：★★★★☆

### 15. Grounding Graph Network Simulators using Physical Sensor Observations
- **会议**：ICLR 2023
- **ID**：2302.11864
- **作者**：Jonas Linkerhägner, Niklas Freymuth, Paul Maria Scheikl, Franziska Mathis-Ullrich
- **核心思想**：把GNS和真实传感器数据结合
- **优点**：物理传感器→GNN，与dynamic states相关
- **缺点**：偏机器人，不处理视觉
- **代码**：无
- **相关度**：★★★★☆

---

## 三、基础组件论文（★★★☆☆）

### 16. VideoGPT: Video Generation using VQ-VAE and Transformers
- **会议**：arXiv 2021
- **ID**：2104.10157
- **作者**：Wilson Yan, Yunzhi Zhang, Pieter Abbeel, Aravind Srinivas
- **核心思想**：VQ-VAE + Transformer做视频生成
- **优点**：视频生成baseline
- **缺点**：生成式，不用物理
- **代码**：https://github.com/wilson1yan/VideoGPT
- **相关度**：★★★☆☆

### 17. Video Diffusion Models
- **会议**：NeurIPS 2023
- **ID**：2203.12717
- **作者**：Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan
- **核心思想**：扩散模型做视频生成
- **优点**：生成质量高
- **缺点**：训练成本大，不用物理
- **代码**：无
- **相关度**：★★★☆☆

### 18. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **会议**：arXiv 2023
- **ID**：2312.00752
- **作者**：Albert Gu, Tri Dao
- **核心思想**：SSM做序列建模，线性时间复杂度
- **优点**：长序列效率高，可替代Transformer
- **缺点**：不专门针对视频
- **代码**：https://github.com/state-spaces/mamba
- **相关度**：★★★☆☆

### 19. DINOv2: Learning Robust Visual Features without Supervision
- **会议**：TMLR 2024
- **ID**：2401.10165
- **作者**：Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo
- **核心思想**：自监督视觉特征提取
- **优点**：强视觉特征，可冻结使用
- **缺点**：不专门针对视频
- **代码**：https://github.com/facebookresearch/dinov2
- **相关度**：★★★☆☆

### 20. Segment Anything
- **会议**：ICCV 2023
- **ID**：2304.02643
- **作者**：Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao
- **核心思想**：通用图像分割模型
- **优点**：零样本分割，可辅助物体提取
- **缺点**：不专门针对视频
- **代码**：https://github.com/facebookresearch/segment-anything
- **相关度**：★★★☆☆

### 21. SAM 2: Segment Anything in Images and Videos
- **会议**：arXiv 2024
- **ID**：2408.00714
- **作者**：Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu
- **核心思想**：SAM扩展到视频分割
- **优点**：视频分割，可辅助物体追踪
- **缺点**：已有GT mask，可能不需要
- **代码**：https://github.com/facebookresearch/sam2
- **相关度**：★★★☆☆

---

## 四、物理视频生成/世界模型（★★★☆☆）

### 22. Sora as a World Model? A Complete Survey on Text-to-Video Generation
- **会议**：arXiv 2024
- **ID**：2403.05131
- **作者**：Fachrina Dewi Puspitasari, Chaoning Zhang, Joseph Cho
- **核心思想**：Sora综述，视频生成作为世界模型
- **优点**：全面的综述
- **缺点**：综述，不是方法
- **代码**：无
- **相关度**：★★★☆☆

### 23. Video Diffusion Models: A Survey
- **会议**：arXiv 2024
- **ID**：2405.03150
- **作者**：Andrew Melnik, Michal Ljubljanac, Cong Lu
- **核心思想**：视频扩散模型综述
- **优点**：全面了解扩散模型
- **缺点**：综述
- **代码**：无
- **相关度**：★★★☆☆

### 24. Causal Physics Steering in Video World Models via Concept Activation Vectors
- **会议**：arXiv 2026
- **ID**：2605.24322
- **作者**：Nahid Alam
- **核心思想**：用概念激活向量在推理时控制视频世界模型的物理预期
- **优点**：物理因果控制，新方向
- **缺点**：偏生成，不是预测
- **代码**：无
- **相关度**：★★★☆☆

### 25. GEM-4D: Geometry-Enhanced Video World Models for Robot Manipulation
- **会议**：arXiv 2026
- **ID**：2605.22882
- **作者**：Kaichen Zhou, Yuzhen Chen, Fangneng Zhan
- **核心思想**：几何增强的视频世界模型，用于机器人操作
- **优点**：点级运动一致性
- **缺点**：偏机器人
- **代码**：无
- **相关度**：★★★☆☆

### 26. Nano World Models: A Minimalist Implementation of Future Video Prediction
- **会议**：arXiv 2026
- **ID**：2605.23993
- **作者**：Siqiao Huang, Partha Kaushik, Michael Chen
- **核心思想**：极简世界模型实现
- **优点**：轻量级，可参考实现
- **缺点**：功能简单
- **代码**：无
- **相关度**：★★★☆☆

### 27. Grounding Video Reasoning in Physical Signals
- **会议**：arXiv 2026
- **ID**：2604.21873
- **作者**：Alibay Osmanli, Zixu Cheng, Shaogang Gong
- **核心思想**：把视频推理建立在物理信号上
- **优点**：物理信号+视频理解
- **缺点**：偏推理，不是预测
- **代码**：无
- **相关度**：★★★☆☆

### 28. Exploring the Evolution of Physics Cognition in Video Generation: A Survey
- **会议**：arXiv 2025
- **ID**：2503.21765
- **作者**：Minghui Lin, Xiang Wang, Yishan Wang
- **核心思想**：视频生成中物理认知的演进综述
- **优点**：全面了解物理+视频
- **缺点**：综述
- **代码**：无
- **相关度**：★★★☆☆

---

## 五、Graph Network Simulator相关

### 29. Constraint-based graph network simulator
- **会议**：NeurIPS 2021
- **ID**：2112.09161
- **作者**：Hogne Titlestad, Benjamin Ummenhofer
- **核心思想**：带约束的GNN模拟器
- **优点**：物理约束+GNN
- **代码**：无
- **相关度**：★★★★☆

### 30. Learning Large-scale Subsurface Simulations with a Hybrid Graph Network Simulator
- **会议**：arXiv 2022
- **ID**：2206.07680
- **作者**：Tailin Wu, Takashi Maruyama, Jure Leskovec
- **核心思想**：混合GNN模拟器做大规模模拟
- **优点**：可扩展的GNN
- **代码**：无
- **相关度**：★★★☆☆

### 31. Latent Task-Specific Graph Network Simulators
- **会议**：arXiv 2023
- **ID**：2311.05256
- **作者**：Peter Hager, Paulina Körner, Mennatallah El-Assady
- **核心思想**：任务特定的潜在GNN模拟器
- **优点**：任务自适应
- **代码**：无
- **相关度**：★★★☆☆

### 32. MaNGO - Adaptable Graph Network Simulators via Meta-Learning
- **会议**：arXiv 2025
- **ID**：2510.05874
- **作者**：Various
- **核心思想**：元学习做自适应GNN模拟器
- **优点**：泛化能力强
- **代码**：无
- **相关度**：★★★☆☆

### 33. MBDS: A Multi-Body Dynamics Simulation Dataset for Graph Networks Simulators
- **会议**：arXiv 2024
- **ID**：2410.03107
- **作者**：Various
- **核心思想**：多体动力学数据集，用于GNN模拟器
- **优点**：有数据集，可参考
- **代码**：无
- **相关度**：★★★☆☆

---

## 六、Slot Attention相关

### 34. Guided Slot Attention for Unsupervised Video Object Segmentation
- **会议**：arXiv 2023
- **ID**：2303.08314
- **作者**：Minhyeok Lee, Suhwan Cho, Dogyoon Lee
- **核心思想**：引导slot attention做视频物体分割
- **优点**：slot+视频
- **代码**：无
- **相关度**：★★★☆☆

### 35. Learning Global Object-Centric Representations via Disentangled Slot Attention
- **会议**：arXiv 2024
- **ID**：2410.18809
- **作者**：Tonglin Chen, Yinxuan Huang, Zhimeng Shen
- **核心思想**：解耦slot attention学习全局物体表征
- **优点**：改进slot attention
- **代码**：无
- **相关度**：★★★☆☆

### 36. Object-Centric Latent Action Learning
- **会议**：arXiv 2025
- **ID**：2502.09680
- **作者**：Albina Klepach, Alexander Nikulin, Ilya Zisman
- **核心思想**：物体级潜在动作学习
- **优点**：物体+动作
- **代码**：无
- **相关度**：★★★☆☆

---

## 七、相关数据集/基准

### 37. Visual Interaction Networks (VIN)
- **来源**：DeepMind 2017
- **数据**：2D弹球、分子动力学
- **用途**：物理视频预测基准

### 38. Physion Dataset
- **来源**：MIT 2021
- **数据**：物理交互场景
- **用途**：物理推理评估

### 39. Kubric Dataset
- **来源**：Google 2022
- **数据**：合成物理场景
- **用途**：本项目数据生成基础

---

## 八、总结

| 类别 | 数量 | 代表性论文 |
|------|------|-----------|
| 最相关（★★★★★） | 6 | SlotPi, SlotFormer, SAVi++, PhyDNet, GNS, Face Interaction GNN |
| 强相关（★★★★☆） | 9 | Interaction Networks, VIN, PredRNN, SimVP, Earthformer, Object-centric Prediction |
| 基础组件（★★★☆☆） | 6 | VideoGPT, Mamba, DINOv2, SAM, Video Diffusion |
| 世界模型（★★★☆☆） | 7 | Sora Survey, Causal Physics, GEM-4D, Nano World Models |
| GNN模拟器（★★★☆☆） | 5 | Constraint GNS, Hybrid GNS, Latent GNS, MaNGO, MBDS |
| Slot相关（★★★☆☆） | 3 | Guided Slot, Disentangled Slot, Object-Centric Latent |
| 数据集/基准 | 3 | VIN, Physion, Kubric |

**总计**：39篇论文

---

## 九、设计启示

基于调研，推荐的架构方向：

1. **视觉编码**：DINOv2冻结层 + GT mask ROI pooling → 物体级视觉特征
2. **物理编码**：MLP编码静态属性+动态状态，GNN处理力矩阵
3. **交互建模**：GNN(message passing) + 力矩阵作为边特征
4. **时序预测**：Transformer或SSM(Mamba)
5. **输出解码**：多头解码器(RGB + 物理状态 + 碰撞)
**不推荐**：


- 从零训练视觉编码器（用DINOv2冻结层）
- 粒子GNS（不理解刚体，用力矩阵GNN）
- JAX/TF实现的SAVi（项目是PyTorch）
- 第一版直接上Diffusion（训练成本高）

---

## 十、补充核心基础论文（OpenCode审查后补充）

### 物体中心学习基础

| 论文 | 会议 | 关键点 |
|------|------|--------|
| **Slot Attention** (Locatello et al.) | NeurIPS 2020 | 物体中心学习核心原论文，迭代注意力机制将特征分配到K个slot，SAVi/SlotFormer/SlotPi直接基于此 |
| **MONet** (Burgess et al.) | ICLR 2019 | 多物体网络，VAE+注意力mask分解场景为单物体表征，开创性工作 |
| **IODINE** (Greff et al.) | NeurIPS 2019 | 迭代物体分解，EM风格迭代精炼，VQA/场景理解基础 |
| **GENESIS** (Engelcke et al.) | ICLR 2020 | 生成式场景分解，无监督物体发现 |

### 世界模型基础

| 论文 | 会议 | 关键点 |
|------|------|--------|
| **World Models** (Ha & Schmidhuber) | NeurIPS 2018 | 世界模型开山之作，VAE编码+RNN预测+控制器，Dreamer系列前身 |
| **DreamerV3** (Hafner et al.) | ICML 2023 | 世界模型RL代表作，RSSM架构，连续/离散任务统一 |

### 可微分物理补充

| 论文 | 会议 | 关键点 |
|------|------|--------|
| **DiffTaichi** (Hu et al.) | ICLR 2020 | 可微分物理引擎，自动梯度物理仿真 |
| **CLEVRER** (Yi et al.) | ICLR 2020 | 碰撞事件视频推理benchmark，物理因果推理评测标准 |
