# SmolVLA微调快速指南 - Parol6机械臂应用

## 🎯 核心问题回答

### Q1: 怎么微调？

**3步搞定**：
```bash
# 1. 准备数据集（已有或自己录制）
# 2. 运行训练脚本
python lerobot/scripts/train.py \\
    --policy.path=lerobot/smolvla_base \\
    --dataset.repo_id=你的数据集 \\
    --batch_size=16 \\
    --steps=20000

# 3. 等待1-2小时完成
```

---

### Q2: 微调数据怎么获取？

#### 方案A: 使用现有数据集（最快）
```python
# HuggingFace上的SO100/SO101数据集
- lerobot/svla_so100_pickplace  ✅ 你已测试的
- lerobot/svla_so101_pickplace
- lerobot/aloha_sim_insertion_human
```

#### 方案B: 自己录制（针对Parol6）⭐ 推荐
```bash
# 使用LeRobot录制工具
lerobot-record \\
    --robot.type=你的机械臂类型 \\
    --robot.port=/dev/ttyUSB0 \\
    --dataset.single_task=\"抓取立方体\" \\
    --dataset.num_episodes=50  # 至少50次演示
```

**录制要求**：
- 最少：30-50次成功演示
- 推荐：100次演示（包含不同位置/角度）
- 每次10-30秒
- 需要摄像头（俯视+腕部视角）

---

### Q3: 3060/4060Ti 怎么微调？

#### 显卡对比

| GPU | 显存 | 批量大小 | 训练时间(20k步) | 可行性 |
|-----|------|---------|----------------|--------|
| **RTX 3060** | 12GB | 8-16 | 2-3小时 | ✅ 可行 |
| **RTX 4060Ti** | 16GB | 16-32 | 1.5-2小时 | ✅✅ 更好 |

#### 优化配置

**3060配置**（12GB显存）：
```bash
python lerobot/scripts/train.py \\
    --policy.path=lerobot/smolvla_base \\
    --dataset.repo_id=你的数据集 \\
    --batch_size=8 \\          # ⚠️ 小批量
    --steps=20000 \\
    --policy.device=cuda \\
    --gradient_accumulation=4  # 模拟batch=32
```

**4060Ti配置**（16GB显存）：
```bash
python lerobot/scripts/train.py \\
    --policy.path=lerobot/smolvla_base \\
    --dataset.repo_id=你的数据集 \\
    --batch_size=16 \\         # ✅ 正常批量
    --steps=20000 \\
    --policy.device=cuda
```

#### 节省显存技巧
```python
# 1. 使用混合精度
--use_amp=true  # 自动混合精度

# 2. 梯度检查点
--gradient_checkpointing=true

# 3. 降低图像分辨率（如需要）
--image_size=224  # 默认256
```

---

### Q4: 微调后能在Parol6上用吗？⭐⭐⭐

#### 挑战与可行性

**❌ 直接问题**：

1. **机械臂不同**
   - SO100: 6自由度机械臂
   - **Parol6: 6自由度，但结构不同**
   - ⚠️ 关节配置可能不同

2. **动作空间可能不匹配**
   - SO100: 关节角度 [0-180°]
   - Parol6: 需要查看具体范围

3. **摄像头位置**
   - 训练数据: 特定摄像头角度
   - Parol6: 需要相似布置

**✅ 解决方案**：

#### 方案1: 在Parol6数据上重新微调（推荐）

```bash
# 流程
1. 在Parol6上录制50-100次演示数据
2. 上传到HuggingFace
3. 使用smolvla_base在你的数据上微调
4. 在Parol6上测试

# 命令
lerobot-record \\
    --robot.type=parol6 \\       # 需要配置支持
    --dataset.single_task=\"抓取立方体放入盒子\" \\
    --dataset.num_episodes=100

python lerobot/scripts/train.py \\
    --policy.path=lerobot/smolvla_base \\
    --dataset.repo_id=你的用户名/parol6_pickplace \\
    --batch_size=16 \\
    --steps=20000
```

**优势**：
- ✅ 完全适配Parol6
- ✅ 学习Parol6的运动特性
- ✅ 最高成功率

---

#### 方案2: 迁移学习（中等难度）

```python
# 1. 先在SO100上微调
微调模型 = 在SO100数据上训练

# 2. 创建动作空间映射
def map_so100_to_parol6(so100_actions):
    \"\"\"将SO100动作映射到Parol6\"\"\"
    # 关节重映射
    parol6_actions = transform_actions(so100_actions)
    return parol6_actions

# 3. 在Parol6上fine-tune少量数据
继续微调 = 在小量Parol6数据上继续训练
```

**适用场景**：
- ⚠️ Parol6与SO100结构相似
- ⚠️ 需要调试映射关系

---

#### 方案3: Sim2Real（高级方案）

```bash
# 1. 在仿真环境中训练Parol6
在MuJoCo/Isaac Sim中建立Parol6模型

# 2. 领域随机化
随机化光照、纹理、物体位置

# 3. 实体测试
部署到真实Parol6
```

---

## 🚀 快速上手：3步行动计划

### 第1步：验证微调可行性（30分钟）

```bash
# 在SO100数据上快速测试微调
cd /root/lerobot_project

python lerobot/scripts/train.py \\
    --policy.path=lerobot/smolvla_base \\
    --dataset.repo_id=lerobot/svla_so100_pickplace \\
    --batch_size=8 \\
    --steps=1000 \\              # 仅1000步测试
    --output_dir=outputs/test_finetune \\
    --policy.device=cuda

# 预期: 5-10分钟完成，MAE应该有下降
```

### 第2步：完整微调（1-2小时）

```bash
# 完整训练20k步
python lerobot/scripts/train.py \\
    --policy.path=lerobot/smolvla_base \\
    --dataset.repo_id=lerobot/svla_so100_pickplace \\
    --batch_size=16 \\           # 3060用8
    --steps=20000 \\
    --output_dir=outputs/smolvla_so100_finetuned \\
    --policy.device=cuda \\
    --save_freq=5000            # 每5k步保存一次

# 预期: MAE从30°降至10-15°
```

### 第3步：准备Parol6应用

#### 3.1 检查Parol6规格

```python
# 需要确认的信息
Parol6规格 = {
    \"关节数\": 6,
    \"关节范围\": [
        (min_angle, max_angle),  # 每个关节
        ...
    ],
    \"工作空间\": \"xyz范围\",
    \"夹爪\": \"开合范围\",
    \"控制接口\": \"ROS/串口/其他\"
}
```

#### 3.2 录制Parol6数据

**设备需求**：
- Parol6机械臂
- 2个摄像头（USB/网络摄像头）
  - 俯视摄像头（看整个工作区）
  - 腕部摄像头（如有）
- 物体：立方体、盒子等

**录制脚本**（需要适配Parol6）：
```python
# 简化版录制脚本
import cv2
import numpy as np

def record_episode():
    \"\"\"录制一次演示\"\"\"
    episode_data = {
        \"observation.images.top\": [],      # 俯视图
        \"observation.state\": [],           # 关节状态
        \"action\": [],                      # 目标动作
        \"task\": \"Pick cube and place in box\"
    }
    
    # 遥操作或示教录制
    while not done:
        # 读取当前状态
        current_state = parol6.get_joint_positions()
        # 读取图像
        image = camera.read()
        # 执行动作
        action = get_next_action()
        
        episode_data[\"observation.state\"].append(current_state)
        episode_data[\"observation.images.top\"].append(image)
        episode_data[\"action\"].append(action)
    
    return episode_data

# 录制50次
for i in range(50):
    episode = record_episode()
    save_episode(episode, f\"episode_{i}\")
```

#### 3.3 在Parol6数据上微调

```bash
# 上传数据后
python lerobot/scripts/train.py \\
    --policy.path=lerobot/smolvla_base \\
    --dataset.repo_id=你的用户名/parol6_pickplace \\
    --batch_size=16 \\
    --steps=30000 \\             # Parol6可能需要更多步数
    --output_dir=outputs/smolvla_parol6
```

---

## 📊 预期效果

### SO100微调后
```
训练前（Base模型）：
- MAE: 30.42°
- 成功率: ~10-20%（估计）

训练后（微调模型）：
- MAE: 10-15° ✅
- 成功率: 60-70% ✅✅
```

### Parol6微调后（在自己数据上）
```
预期：
- MAE: 5-10° ✅✅✅
- 成功率: 70-85% ✅✅✅
- 抓取精度: 高
```

---

## ⚠️ 关键注意事项

### 1. MAE降低 ≠ 能控制Parol6

```
SO100上MAE=5° → ❌ 不能直接用在Parol6
Parol6上MAE=5° → ✅ 能在Parol6上工作
```

**原因**：
- 机械臂结构不同
- 动作空间不同
- 需要在目标机械臂上训练

### 2. 必须用Parol6数据微调

```
方案A: SO100微调 → Parol6测试
结果: ❌ 可能完全不工作

方案B: Parol6数据微调 → Parol6测试  
结果: ✅ 高成功率
```

### 3. 数据质量 > 数据数量

```
50次高质量演示 > 200次低质量演示
```

**高质量标准**：
- ✅ 顺畅的运动
- ✅ 成功完成任务
- ✅ 多样化的起始位置
- ✅ 稳定的摄像头角度

---

## 🎯 总结：你的行动路线

### 现在（验证阶段）

```bash
# 1. 在3060上测试微调SO100（1小时）
python lerobot/scripts/train.py \\
    --policy.path=lerobot/smolvla_base \\
    --dataset.repo_id=lerobot/svla_so100_pickplace \\
    --batch_size=8 \\
    --steps=10000

# 2. 验证MAE下降
运行测试脚本，看MAE是否从30°降到15°以下
```

### 下一步（Parol6准备）

```bash
# 1. 准备Parol6录制环境
- 安装2个摄像头
- 准备抓取物体
- 测试控制接口

# 2. 录制50-100次演示
- 手动示教或遥操作
- 保存为LeRobot格式

# 3. 上传到HuggingFace
huggingface-cli upload 你的用户名/parol6_pickplace ./data
```

### 最终（Parol6应用）

```bash
# 1. 在Parol6数据上微调
python lerobot/scripts/train.py \\
    --policy.path=lerobot/smolvla_base \\
    --dataset.repo_id=你的用户名/parol6_pickplace \\
    --batch_size=16 \\
    --steps=30000

# 2. 部署测试
加载微调模型 → Parol6上实时推理 → 抓取测试

# 3. 迭代改进
收集失败案例 → 添加到数据集 → 继续微调
```

---

## 💰 成本估算

| 项目 | 3060 | 4060Ti |
|------|------|--------|
| **SO100微调(20k步)** | 2-3小时 | 1.5-2小时 |
| **Parol6微调(30k步)** | 3-4小时 | 2-3小时 |
| **电费** | ~2元 | ~2元 |
| **总计** | ✅ 可行 | ✅✅ 更快 |

---

## 📞 如果需要帮助

1. **微调脚本错误** → 检查lerobot安装
2. **显存不足** → 降低batch_size到4
3. **Parol6录制** → 先搜索是否有现成工具
4. **部署问题** → 需要Parol6的控制接口代码

**下一步建议**：
1. ✅ 先在SO100上完整微调，验证流程
2. ✅ 准备Parol6的录制环境
3. ✅ 录制高质量数据
4. ✅ 在Parol6数据上微调测试