# SmolVLA 训练数据集测试计划

## 概述

本文档记录了对 SmolVLA 官方训练数据集的测试计划。由于 PushT 数据集不在 SmolVLA 的训练数据中，导致测试结果不理想（MAE=103 像素），现改为测试 SmolVLA 实际训练过的数据集。

---

## 数据集可用性检查结果

✅ **所有 4 个真实世界数据集均可下载**

### 1. SO100 Pick-Place
- **数据集ID**: `lerobot/svla_so100_pickplace`
- **任务**: 抓取立方体并放入盒子
- **样本数**: 19,631
- **相机**: top (俯视) + wrist (腕部)
- **图像尺寸**: 480×640×3
- **动作维度**: 6 (6-DOF 机器人臂)
- **动作范围**: [0.11, 177.19]
- **状态维度**: 6

### 2. SO100 Stacking
- **数据集ID**: `lerobot/svla_so100_stacking`
- **任务**: 堆叠立方体
- **样本数**: 22,956
- **相机**: top + wrist
- **图像尺寸**: 480×640×3
- **动作维度**: 6
- **动作范围**: [0.32, 177.10]
- **状态维度**: 6

### 3. SO100 Sorting
- **数据集ID**: `lerobot/svla_so100_sorting`
- **任务**: 分类物品
- **样本数**: 35,713
- **相机**: top + wrist
- **图像尺寸**: 480×640×3
- **动作维度**: 6
- **动作范围**: [-0.53, 177.19]
- **状态维度**: 6

### 4. SO101 Pick-Place
- **数据集ID**: `lerobot/svla_so101_pickplace`
- **任务**: 抓取乐高块并放入盒子
- **样本数**: 11,939
- **相机**: up (上方) + side (侧面)
- **图像尺寸**: 480×640×3
- **动作维度**: 6
- **动作范围**: [-99.41, 99.54] ⚠️ 注意归一化范围不同
- **状态维度**: 6

---

## 数据集特点对比

| 特性 | SO100 系列 | SO101 | PushT (之前测试) |
|------|-----------|-------|-----------------|
| **机器人类型** | SO-100 臂 | SO-101 臂 | 2D 推块 |
| **动作空间** | 6-DOF 关节 | 6-DOF 关节 | 2D 像素坐标 |
| **相机数量** | 2 (top+wrist) | 2 (up+side) | 1 (单视角) |
| **图像尺寸** | 480×640 | 480×640 | 96×96 |
| **动作单位** | 角度(度?) | 归一化值 | 像素 [0,512] |
| **任务类型** | 3D 操作 | 3D 操作 | 2D 推动 |
| **训练状态** | ✅ 预训练 | ⚠️ 未预训练 | ❌ 未见过 |

---

## 关键发现

### 动作范围差异

1. **SO100 数据集**: 动作值在 [0, 177] 范围
   - 可能是**角度（度）**：0-180° 范围
   - 直接控制伺服电机位置

2. **SO101 数据集**: 动作值在 [-99, 99] 范围  
   - 可能是**归一化值**：[-1, 1] 的缩放版本
   - 或者是相对位置变化

3. **PushT 数据集**: 动作值在 [0, 512] 范围
   - **像素坐标**：屏幕/图像空间
   - 与机器人关节空间完全不同

### 相机配置差异

- **SO100**: `top` (俯视) + `wrist` (腕部视角)
- **SO101**: `up` (上方) + `side` (侧面视角)  
- **PushT**: 单个固定视角

SmolVLA 需要**3个相机输入**（camera1, camera2, camera3），我们之前对 PushT 重复了同一个图像。

---

## 推荐测试策略

### 优先级 1: SO100 Pick-Place ⭐⭐⭐

**推荐理由**:
1. ✅ SmolVLA 在此数据集上预训练
2. ✅ 样本数适中 (19,631)
3. ✅ 任务明确且易于评估
4. ✅ 与模型训练数据完全匹配

**预期结果**:
- MSE: < 10 (关节角度²)
- MAE: < 3° (关节角度)
- 这将证明模型本身工作正常

### 优先级 2: SO101 Pick-Place ⭐⭐

**推荐理由**:
1. ⚠️ 未在预训练中，测试**泛化能力**
2. ✅ 任务类似，但机器人和相机不同
3. ✅ 样本数较少 (11,939)，测试快

**预期结果**:
- MSE: 10-50 (需要适应)
- MAE: 5-10° 
- 测试跨机器人平台的能力

### 优先级 3: SO100 Stacking ⭐

**推荐理由**:
1. ✅ 预训练数据
2. ⚠️ 任务更复杂（堆叠需要精确性）
3. ✅ 样本数多 (22,956)

### 优先级 4: SO100 Sorting ⭐

**推荐理由**:
1. ✅ 预训练数据
2. ⚠️ 任务最复杂
3. ⚠️ 样本数最多 (35,713)，测试耗时

---

## 测试脚本修改要点

### 1. 相机输入映射

```python
# SO100 数据集
batch = {
    'observation.images.camera1': sample['observation.images.top'],
    'observation.images.camera2': sample['observation.images.wrist'],
    'observation.images.camera3': sample['observation.images.wrist'],  # 重复使用
    ...
}

# SO101 数据集  
batch = {
    'observation.images.camera1': sample['observation.images.up'],
    'observation.images.camera2': sample['observation.images.side'],
    'observation.images.camera3': sample['observation.images.side'],  # 重复使用
    ...
}
```

### 2. 图像尺寸调整

```python
# 数据集图像: 480×640
# 模型期望: 256×256
image = F.interpolate(image, size=(256, 256), mode='bilinear', align_corners=False)
```

### 3. 动作维度匹配

```python
# 数据集动作: [6] (所有6个关节)
# 模型输出: [batch, action_chunk_size * action_dim]
#          = [1, 3 * 6] = [1, 18]

# 取第一个时间步的所有6个关节
pred_action = output[0, :6].cpu().numpy()
true_action = sample['action'].cpu().numpy()
```

### 4. 不需要归一化！

```python
# ✅ 直接比较原始值即可
# SO100/SO101 的动作值已经是模型输出的尺度
# 不需要像 PushT 那样做反归一化

mse = np.mean((pred_action - true_action) ** 2)
mae = np.mean(np.abs(pred_action - true_action))
```

---

## 预期性能指标

基于 SmolVLA 论文报告的性能：

### SO100 数据集（预训练）
- **成功率**: 78.3% (多任务设置)
- **预期 MAE**: < 5° (关节角度误差)
- **预期 MSE**: < 25 (关节角度平方误差)

### SO101 数据集（跨平台）
- **成功率**: 90% (in-distribution)
- **成功率**: 50% (out-of-distribution)
- **预期 MAE**: 5-15° 
- **预期 MSE**: 25-225

---

## 测试步骤

### 第一步: 快速验证（建议）

```bash
# 测试 SO100 Pick-Place 的前 100 个样本
python 10-test_so100_pickplace.py --num_samples 100
```

**验证点**:
1. 模型能正确加载
2. 数据集格式正确
3. 推理速度正常
4. 输出范围合理

### 第二步: 完整测试

```bash
# 测试更多样本以获得稳定指标
python 10-test_so100_pickplace.py --num_samples 1000
```

### 第三步: 跨平台测试

```bash
# 测试 SO101（未见过的机器人）
python 11-test_so101_pickplace.py --num_samples 500
```

### 第四步: 复杂任务测试

```bash
# 测试堆叠和分类任务
python 12-test_so100_stacking.py --num_samples 500
python 13-test_so100_sorting.py --num_samples 500
```

---

## 成功标准

### ✅ 测试通过条件

**SO100 Pick-Place (预训练数据)**:
- MAE < 10° (关节角度误差)
- MSE < 100
- 推理时间 < 20ms/样本

**SO101 Pick-Place (跨平台)**:
- MAE < 20° 
- MSE < 400
- 显示出合理的泛化能力

### ❌ 测试失败条件

- MAE > 50°（完全随机的水平）
- 推理崩溃或错误
- 输出值异常（NaN, Inf）

---

## 与 PushT 测试的对比

| 指标 | PushT 测试 | SO100 测试（预期）|
|------|-----------|------------------|
| **数据集匹配** | ❌ 未见过 | ✅ 预训练数据 |
| **动作空间** | 2D 像素 | 6D 关节角度 |
| **MAE** | 103 像素 | < 5° |
| **MSE** | 15,790 | < 25 |
| **尺度问题** | ✅ 已修复 | ✅ 无需处理 |
| **预期性能** | 差 | 好 |

---

## 后续工作

### 如果 SO100 测试成功

1. ✅ 证明模型工作正常
2. ✅ 证明测试流程正确
3. ➡️ 可以尝试 SO101 跨平台测试
4. ➡️ 可以尝试复杂任务（堆叠、分类）

### 如果想在 PushT 上成功

需要以下之一：
1. **微调 SmolVLA**: 在 PushT 数据集上训练 20k 步
2. **使用其他模型**: 寻找专门为 PushT 训练的策略
3. **数据转换**: 将 PushT 的像素坐标转换为类似机器人关节的表示

---

## 参考资料

- SmolVLA 论文: https://arxiv.org/abs/2506.01844
- SO100 数据集: https://huggingface.co/datasets/lerobot/svla_so100_pickplace
- SO101 数据集: https://huggingface.co/datasets/lerobot/svla_so101_pickplace
- LeRobot 文档: https://huggingface.co/docs/lerobot

---

*测试计划创建时间: 2024-10-31*
*下一步: 创建并运行 10-test_so100_pickplace.py*
