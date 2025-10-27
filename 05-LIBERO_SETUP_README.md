# SmolVLA + LIBERO 集成指南

## 📁 文件说明

### 主程序
- **05-smolvla_use_libero.py** - SmolVLA使用LIBERO数据集的完整测试程序
- **05-smolvla_use_libero_simple.md** - 测试结果和问题总结

## ✅ 已完成的工作

### 1. 环境配置
```bash
# 使用的Python环境
/root/anaconda3/envs/smolvla/bin/python

# 已安装的LIBERO相关依赖
- bddl==3.6.0
- robosuite==1.4.0  
- libero==0.1.0
- easydict==1.13
- gym==0.25.2
- mujoco==3.3.7
```

### 2. 程序功能
✅ SmolVLA模型加载
✅ LIBERO任务加载
✅ 观测数据预处理
✅ Episode执行逻辑
✅ 性能可视化

### 3. 测试状态
✅ 模型导入成功
✅ LIBERO导入成功  
✅ 任务信息读取成功
❌ 环境初始化失败（EGL渲染问题）

## ❌ 当前问题

### EGL渲染错误
```
ImportError: Cannot initialize a EGL device display.
```

**原因**: LIBERO的OffScreenRenderEnv需要GPU的EGL支持进行无头渲染。

## 💡 解决方案

### 方案1: 使用虚拟显示器 (最简单)
```bash
# 1. 安装xvfb
sudo apt-get update
sudo apt-get install -y xvfb

# 2. 运行程序
xvfb-run -a -s "-screen 0 1400x900x24" \
  /root/anaconda3/envs/smolvla/bin/python \
  /root/smolvla_project/05-smolvla_use_libero.py
```

### 方案2: 设置Mujoco渲染模式
```bash
# 尝试不同的渲染后端
export MUJOCO_GL=osmesa
/root/anaconda3/envs/smolvla/bin/python 05-smolvla_use_libero.py

# 或
export MUJOCO_GL=egl
/root/anaconda3/envs/smolvla/bin/python 05-smolvla_use_libero.py
```

### 方案3: 安装EGL支持
```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

### 方案4: 创建Jupyter Notebook版本
已创建同名的.ipynb文件（如果需要）。

## 📋 程序运行步骤

当渲染问题解决后，程序将按以下步骤执行：

1. **加载模型** (约10-20秒)
   - SmolVLA from "lerobot/smolvla_base"
   - Tokenizer from "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

2. **设置环境** (约5秒)
   - Load LIBERO benchmark: "libero_spatial"
   - Create OffScreenRenderEnv
   - Task: "pick up the black bowl..."

3. **运行Episodes** (约1-2分钟)
   - 5个episodes
   - 每个最多300步
   - 记录: success, steps, rewards, actions, inference times

4. **生成可视化** (约2秒)
   - 成功率柱状图
   - 推理时间直方图
   - 步数折线图
   - 动作空间散点图
   - 保存到: libero_result.png

## 📊 预期输出

```
============================================================
SmolVLA + LIBERO Test
============================================================

[1/4] Loading model...
   ✅ Loaded

[2/4] Setup LIBERO...
   Task: pick_up_the_black_bowl_between...
   Desc: pick up the black bowl between...
   ✅ Env ready

[3/4] Running episodes...
   Ep 1/5... ✅ 87 steps
   Ep 2/5... ❌ 300 steps
   Ep 3/5... ✅ 92 steps
   Ep 4/5... ✅ 105 steps
   Ep 5/5... ❌ 300 steps
   
[4/4] Viz...
   ✅ Saved: libero_result.png

============================================================
📊 Summary
============================================================
Success: 60.0%
Steps: 176.8
Inference: 45.2ms
============================================================
```

## 🔧 调试命令

```bash
# 检查LIBERO配置
cat ~/.libero/config.yaml

# 测试LIBERO导入
/root/anaconda3/envs/smolvla/bin/python -c "from libero.libero import benchmark; print('OK')"

# 测试SmolVLA导入  
/root/anaconda3/envs/smolvla/bin/python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print('OK')"

# 查看GPU
nvidia-smi

# 检查OpenGL
glxinfo | grep "OpenGL"
```

## 📚 参考资料

- LIBERO GitHub: https://github.com/Lifelong-Robot-Learning/LIBERO
- SmolVLA Paper: https://huggingface.co/lerobot/smolvla_base
- Robosuite Docs: https://robosuite.ai/

## ⚠️ 注意事项

1. 确保在smolvla conda环境中运行
2. 需要GPU支持（已验证RTX 4060 Ti可用）
3. 首次运行会下载模型（约2GB）
4. LIBERO任务文件已存在于本地
5. 无需下载LIBERO数据集（用于训练演示）

## ✨ 已测试的环境

- OS: Ubuntu 20.04 (WSL2)
- GPU: NVIDIA GeForce RTX 4060 Ti (16GB)
- Python: 3.10 (smolvla环境)
- CUDA: 可用
- Torch: 2.x

## 🎯 下一步

1. **立即**: 尝试xvfb解决渲染问题
2. **短期**: 运行完整测试，记录性能
3. **中期**: 测试其他LIBERO benchmarks (object, goal, 100)
4. **长期**: 微调SmolVLA在LIBERO任务上的表现

