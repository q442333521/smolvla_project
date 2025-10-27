#!/usr/bin/env python3
"""
SmolVLA + LIBERO 测试脚本 (WSL2修复版)

修复内容：
1. 使用OSMesa软件渲染替代EGL
2. 设置正确的环境变量
3. 添加数据集路径检查
"""

import os
import sys
import warnings

# ==========================================
# 🔧 WSL2渲染修复 - 必须在导入其他库之前设置
# ==========================================
print("[1/5] 🔧 配置WSL2渲染环境...")

# 强制使用OSMesa软件渲染（不需要GPU的EGL支持）
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

# 禁用EGL相关警告
os.environ['DISPLAY'] = ''

print("  ✓ 设置 MUJOCO_GL=osmesa (使用CPU软件渲染)")
print("  ✓ 设置 PYOPENGL_PLATFORM=osmesa")

# ==========================================
# 导入库
# ==========================================
print("[2/5] 📦 导入依赖库...")

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import time

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print("  ✓ LeRobot已安装")
except ImportError as e:
    print(f"  ✗ LeRobot导入失败: {e}")
    print("  请运行: pip install lerobot")
    sys.exit(1)

try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    print("  ✓ LIBERO已安装")
except ImportError as e:
    print(f"  ✗ LIBERO导入失败: {e}")
    print("  请运行安装脚本")
    sys.exit(1)

# ==========================================
# 配置
# ==========================================
print("[3/5] ⚙️  加载配置...")

class Config:
    # 模型配置
    model_name = "lerobot/smolvla_base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # LIBERO配置
    task_suite_name = "libero_10"  # 10个任务的套件
    num_episodes = 3  # 测试episode数量
    max_steps = 300   # 每个episode最大步数
    
    # 渲染配置（关键修复）
    render_mode = "rgb_array"  # 使用数组渲染，不依赖显示器
    camera_names = ["agentview", "robot0_eye_in_hand"]  # LIBERO标准相机
    
    # 输出配置
    save_videos = False  # WSL2建议先关闭视频保存
    verbose = True

config = Config()

print(f"  ✓ 设备: {config.device}")
print(f"  ✓ 任务套件: {config.task_suite_name}")
print(f"  ✓ 渲染模式: {config.render_mode} (OSMesa)")

# ==========================================
# 检查和创建数据集目录
# ==========================================
print("[4/5] 📁 检查LIBERO数据集...")

# 查找LIBERO安装路径
try:
    import libero
    libero_path = Path(libero.__file__).parent
    datasets_path = libero_path / "datasets"
    
    if not datasets_path.exists():
        print(f"  ⚠️  创建数据集目录: {datasets_path}")
        datasets_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ 目录已创建")
    else:
        print(f"  ✓ 数据集路径存在: {datasets_path}")
        
except Exception as e:
    print(f"  ⚠️  警告: {e}")
    print("  ℹ️  将在运行时下载数据集")

# ==========================================
# 加载SmolVLA模型
# ==========================================
print("[5/5] 🤖 加载SmolVLA模型...")

try:
    policy = SmolVLAPolicy.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        device=config.device
    )
    policy.eval()
    print(f"  ✓ 模型加载成功 ({config.model_name})")
    print(f"  ✓ 参数量: ~450M")
except Exception as e:
    print(f"  ✗ 模型加载失败: {e}")
    sys.exit(1)

# ==========================================
# 主测试函数
# ==========================================
def test_smolvla_on_libero():
    """
    在LIBERO环境中测试SmolVLA
    """
    print("\n" + "="*60)
    print("🚀 开始 SmolVLA + LIBERO 测试")
    print("="*60 + "\n")
    
    # 加载任务套件
    print(f"📋 加载任务套件: {config.task_suite_name}")
    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[config.task_suite_name]()
        num_tasks = task_suite.n_tasks
        print(f"  ✓ 任务数量: {num_tasks}")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return
    
    # 统计
    results = {
        'total_episodes': 0,
        'success': 0,
        'failure': 0,
        'avg_steps': [],
        'avg_inference_time': []
    }
    
    # 遍历任务
    for task_id in range(min(2, num_tasks)):  # 先测试2个任务
        print(f"\n{'─'*60}")
        print(f"📝 任务 {task_id}/{num_tasks}")
        
        # 获取任务
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        
        print(f"  名称: {task_name}")
        print(f"  描述: {task_description}")
        
        # 创建环境 - 关键修复点
        print(f"  🔧 创建环境 (使用OSMesa渲染)...")
        try:
            env = OffScreenRenderEnv(
                task_name=task_name,
                task_description=task_description,
                # 关键修复参数
                has_renderer=False,          # 不使用在线渲染器
                has_offscreen_renderer=True,  # 使用离线渲染
                use_camera_obs=True,
                camera_names=config.camera_names,
                camera_heights=224,
                camera_widths=224,
                render_gpu_device_id=-1,     # 使用CPU渲染
            )
            print("    ✓ 环境创建成功")
        except Exception as e:
            print(f"    ✗ 环境创建失败: {e}")
            print(f"    ℹ️  跳过此任务")
            continue
        
        # 运行episodes
        for episode in range(config.num_episodes):
            print(f"\n  📹 Episode {episode+1}/{config.num_episodes}")
            
            # 重置环境
            obs = env.reset()
            done = False
            step_count = 0
            episode_inference_times = []
            
            while not done and step_count < config.max_steps:
                # 获取图像（使用agentview）
                if 'agentview_image' in obs:
                    image = Image.fromarray(obs['agentview_image'])
                elif 'image' in obs:
                    image = Image.fromarray(obs['image'])
                else:
                    # 尝试从其他相机获取
                    cam_name = config.camera_names[0]
                    img_key = f"{cam_name}_image"
                    if img_key in obs:
                        image = Image.fromarray(obs[img_key])
                    else:
                        print(f"      ⚠️  未找到图像，obs keys: {obs.keys()}")
                        break
                
                # 获取机器人状态
                if 'robot0_proprio-state' in obs:
                    state = obs['robot0_proprio-state']
                elif 'state' in obs:
                    state = obs['state']
                else:
                    print(f"      ⚠️  未找到状态，obs keys: {obs.keys()}")
                    break
                
                # SmolVLA推理
                start_time = time.time()
                try:
                    with torch.no_grad():
                        # 准备输入
                        observation = {
                            'image': image,
                            'state': torch.from_numpy(state).float().to(config.device)
                        }
                        
                        # 推理
                        action = policy.select_action(
                            observation=observation,
                            instruction=task_description
                        )
                        
                        # 转换为numpy
                        if isinstance(action, torch.Tensor):
                            action = action.cpu().numpy()
                        
                        # 取第一步动作
                        if action.ndim > 1:
                            action = action[0]
                        
                except Exception as e:
                    print(f"      ✗ 推理失败: {e}")
                    break
                
                inference_time = time.time() - start_time
                episode_inference_times.append(inference_time)
                
                # 执行动作
                try:
                    obs, reward, done, info = env.step(action)
                    step_count += 1
                    
                    if step_count % 50 == 0:
                        avg_time = np.mean(episode_inference_times[-50:])
                        print(f"      步骤 {step_count}/{config.max_steps} | "
                              f"推理: {avg_time*1000:.1f}ms | "
                              f"奖励: {reward:.3f}")
                        
                except Exception as e:
                    print(f"      ✗ 执行失败: {e}")
                    break
            
            # Episode结果
            success = done and info.get('success', False)
            results['total_episodes'] += 1
            results['success' if success else 'failure'] += 1
            results['avg_steps'].append(step_count)
            results['avg_inference_time'].extend(episode_inference_times)
            
            status = "✓ 成功" if success else "✗ 失败"
            print(f"    {status} | 步数: {step_count} | "
                  f"平均推理: {np.mean(episode_inference_times)*1000:.1f}ms")
        
        # 清理环境
        env.close()
    
    # ==========================================
    # 最终统计
    # ==========================================
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    print(f"总Episodes: {results['total_episodes']}")
    print(f"成功: {results['success']} ({results['success']/max(results['total_episodes'],1)*100:.1f}%)")
    print(f"失败: {results['failure']}")
    
    if results['avg_steps']:
        print(f"平均步数: {np.mean(results['avg_steps']):.1f}")
    
    if results['avg_inference_time']:
        print(f"平均推理时间: {np.mean(results['avg_inference_time'])*1000:.1f}ms")
        print(f"推理频率: {1.0/np.mean(results['avg_inference_time']):.1f} Hz")
    
    print("\n✅ 测试完成！")
    
    return results

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SmolVLA + LIBERO 测试 (WSL2修复版)")
    print("="*60 + "\n")
    
    try:
        results = test_smolvla_on_libero()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ 程序错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)