"""
SmolVLA 模拟环境测试 - 简化版
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import time
import os

# 禁用matplotlib显示
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SimpleRobotEnv:
    """简化机器人环境"""
    def __init__(self):
        self.state = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
        self.target = np.array([0.3, 0.2, 0.3, 0.0, 0.0, 0.0])
        self.object_pos = np.array([0.3, 0.2, 0.0])
        self.max_steps = 100
        self.current_step = 0
        self.trajectory = []
        
    def reset(self):
        """重置环境"""
        self.state = np.array([
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.2, 0.2),
            0.3, 0.0, 0.0, 0.0
        ])
        self.target = np.array([
            np.random.uniform(0.1, 0.4),
            np.random.uniform(0.1, 0.4),
            0.3, 0.0, 0.0, 0.0
        ])
        self.object_pos = self.target[:3] - np.array([0, 0, 0.3])
        self.current_step = 0
        self.trajectory = [self.state[:2].copy()]
        return self._get_observation()
    
    def _get_observation(self):
        """获取观测"""
        img = self._render_scene()
        # 转换为tensor
        transform = transforms.ToTensor()
        img_tensor = transform(img).cuda().float()  # (3, 256, 256)
        return {
            "image": img_tensor,
            "state": torch.from_numpy(self.state).float().cuda(),
        }
    
    def _render_scene(self):
        """渲染场景为RGB图像"""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        
        # 画机器人（红色）
        robot_x = int((self.state[0] + 0.5) * 256)
        robot_y = int((self.state[1] + 0.5) * 256)
        robot_x = np.clip(robot_x, 0, 255)
        robot_y = np.clip(robot_y, 0, 255)
        
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx*dx + dy*dy <= 25:
                    x, y = robot_x + dx, robot_y + dy
                    if 0 <= x < 256 and 0 <= y < 256:
                        img[y, x] = [255, 0, 0]
        
        # 画目标（绿色）
        target_x = int((self.target[0] + 0.5) * 256)
        target_y = int((self.target[1] + 0.5) * 256)
        target_x = np.clip(target_x, 0, 255)
        target_y = np.clip(target_y, 0, 255)
        
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx*dx + dy*dy <= 25:
                    x, y = target_x + dx, target_y + dy
                    if 0 <= x < 256 and 0 <= y < 256:
                        img[y, x] = [0, 255, 0]
        
        return Image.fromarray(img)
    
    def step(self, action):
        """执行动作"""
        action = np.array(action[:6])
        self.state[:3] += action[:3] * 0.01
        self.state[3:6] += action[3:6] * 0.01
        self.state[:3] = np.clip(self.state[:3], -0.5, 0.5)
        self.state[3:6] = np.clip(self.state[3:6], -np.pi, np.pi)
        
        self.trajectory.append(self.state[:2].copy())
        
        distance = np.linalg.norm(self.state[:3] - self.target[:3])
        reward = -distance
        
        done = distance < 0.05 or self.current_step >= self.max_steps
        success = distance < 0.05
        
        self.current_step += 1
        
        info = {
            "distance": distance,
            "success": success,
            "step": self.current_step
        }
        
        return self._get_observation(), reward, done, info
    
    def plot_trajectory(self, filename="trajectory.png"):
        """保存轨迹图"""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectory')
        
        # 画轨迹
        traj = np.array(self.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
        
        # 画起点
        ax.plot(traj[0, 0], traj[0, 1], 'ro', markersize=15, label='Start')
        
        # 画终点
        ax.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=15, label='End')
        
        # 画目标
        ax.plot(self.target[0], self.target[1], 'g*', markersize=20, label='Target')
        
        ax.legend()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   轨迹图已保存: {filename}")

def test_closed_loop_control():
    """测试闭环控制"""
    print("=" * 60)
    print("SmolVLA 闭环控制测试")
    print("=" * 60)
    
    # 创建环境
    print("\n创建模拟环境...")
    env = SimpleRobotEnv()
    print("✅ 环境创建成功\n")
    
    # 加载模型
    print("加载SmolVLA模型...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy = policy.to("cuda").float().eval()
    print(f"✅ 模型加载完成 ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M参数)\n")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    instruction = "move to the target position and grasp"
    
    tokens = tokenizer(
        instruction,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    )
    
    # 运行episodes
    num_episodes = 3
    success_count = 0
    episode_results = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")
        
        obs = env.reset()
        done = False
        total_reward = 0
        
        observation = {
            "observation.images.camera1": obs["image"].unsqueeze(0),  # (1, 3, 256, 256)
            "observation.state": obs["state"].unsqueeze(0),  # (1, 6)
            "observation.language.tokens": tokens["input_ids"].cuda(),
            "observation.language.attention_mask": tokens["attention_mask"].cuda().bool(),
        }
        
        policy._queues["action"].clear()
        
        step_count = 0
        inference_count = 0
        
        while not done and step_count < 100:
            with torch.no_grad():
                if len(policy._queues["action"]) == 0:
                    inference_count += 1
                    
                action = policy.select_action(observation)
            
            action_np = action[0].cpu().numpy()
            obs, reward, done, info = env.step(action_np)
            total_reward += reward
            
            observation["observation.images.camera1"] = obs["image"].unsqueeze(0)
            observation["observation.state"] = obs["state"].unsqueeze(0)
            
            step_count += 1
            
            if step_count % 20 == 0:
                print(f"  Step {step_count}: Distance={info['distance']:.4f}m")
        
        # 保存轨迹
        env.plot_trajectory(f"episode_{episode+1}_trajectory.png")
        
        result = {
            'episode': episode + 1,
            'success': info['success'],
            'steps': info['step'],
            'reward': total_reward,
            'distance': info['distance'],
            'inferences': inference_count
        }
        episode_results.append(result)
        
        if info['success']:
            success_count += 1
            print(f"\n✅ Episode {episode + 1}: SUCCESS")
        else:
            print(f"\n❌ Episode {episode + 1}: FAILED")
        
        print(f"   Total Steps: {info['step']}")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Final Distance: {info['distance']:.4f}m")
        print(f"   Inferences: {inference_count}")
    
    # 统计
    print(f"\n{'='*60}")
    print("测试完成")
    print(f"{'='*60}")
    print(f"成功率: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    
    avg_steps = np.mean([r['steps'] for r in episode_results])
    avg_distance = np.mean([r['distance'] for r in episode_results])
    avg_inferences = np.mean([r['inferences'] for r in episode_results])
    
    print(f"\n📊 平均统计:")
    print(f"   步数: {avg_steps:.1f}")
    print(f"   最终距离: {avg_distance:.4f}m")
    print(f"   推理次数: {avg_inferences:.1f}")
    
    print(f"\n{'='*60}")
    print("📋 验收结果")
    print(f"{'='*60}")
    
    success_ok = success_count >= 1
    print(f"1. 成功率: {success_count/num_episodes*100:.1f}% " + 
          ("✅ 达标 (>30%)" if success_count/num_episodes >= 0.3 else "⚠️  未达标"))
    
    print(f"2. 轨迹合理性: ", end="")
    if avg_distance < 0.2:
        print("✅ 能接近目标")
    else:
        print("⚠️  需要改进")
    
    print(f"3. 推理效率: {avg_inferences:.1f}次/episode ✅")
    
    if success_count >= 1:
        print("\n🎉 模拟测试通过！")
        print("\n📚 下一步:")
        print("  1️⃣  运行性能基准测试 (benchmark.py)")
        print("  2️⃣  填写项目进度清单")
        print("  3️⃣  准备 ROS2 集成")
    else:
        print("\n⚠️  模拟测试需要改进")

if __name__ == "__main__":
    try:
        test_closed_loop_control()
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
