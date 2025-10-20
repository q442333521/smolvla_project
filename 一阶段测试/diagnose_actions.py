"""
诊断SmolVLA的动作输出
查看模型到底在预测什么
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import torchvision.transforms as transforms

print("🔍 SmolVLA 动作输出诊断\n")

# 加载模型
print("加载模型...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda").float().eval()
print(f"✅ 模型加载\n")

tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)

# 创建一个明确的场景
def create_clear_scene(robot_pos, target_pos):
    """创建清晰的场景图像"""
    img = Image.new('RGB', (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 转换坐标
    robot_x = int((robot_pos[0] + 0.5) * 256)
    robot_y = int((robot_pos[1] + 0.5) * 256)
    target_x = int((target_pos[0] + 0.5) * 256)
    target_y = int((target_pos[1] + 0.5) * 256)
    
    # 画目标（大绿圈）
    draw.ellipse([target_x-20, target_y-20, target_x+20, target_y+20],
                 fill=(0, 255, 0), outline=(0, 200, 0), width=3)
    
    # 画机器人（大红圈）
    draw.ellipse([robot_x-15, robot_y-15, robot_x+15, robot_y+15],
                 fill=(255, 0, 0), outline=(200, 0, 0), width=3)
    
    # 画箭头指向目标
    draw.line([(robot_x, robot_y), (target_x, target_y)],
              fill=(0, 0, 255), width=3)
    
    return img

# 测试场景1: 机器人在左，目标在右（应该向右移动）
print("=" * 60)
print("场景1: 机器人在左(-0.2, 0), 目标在右(0.2, 0)")
print("预期动作: X方向正值 (向右)")
print("=" * 60)

robot_state = np.array([-0.2, 0.0, 0.3, 0.0, 0.0, 0.0])
target_pos = np.array([0.2, 0.0, 0.3, 0.0, 0.0, 0.0])

img = create_clear_scene(robot_state[:2], target_pos[:2])
transform = transforms.ToTensor()
img_tensor = transform(img).cuda().float().unsqueeze(0)

tokens = tokenizer(
    "move to the green target",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=77
)

observation = {
    "observation.images.camera1": img_tensor,
    "observation.state": torch.from_numpy(robot_state).float().cuda().unsqueeze(0),
    "observation.language.tokens": tokens["input_ids"].cuda(),
    "observation.language.attention_mask": tokens["attention_mask"].cuda().bool(),
}

policy._queues["action"].clear()

with torch.no_grad():
    action = policy.select_action(observation)

action_np = action[0].cpu().numpy()
print(f"\n预测动作: {action_np}")
print(f"X方向: {action_np[0]:.4f} {'✅ 正确(向右)' if action_np[0] > 0 else '❌ 错误(向左)'}")
print(f"Y方向: {action_np[1]:.4f}")
print(f"动作幅度: {np.linalg.norm(action_np[:2]):.4f}")

# 测试场景2: 机器人在下，目标在上（应该向上移动）
print("\n" + "=" * 60)
print("场景2: 机器人在下(0, -0.2), 目标在上(0, 0.2)")
print("预期动作: Y方向正值 (向上)")
print("=" * 60)

robot_state2 = np.array([0.0, -0.2, 0.3, 0.0, 0.0, 0.0])
target_pos2 = np.array([0.0, 0.2, 0.3, 0.0, 0.0, 0.0])

img2 = create_clear_scene(robot_state2[:2], target_pos2[:2])
img_tensor2 = transform(img2).cuda().float().unsqueeze(0)

observation2 = {
    "observation.images.camera1": img_tensor2,
    "observation.state": torch.from_numpy(robot_state2).float().cuda().unsqueeze(0),
    "observation.language.tokens": tokens["input_ids"].cuda(),
    "observation.language.attention_mask": tokens["attention_mask"].cuda().bool(),
}

policy._queues["action"].clear()

with torch.no_grad():
    action2 = policy.select_action(observation2)

action_np2 = action2[0].cpu().numpy()
print(f"\n预测动作: {action_np2}")
print(f"X方向: {action_np2[0]:.4f}")
print(f"Y方向: {action_np2[1]:.4f} {'✅ 正确(向上)' if action_np2[1] > 0 else '❌ 错误(向下)'}")
print(f"动作幅度: {np.linalg.norm(action_np2[:2]):.4f}")

# 测试场景3: 查看完整的50步动作序列
print("\n" + "=" * 60)
print("场景3: 分析完整动作序列 (50步)")
print("=" * 60)

from lerobot.policies.utils import populate_queues

policy._queues = populate_queues(policy._queues, observation, exclude_keys=["action"])

with torch.no_grad():
    action_chunk = policy._get_action_chunk(observation, noise=None)

actions = action_chunk[0].cpu().numpy()  # (50, 6)

print(f"\n动作序列形状: {actions.shape}")
print(f"\nX方向统计:")
print(f"  均值: {actions[:, 0].mean():.4f}")
print(f"  标准差: {actions[:, 0].std():.4f}")
print(f"  范围: [{actions[:, 0].min():.4f}, {actions[:, 0].max():.4f}]")

print(f"\nY方向统计:")
print(f"  均值: {actions[:, 1].mean():.4f}")
print(f"  标准差: {actions[:, 1].std():.4f}")
print(f"  范围: [{actions[:, 1].min():.4f}, {actions[:, 1].max():.4f}]")

print(f"\n前10步动作:")
for i in range(10):
    print(f"  步{i+1}: X={actions[i, 0]:.4f}, Y={actions[i, 1]:.4f}")

# 保存诊断图像
img.save("diagnose_scene1.png")
img2.save("diagnose_scene2.png")
print(f"\n✅ 诊断图像已保存: diagnose_scene1.png, diagnose_scene2.png")

# 分析结论
print("\n" + "=" * 60)
print("🔍 诊断结论")
print("=" * 60)

x_correct = action_np[0] > 0
y_correct = action_np2[1] > 0

if x_correct and y_correct:
    print("✅ 模型能理解方向！")
    print("   问题可能在于: 动作幅度太小或环境设置")
elif not x_correct and not y_correct:
    print("❌ 模型完全不理解简化图像！")
    print("   原因: Domain gap - 训练数据与简化图像差异太大")
    print("   建议: 使用真实相机图像或更复杂的模拟环境")
else:
    print("⚠️  模型部分理解图像")
    print("   需要进一步调试")

print("\n💡 建议:")
if not (x_correct and y_correct):
    print("  1. ❌ 简化模拟环境不适合SmolVLA")
    print("  2. ✅ 直接进入ROS2集成，使用真实相机")
    print("  3. ✅ 或使用SimplerEnv等标准环境")
else:
    print("  1. ✅ 增加动作缩放因子")
    print("  2. ✅ 检查坐标系统是否一致")
    print("  3. ✅ 优化环境渲染")
