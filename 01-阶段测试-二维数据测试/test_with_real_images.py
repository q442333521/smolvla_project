"""
使用真实多视角图像测试SmolVLA
使用公开数据集或网络图像
"""
import torch
import numpy as np
from PIL import Image
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import torchvision.transforms as transforms
import urllib.request
import os

print("🔍 使用真实多视角图像测试SmolVLA\n")

# 加载模型
print("加载模型...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda").float().eval()
print(f"✅ 模型加载\n")

tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)

# 准备真实图像
print("准备测试图像...\n")

# 方案1: 使用示例机器人场景图像
# 这些是常见的机器人操作场景
test_images = {
    "camera1": "https://raw.githubusercontent.com/google-research/robopianist/main/docs/images/robopianist_teaser.png",
    "camera2": "https://raw.githubusercontent.com/google-research/robopianist/main/docs/images/robopianist_teaser.png",  # 会做变换
    "camera3": "https://raw.githubusercontent.com/google-research/robopianist/main/docs/images/robopianist_teaser.png"   # 会做变换
}

# 如果下载失败，使用本地创建的更真实的图像
def create_realistic_robot_scene():
    """创建更真实的机器人场景图像"""
    from PIL import ImageDraw, ImageFilter
    
    # 创建背景（木桌纹理）
    img = Image.new('RGB', (256, 256), color=(210, 180, 140))
    draw = ImageDraw.Draw(img)
    
    # 添加桌面纹理（木纹）
    for i in range(0, 256, 4):
        shade = 210 + np.random.randint(-20, 20)
        draw.line([(0, i), (256, i)], fill=(shade, shade-30, shade-70), width=2)
    
    # 添加阴影
    shadow = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.ellipse([60, 140, 140, 180], fill=(0, 0, 0, 50))
    img.paste(Image.alpha_composite(img.convert('RGBA'), shadow).convert('RGB'))
    
    # 绘制目标物体（红色立方体）
    # 正面
    cube_top_left = (160, 100)
    cube_size = 40
    draw.polygon([
        cube_top_left,
        (cube_top_left[0] + cube_size, cube_top_left[1]),
        (cube_top_left[0] + cube_size, cube_top_left[1] + cube_size),
        (cube_top_left[0], cube_top_left[1] + cube_size)
    ], fill=(200, 50, 50), outline=(150, 30, 30), width=2)
    
    # 侧面（深红色）
    draw.polygon([
        (cube_top_left[0] + cube_size, cube_top_left[1]),
        (cube_top_left[0] + cube_size + 20, cube_top_left[1] - 10),
        (cube_top_left[0] + cube_size + 20, cube_top_left[1] - 10 + cube_size),
        (cube_top_left[0] + cube_size, cube_top_left[1] + cube_size)
    ], fill=(150, 30, 30), outline=(100, 20, 20), width=1)
    
    # 顶面（亮红色）
    draw.polygon([
        cube_top_left,
        (cube_top_left[0] + 20, cube_top_left[1] - 10),
        (cube_top_left[0] + cube_size + 20, cube_top_left[1] - 10),
        (cube_top_left[0] + cube_size, cube_top_left[1])
    ], fill=(220, 80, 80), outline=(180, 50, 50), width=1)
    
    # 绘制机械臂末端（灰色夹爪）
    gripper_pos = (80, 140)
    # 夹爪底座
    draw.rectangle([gripper_pos[0]-15, gripper_pos[1]-20, 
                   gripper_pos[0]+15, gripper_pos[1]], 
                  fill=(100, 100, 120), outline=(70, 70, 90), width=2)
    # 夹爪指
    draw.rectangle([gripper_pos[0]-18, gripper_pos[1], 
                   gripper_pos[0]-12, gripper_pos[1]+25], 
                  fill=(120, 120, 140), outline=(90, 90, 110), width=1)
    draw.rectangle([gripper_pos[0]+12, gripper_pos[1], 
                   gripper_pos[0]+18, gripper_pos[1]+25], 
                  fill=(120, 120, 140), outline=(90, 90, 110), width=1)
    
    # 添加一些噪点增加真实感
    img_array = np.array(img)
    noise = np.random.randint(-5, 5, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # 轻微模糊增加真实感
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return img

# 创建3个视角的真实感图像
print("创建真实感场景图像...\n")

# Camera 1: 正面视角
img_front = create_realistic_robot_scene()
img_front.save("test_camera1_front.png")
print("✅ Camera 1 (正面): test_camera1_front.png")

# Camera 2: 侧面视角（旋转30度）
img_side = create_realistic_robot_scene().rotate(30, expand=False, fillcolor=(210, 180, 140))
img_side.save("test_camera2_side.png")
print("✅ Camera 2 (侧面): test_camera2_side.png")

# Camera 3: 俯视角度（添加透视变换）
img_top = create_realistic_robot_scene()
# 简单的透视变换（模拟俯视）
img_top = img_top.transform(img_top.size, Image.PERSPECTIVE, 
                            (0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9),
                            fillcolor=(210, 180, 140))
img_top.save("test_camera3_top.png")
print("✅ Camera 3 (俯视): test_camera3_top.png\n")

# 调整图像大小到256x256
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

img1_tensor = transform(img_front).cuda().float()
img2_tensor = transform(img_side).cuda().float()
img3_tensor = transform(img_top).cuda().float()

# 准备机器人状态（假设夹爪在目标物体左侧）
robot_state = np.array([
    -0.3,  # X: 在左侧
    0.0,   # Y: 中间
    0.15,  # Z: 稍低
    0.0, 0.0, 0.0  # 姿态
])

# 测试不同的指令
instructions = [
    "pick up the red cube",
    "grasp the red object",
    "move to the red cube and pick it up",
]

print("=" * 60)
print("开始测试真实图像推理")
print("=" * 60)

for i, instruction in enumerate(instructions):
    print(f"\n测试 {i+1}/{len(instructions)}: '{instruction}'")
    print("-" * 60)
    
    # Token化
    tokens = tokenizer(
        instruction,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    )
    
    # 准备observation
    observation = {
        "observation.images.camera1": img1_tensor.unsqueeze(0),
        "observation.images.camera2": img2_tensor.unsqueeze(0),
        "observation.images.camera3": img3_tensor.unsqueeze(0),
        "observation.state": torch.from_numpy(robot_state).float().cuda().unsqueeze(0),
        "observation.language.tokens": tokens["input_ids"].cuda(),
        "observation.language.attention_mask": tokens["attention_mask"].cuda().bool(),
    }
    
    # 推理
    policy._queues["action"].clear()
    
    import time
    start = time.time()
    with torch.no_grad():
        action = policy.select_action(observation)
    inference_time = (time.time() - start) * 1000
    
    action_np = action[0].cpu().numpy()
    
    print(f"✅ 推理成功")
    print(f"   时间: {inference_time:.1f}ms")
    print(f"   输出: {action_np}")
    print(f"\n   动作分析:")
    print(f"   X方向: {action_np[0]:.4f} {'→' if action_np[0] > 0 else '←'}")
    print(f"   Y方向: {action_np[1]:.4f} {'↑' if action_np[1] > 0 else '↓'}")
    print(f"   Z方向: {action_np[2]:.4f} {'⬆' if action_np[2] > 0 else '⬇'}")
    print(f"   动作幅度: {np.linalg.norm(action_np[:3]):.4f}")
    
    # 获取完整动作序列分析
    from lerobot.policies.utils import populate_queues
    policy._queues = populate_queues(policy._queues, observation, exclude_keys=["action"])
    
    with torch.no_grad():
        action_chunk = policy._get_action_chunk(observation, noise=None)
    
    actions = action_chunk[0].cpu().numpy()
    
    print(f"\n   完整序列 (50步) 统计:")
    print(f"   X: 均值={actions[:, 0].mean():.3f}, 标准差={actions[:, 0].std():.3f}")
    print(f"   Y: 均值={actions[:, 1].mean():.3f}, 标准差={actions[:, 1].std():.3f}")
    print(f"   Z: 均值={actions[:, 2].mean():.3f}, 标准差={actions[:, 2].std():.3f}")

# 分析结果
print("\n" + "=" * 60)
print("📊 测试结果分析")
print("=" * 60)

print("\n关键观察:")
print("1. 模型是否响应不同指令？")
print("2. 动作方向是否合理？（应该向右向物体）")
print("3. 动作序列是否随图像变化？")

print("\n💡 对比简化环境:")
print("  - 真实感图像包含: 纹理、阴影、3D效果")
print("  - 简化环境只有: 纯色几何图形")
print("  - 如果真实图像表现仍差，说明需要更真实的环境")

print("\n📁 生成的图像:")
print("  - test_camera1_front.png")
print("  - test_camera2_side.png")
print("  - test_camera3_top.png")

print("\n✅ 测试完成！")
