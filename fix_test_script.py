#!/usr/bin/env python3
"""修复 download_and_test_dataset.py 中的参数问题"""

import re

# 读取文件
with open('/root/smolvla_project/download_and_test_dataset.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 定义要替换的旧代码模式
old_pattern = r'''        # 加载SmolVLA模型
        print\("\\n加载SmolVLA模型..."\)
        policy = SmolVLAPolicy\.from_pretrained\(
            "lerobot/smolvla_base",
            torch_dtype=torch\.float16,
            device="cuda" if torch\.cuda\.is_available\(\) else "cpu"
        \)
        policy\.eval\(\)
        print\("✅ 模型加载成功"\)'''

# 新的正确代码
new_code = '''        # 加载SmolVLA模型
        print("\\n加载SmolVLA模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # from_pretrained 不接受 torch_dtype 和 device 参数
        # 需要先加载，然后手动设置设备和精度
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        
        # 根据你的测试结果，统一使用 float32 避免 dtype 不匹配
        policy = policy.to(device).float()
        policy.eval()
        
        print(f"✅ 模型加载成功 (device={device}, dtype=float32)")'''

# 使用正则表达式替换
content = re.sub(old_pattern, new_code, content, flags=re.MULTILINE)

# 如果正则替换失败，使用简单的字符串替换
if 'torch_dtype=torch.float16' in content:
    old_text = '''        # 加载SmolVLA模型
        print("\\n加载SmolVLA模型...")
        policy = SmolVLAPolicy.from_pretrained(
            "lerobot/smolvla_base",
            torch_dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        policy.eval()
        print("✅ 模型加载成功")'''
    
    new_text = '''        # 加载SmolVLA模型
        print("\\n加载SmolVLA模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # from_pretrained 不接受 torch_dtype 和 device 参数
        # 需要先加载,然后手动设置设备和精度
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        
        # 根据你的测试结果,统一使用 float32 避免 dtype 不匹配
        policy = policy.to(device).float()
        policy.eval()
        
        print(f"✅ 模型加载成功 (device={device}, dtype=float32)")'''
    
    content = content.replace(old_text, new_text)

# 写回文件
with open('/root/smolvla_project/download_and_test_dataset.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 修复完成!")
print("\n修改内容:")
print("- 移除了 torch_dtype 和 device 参数")
print("- 改为先加载模型,然后使用 .to(device).float()")
print("- 使用 float32 避免 dtype 不匹配问题")
