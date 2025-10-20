#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/smolvla_project/lerobot/src')
import torch, numpy as np, time
from pathlib import Path
import pyarrow.parquet as pq
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer

print("\n" + "="*60)
print("使用数据集测试 SmolVLA")
print("="*60 + "\n")

device = "cuda"
df = pq.read_table(Path("/root/smolvla_project/datasets/lerobot_pusht/data/chunk-000/file-000.parquet")).to_pandas()
print(f"✅ 数据: {len(df)} 样本")

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).float().eval()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
print(f"✅ 模型加载\n")

print("测试 5 个样本:\n")
times = []
for i in range(5):
    state = torch.tensor(df.iloc[i]['observation.state'], dtype=torch.float32)
    if len(state) < 14:
        state = torch.cat([state, torch.zeros(14-len(state))])
    
    img = torch.rand(1,3,256,256).to(device)
    tokens = tokenizer("Push the block", return_tensors="pt")
    
    obs = {
        "observation.images.camera1": img,
        "observation.images.camera2": img.clone(),
        "observation.images.camera3": img.clone(),
        "observation.state": state.unsqueeze(0).to(device),
        "observation.language.tokens": tokens['input_ids'].to(device),
        "observation.language.attention_mask": tokens['attention_mask'].to(device).bool(),
    }
    
    start = time.time()
    with torch.no_grad():
        action = policy.select_action(obs)
    t = (time.time()-start)*1000
    times.append(t)
    
    print(f"  [{i+1}] {t:.1f}ms - 动作: {action.shape}")

print(f"\n平均: {np.mean(times):.1f}ms, 频率: {1000/np.mean(times):.1f}Hz")
print(f"显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
print("\n✅ 完成！这就是用数据集复现 SmolVLA 的方法\n")
