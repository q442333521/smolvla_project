# 在DataLoader之前添加这个函数
def create_collate_fn():
    """创建collate函数来重映射图像key"""
    from torch.utils.data.dataloader import default_collate
    
    def collate_fn(batch_list):
        batch = default_collate(batch_list)
        # 重映射: top->camera1, wrist->camera2和camera3  
        if "observation.images.top" in batch:
            batch["observation.images.camera1"] = batch.pop("observation.images.top")
        if "observation.images.wrist" in batch:
            batch["observation.images.camera2"] = batch["observation.images.wrist"]
            batch["observation.images.camera3"] = batch.pop("observation.images.wrist")
        return batch
    
    return collate_fn

# 使用方法：
# collate_fn = create_collate_fn()
# dataloader = DataLoader(..., collate_fn=collate_fn)
