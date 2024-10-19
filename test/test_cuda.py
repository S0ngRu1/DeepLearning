import torch

# 检查CUDA是否可用
is_cuda_available = torch.cuda.is_available()
print(f"CUDA available: {is_cuda_available}")

# 如果CUDA可用，尝试获取CUDA设备数量
if is_cuda_available:
    device_count = torch.cuda.device_count()
    print(f"Device count: {device_count}")

    # 尝试选择第一个GPU并创建一个tensor
    if device_count > 0:
        device = torch.device(f"cuda:0")
        tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"Tensor on GPU: {tensor}")