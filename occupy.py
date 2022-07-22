import os; import torch

def occumpy_mem(cuda_device):
    def check_mem(cuda_device):
        devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
        total, used = devices_info[int(cuda_device)].split(',')
        return total,used
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.85)
    block_mem = max_mem - used
    x = torch.FloatTensor(256,1024,block_mem).to(torch.device(f"cuda:{cuda_device}"))
    del x

occumpy_mem('0')
occumpy_mem('1')