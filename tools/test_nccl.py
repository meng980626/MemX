import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank, local = dist.get_rank(), int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
x = torch.ones(1024, device=f"cuda:{local}") * rank
dist.all_reduce(x)
print(f"rank {rank} OK, sum={x[0].item()}", flush=True)
dist.destroy_process_group()