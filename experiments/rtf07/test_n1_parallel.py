import torch
import cpufast_cpp_ops
torch.manual_seed(42)
for tile in (8,16):
 for n,k in ((512,512),(1536,512),(2048,512),(512,2048)):
  x=torch.randn(1,1,k); w=torch.randn(n//tile,k,tile).half(); b=torch.randn(n)
  op=torch.ops.cpufast_cpp.lossless_linear if tile == 8 else torch.ops.cpufast_cpp.lossless_linear16
  torch.set_num_threads(1); a=op(x,w,b)
  torch.set_num_threads(2); c=op(x,w,b)
  assert torch.equal(a,c), (tile,n,k,(a-c).abs().max().item())
print("8 GEMV geometries: exact equality, 1 versus 2 threads")
