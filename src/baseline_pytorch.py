import torch
import time
import numpy as np

B=4
KH=3
KW=3
H=256
W=256
Cin=64
Cout=64

a = torch.from_numpy(np.fromfile("in.txt", dtype=np.double).reshape([B,Cin,H,W]))
k = torch.from_numpy(np.fromfile("kernel.txt", dtype=np.double).reshape([Cout,Cin,KH,KW]))
out = torch.from_numpy(np.fromfile("out.txt", dtype=np.double).reshape([B,Cout,H,W]))

for i in range(10):
    time_start=time.time()
    p=torch.nn.functional.conv2d(a,k,padding=(1,1))
    time_end=time.time()
    
    if p.isclose(out).all().item():
        print("结果一致!")
    else:
        print("结果不一致!")
    print('pytorch time cost',time_end-time_start,'s')