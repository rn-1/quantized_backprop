from simulate import *

import torch

def test_frrt():
    print("original",x := torch.rand(2,2)*20)
    print("true val",(1/torch.sqrt(x)))

    x, = to_fixed(x)

    print("approx val",y:=fast_rsqrt(x))

def test_sqrt():
    print("original",x := torch.rand(2,2)*10)
    
    print("true val",(1/torch.sqrt(x)))

    x, = to_fixed(x)
    # x, = to_fixed(*p_truncate(x))
    y = inverse_sqrt(x)
    # y, = p_truncate(y)
    print("inv sqrt approx",y)

def test_ptrunc():
    x = torch.rand((2,2,2))
    y = torch.rand((2,2,2))
    _z = x*y

    print("true z:",_z)
    
    x, y = to_fixed(x, y)
    z = x*y

    print("before trunc",z)
    t, = p_truncation(z) # should be float afterwards? idk
    print("after trunc",t)

if __name__ == "__main__":
    test_sqrt()
    # test_frrt()