from simulate import *
import torch
from torch import nn
import time
import argparse

torch.manual_seed(time.time())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def backward_hook(module, grad_input, grad_output):
    print("Inside " + module.__class__.__name__ + " backward")
    print("grad_input: ", type(grad_input))
    print("grad_output: ", type(grad_output))
    return None

def compare_conv2d_gradients(kernel_size, input_size, input_channels):
    # prepare input and baseline Conv2d
    test_input = torch.rand(1, input_channels, input_size, input_size, dtype=torch.float64, device=device)
    test_output = torch.rand(1, 1, (input_size - kernel_size) + 1, (input_size - kernel_size) + 1, dtype = torch.float64, device = device) * 3.2

    test_input.requires_grad_(True)
    test_output.requires_grad_(True)

    if test_input.grad is not None:
        test_input.grad.zero_()
    if test_output.grad is not None:
        test_output.grad.zero_()

    crit = torch.nn.MSELoss(reduction = "sum")

    conv2d = nn.Conv2d(
        in_channels=input_channels,
        out_channels=1,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
        bias=True,
        dtype=torch.float64
    ).to(device)
    with torch.no_grad():
        conv2d.weight.fill_(1.0)
        conv2d.bias.fill_(1.0)

    # baseline backward
    def grad_baseline():
        conv2d.zero_grad()
        out_base = conv2d(test_input)
        print(test_output.shape)
        print(out_base.shape)
        loss_base = crit(out_base, test_output)
        loss_base.backward()
        grad_w_base = conv2d.weight.grad.detach().clone()
        grad_b_base = conv2d.bias.grad.detach().clone()
        return grad_w_base, grad_b_base


    print("==== BASELINE ====")
    gwb, gbb = time_function(grad_baseline)
    gib = test_input.grad.detach().clone()
    gob = test_output.grad.detach().clone()
    print("grad weight",gwb)
    print("grad bias",gbb)
    print("grad input",gib)
    print("grad output",gob)

    # prepare Conv2dQ parameters (same values)
    k1 = nn.Parameter(torch.ones(1, input_channels, kernel_size, kernel_size, dtype=torch.float64, device=device), requires_grad=True)
    kb1 = nn.Parameter(torch.ones(1, dtype=torch.float64, device=device), requires_grad=True)

    if k1.grad is not None:
        k1.grad.zero_()
    if kb1.grad is not None:
        kb1.grad.zero_()

    # forward/backward through Conv2dQ (same signature as used elsewhere)
    def grad_q():
        out_q = Conv2dQ.apply(test_input, k1, kb1, 1, 0, 1)
        loss_q = crit(out_q, test_output)
        loss_q.backward()
        grad_w_q = k1.grad.detach().clone()
        grad_b_q = kb1.grad.detach().clone()
        return grad_w_q, grad_b_q

    if test_input.grad is not None:
        test_input.grad.zero_()
    if test_output.grad is not None:
        test_output.grad.zero_()

    print("==== FP ====")
    gwq, gbq = time_function(grad_q)
    giq = test_input.grad.detach().clone()
    goq = test_output.grad.detach().clone()
    print("grad weight",gwq)
    print("grad bias",gbq)
    print("grad input",giq)
    print("grad output",goq)

    # compare gradients

    report_error("weight grad", gwq, gwb)
    report_error("bias grad  ", gbq, gbb)
    report_error("input grad ", giq, gib)
    report_error("output grad", goq, gob)

def compare_linear_gradients(input_size):
    # prepare input and baseline Conv2d
    test_input = torch.rand(1, input_size, dtype=torch.float64, device=device)
    test_output = torch.rand(1, 4, dtype = torch.float64, device = device)

    test_input.requires_grad_(True)
    test_output.requires_grad_(True)

    if test_input.grad is not None:
        test_input.grad.zero_()
    if test_output.grad is not None:
        test_output.grad.zero_()

    crit = torch.nn.MSELoss(reduction = "sum")

    blayer = nn.Linear(in_features=input_size, out_features=4, bias=True, dtype=torch.float64).to(device)
    
    with torch.no_grad():
        blayer.weight.fill_(1.0)
        blayer.bias.fill_(1.0)

    # baseline backward
    def grad_baseline():
        blayer.zero_grad()
        out_base = blayer(test_input)
        loss_base = crit(out_base, test_output)
        loss_base.backward()
        grad_w_base = blayer.weight.grad.detach().clone()
        grad_b_base = blayer.bias.grad.detach().clone()
        return grad_w_base, grad_b_base


    print("==== BASELINE ====")
    gwb, gbb = time_function(grad_baseline)
    gib = test_input.grad.detach().clone()
    gob = test_output.grad.detach().clone()
    print("grad weight",gwb)
    print("grad bias",gbb)
    print("grad input",gib)
    print("grad output",gob)

    # prepare Conv2dQ parameters (same values)
    w3 = nn.Parameter(torch.ones(4, input_size, dtype=torch.float64, device = device), requires_grad = True) 
    b3 = nn.Parameter(torch.ones(4, dtype=torch.float64, device = device), requires_grad = True)

    if w3.grad is not None:
        w3.grad.zero_()
    if b3.grad is not None:
        b3.grad.zero_()

    if test_input.grad is not None:
        test_input.grad.zero_()
    if test_output.grad is not None:
        test_output.grad.zero_()

    # forward/backward through Conv2dQ (same signature as used elsewhere)
    def grad_q():
        out_q = LinearQ.apply(test_input, w3, b3)
        loss_q = crit(out_q, test_output)
        loss_q.backward()
        grad_w_q = w3.grad.detach().clone()
        grad_b_q = b3.grad.detach().clone()
        return grad_w_q, grad_b_q


    print("==== FP ====")
    gwq, gbq = time_function(grad_q)
    giq = test_input.grad.detach().clone()
    goq = test_output.grad.detach().clone()
    print("grad weight",gwq)
    print("grad bias",gbq)
    print("grad input",giq)
    print("grad output",goq)

    # compare gradients

    report_error("weight grad", gwq, gwb)
    report_error("bias grad  ", gbq, gbb)
    report_error("input grad ", giq, gib)
    report_error("output grad", goq, gob)

def unit_test_batchnorm_grad(input_size, input_channels):    
    test_input = torch.abs(torch.rand(1, input_channels, input_size, input_size, dtype=torch.float64, device=device))*-4.5
    test_output = torch.randn(1, input_channels, input_size, input_size, dtype=torch.float64, device=device)
    print("test_input", test_input)

    test_input.requires_grad_(True)
    test_output.requires_grad_(True)

    if test_input.grad is not None:
        test_input.grad.zero_()
    if test_output.grad is not None:
        test_output.grad.zero_()
    
    crit = torch.nn.MSELoss(reduction = "sum")

    blayer = nn.BatchNorm2d(num_features=input_channels, dtype=torch.float64, affine=True, track_running_stats=True).to(device)
    with torch.no_grad():
        blayer.weight.fill_(1.0)
        blayer.bias.fill_(1.0)
        blayer.running_mean.fill_(0.0)
        blayer.running_var.fill_(1.0)

    # baseline backward
    def grad_baseline():
        blayer.zero_grad()
        if test_input.grad is not None:
            test_input.grad.zero_()
        out_base = blayer(test_input)
        loss_base = crit(out_base, test_output)
        loss_base.backward()
        grad_w_base = blayer.weight.grad.detach().clone()
        grad_b_base = blayer.bias.grad.detach().clone()
        return grad_w_base, grad_b_base

    print("==== BASELINE ====")
    gwb, gbb = time_function(grad_baseline)
    gib = test_input.grad.detach().clone()
    gob = test_output.grad.detach().clone()
    print("grad weight", gwb)
    print("grad bias", gbb)
    print("grad input", gib)
    print("grad output", gob)

    if test_input.grad is not None:
        test_input.grad.zero_()
    if test_output.grad is not None:
        test_output.grad.zero_()

    # prepare Conv2dQ parameters (same values)
    bnw1 = nn.Parameter(torch.ones(input_channels, dtype=torch.float64, device = device), requires_grad = True)
    bnb1 = nn.Parameter(torch.ones(input_channels, dtype=torch.float64, device = device), requires_grad = True)
    running_mean1 = nn.Parameter(torch.zeros(input_channels, dtype=torch.float64, device = device), requires_grad=False)
    running_var1 = nn.Parameter(torch.ones(input_channels, dtype=torch.float64, device = device), requires_grad=False)

    if bnw1.grad is not None:
        bnw1.grad.zero_()
    if bnb1.grad is not None:
        bnb1.grad.zero_()

    # forward/backward through Conv2dQ (same signature as used elsewhere)
    def grad_q():
        if test_input.grad is not None:
            test_input.grad.zero_()
        out_q = BatchNorm2dQ.apply(test_input, running_mean1, running_var1, bnw1, bnb1)
        loss_q = crit(out_q, test_output)
        loss_q.backward()
        grad_w_q = bnw1.grad.detach().clone()
        grad_b_q = bnb1.grad.detach().clone()
        grad_in_q = test_input.grad.detach().clone()
        return grad_w_q, grad_b_q

    print("==== FP ====")
    gwq, gbq = time_function(grad_q)
    giq = test_input.grad.detach().clone()
    goq = test_output.grad.detach().clone()
    print("grad weight", gwq)
    print("grad bias", gbq)
    print("grad input", giq)
    print("grad output", goq)

    # compare gradients
    report_error("weight grad ", gwq, gwb)
    report_error("bias grad   ", gbq, gbb)
    report_error("input grad  ", giq, gib)
    report_error("output grad ", goq, gob)

    diff_rm = torch.abs(blayer.running_mean - running_mean1)
    diff_rv = torch.abs(blayer.running_var - running_var1)
    if(torch.equal(blayer.running_mean, diff_rm)):
        print("Running mean didn't update")
    if(torch.equal(blayer.running_var, diff_rv)):
        print("Running var didn't update")

    report_error("running mean", running_mean1, blayer.running_mean)
    report_error("running var ", running_var1, blayer.running_var)

    print("running mean", running_mean1)
    print("running var", running_var1)

    return 0

def unit_test_conv2d(kernel_size, input_size, input_channels):

    # TODO
    test_input = torch.rand(1,input_channels, input_size, input_size, dtype=torch.float64, device = device)
    print("TESTING WITH INPUT:\n",test_input)

    conv2d = nn.Conv2d(
        in_channels=input_channels, 
        out_channels=1, 
        kernel_size=kernel_size, 
        stride=1, 
        padding=0, 
        bias=True, 
        padding_mode='zeros', 
        dtype=torch.float64
    ).to(device)
    with torch.no_grad():
        conv2d.weight.fill_(1.0)
        conv2d.bias.fill_(1.0)

    print("===BASELINE===")
    print(baseline := time_function(conv2d, test_input))

    k1 = nn.Parameter(torch.ones(1,input_channels,kernel_size,kernel_size, dtype=torch.float64, device = device), requires_grad = True)
    kb1 = nn.Parameter(torch.ones(1, dtype=torch.float64, device = device), requires_grad = True)

    # everything is ones so that we can have calculable result.
    conv = Conv2dQ.apply
    print("===FP===")
    print(fp := time_function(conv, test_input, k1, kb1, 1, 0, 1) )

    report_error("conv2d forward", fp, baseline)



    
def unit_test_linear(input_size):
    test_input = torch.rand(1,input_size, dtype=torch.float64, device = device)
    print("TESTING WITH INPUT:\n",test_input)

    linear = nn.Linear(in_features=input_size, out_features=4, bias=True, dtype=torch.float64).to(device)
    with torch.no_grad():
        linear.weight.fill_(1.0)
        linear.bias.fill_(1.0)


    print("===BASELINE===\n")
    baseline = time_function(linear, test_input)
    print(baseline)

    w3 = nn.Parameter(torch.ones(4, input_size, dtype=torch.float64, device = device), requires_grad = True) 
    b3 = nn.Parameter(torch.ones(4, dtype=torch.float64, device = device), requires_grad = True)

    linearq = LinearQ.apply
    print("===FP===\n")
    fp = time_function(linearq, test_input, w3, b3)
    print(fp)

    report_error("linear forward", fp, baseline)

def unit_test_batchnorm(input_size, input_channels):
    test_input = torch.rand(1,input_channels,32,32, dtype=torch.float64, device = device) * -8
    print("TESTING WITH INPUT:\n",test_input)

    batchnorm = nn.BatchNorm2d(num_features=input_channels, dtype=torch.float64, affine=True, track_running_stats=True).to(device)
    with torch.no_grad():
        batchnorm.weight.fill_(1.0)
        batchnorm.bias.fill_(1.0)
        batchnorm.running_mean.fill_(0.0)
        batchnorm.running_var.fill_(1.0)

    print("===BASELINE===")
    baseline = time_function(batchnorm, test_input)
    print(baseline)
    
    bnw1 = nn.Parameter(torch.ones(input_channels, dtype=torch.float64, device = device), requires_grad = True)
    bnb1 = nn.Parameter(torch.ones(input_channels, dtype=torch.float64, device = device), requires_grad = True)
    running_mean1 = nn.Parameter(torch.zeros(input_channels, dtype=torch.float64, device = device), requires_grad=False)
    running_var1 = nn.Parameter(torch.ones(input_channels, dtype=torch.float64, device = device), requires_grad=False)

    batchnormq = BatchNorm2dQ.apply
    print("===FP===\n",)
    fp = time_function(batchnormq, test_input, running_mean1,running_var1 ,bnw1, bnb1)
    print(fp)

    report_error("batchnorm forward", fp, baseline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='conv2d', help='which unit test to run: conv2d, linear, batchnorm')

    parser.add_argument("--kernel_size", type = int, default = 2, help = "kernel size for conv2d test")
    parser.add_argument("--input_size", type = int, default = 4, help = "kernel size for conv2d test")
    parser.add_argument("--input_channels", type = int, default = 1, help = "kernel size for conv2d test")
    
    
    args = parser.parse_args()

    if args.test == 'conv2d':
        unit_test_conv2d(args.kernel_size, args.input_size, args.input_channels)
    elif args.test == 'linear':
        unit_test_linear(args.input_size)
    elif args.test == 'batchnorm':
        unit_test_batchnorm(args.input_size, args.input_channels)
    elif args.test == "conv2db":
        compare_conv2d_gradients(args.kernel_size, args.input_size, args.input_channels)
    elif args.test == "linearb":
        compare_linear_gradients(args.input_size)
    elif args.test == "batchnormb":
        unit_test_batchnorm_grad(args.input_size, args.input_channels)
    else:
        print("Unknown test. Available tests: conv2d, linear, batchnorm")
