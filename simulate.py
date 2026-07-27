from torch import nn
# qtorch is a very useful library here.
import torch
import numpy as np
import math
import time

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import random # we will prob need something better for this
import argparse
from tqdm import tqdm

BITs = 22

# TODO
# Conv2d needs float --> int to float then calculate, bc it doesnt change should be fine.
# everything else should be in form of integers with last 7 bits for decimal interpretation.

# ** drawing board

def fast_rsqrt(x):
    # x is a fp number
    threehalfs, = to_fixed(torch.tensor([1.5]))
    onehalf, = to_fixed(torch.tensor([0.5]))
    x2 = x * onehalf
    x2, = to_fixed(*p_truncate(x2))
    magic_number = 1597463007 * 2**7 # 38 bits, fits
    # we need a float number for this to work. hmmm.
    
    # undefined for truncation
    x2, = to_float_no_shift(x2)
    bithacking = x2.view(torch.int64).clone()
    bithacking = magic_number - bithacking
    y = bithacking.view(torch.float64).clone()
    # undef
    
    y, = to_fixed_no_shift(y) 
    
    x2, = to_fixed(x)
    y2, = to_fixed(*p_truncate(y**2))
    y2, = to_fixed(*p_truncate(x2*y))
    y2 = threehalfs - y
    y, = to_fixed(*p_truncate(y*y2))

    print(*to_float(y))
    return y


# ** operation approx

def pseudosep(x):
    # decompose into fractional and exp part
    # note that x is fp number
    i_star = torch.floor(torch.log2(x)).to(torch.int64) #
    exp = i_star + 1 - BITs
    # print("x",x)
    # print("exp",exp)


    # exp + 1 goes negative once x < 0.25, and a right shift by a negative amount
    # yields 0, which collapses the polynomial below to the constant c.
    shift = exp + 1
    frac = torch.where(
        shift >= 0,
        torch.bitwise_right_shift(x, torch.clamp(shift, min=0)),
        torch.bitwise_left_shift(x, torch.clamp(-shift, min=0)),
    ) # both branches evaluate, so clamp keeps each shift amount valid
    # print(frac/2**BITs)
    # print(frac.dtype)
    # print(frac)

    # we still need sqrt compensation
    # do we need extra bits for compensation?

    return frac, exp

def inverse_sqrt(x):
    # adapted from lu et al
    # assume that x is an fp number
    a, = to_fixed(torch.tensor([4.63887], dtype=torch.float64, device = x.device))
    b, = to_fixed(torch.tensor([5.77789], dtype=torch.float64, device = x.device))
    c, = to_fixed(torch.tensor([3.14736], dtype=torch.float64, device = x.device))

    # print("a,b,c || ",a,b,c)
    u, e = pseudosep(x) # not sure what to do with yet
    # print("u",u," e",e)

    p, = to_fixed(*p_truncate(a*u))
    p, = to_fixed(*p_truncate((p - b) * u))
    g = p + c
    m_float = torch.pow(torch.tensor(2.0, device=x.device), -(e.float() + 1.0)/2.0)
    m, = to_fixed(m_float)
    g_mult = (g * m)
    # g ~= 1/sqrt(u)
    # print(g_mult.dtype)

    result, = p_truncate(g_mult)

    # comparison
    base, = to_float(x)
    # print("expected inv sqrt:", 1/torch.sqrt(base))
    # print("approx inv sqrt:", result)

    return result


# ** Evaluation functions

# CNNModelBaseline and CNNModelQ are the same architecture layer-for-layer, but the baseline
# is built from stock nn modules (conv1.weight) while CNNModelQ holds bare Parameters (k1),
# so no name ever lines up. zipping named_parameters() positionally is not a fix either: the
# baseline keeps running stats as buffers whereas CNNModelQ keeps them as Parameters, so the
# lists are different lengths (18 vs 24) and a positional zip mispairs everything after bn1.
BASELINE_TO_Q = {
    "conv1.weight": "k1",   "conv1.bias": "kb1",
    "bn1.weight":   "bnw1", "bn1.bias":   "bnb1",
    "conv2.weight": "k2",   "conv2.bias": "kb2",
    "bn2.weight":   "bnw2", "bn2.bias":   "bnb2",
    "conv3.weight": "k3",   "conv3.bias": "kb3",
    "bn3.weight":   "bnw3", "bn3.bias":   "bnb3",
    "fc1.weight":   "w1",   "fc1.bias":   "b1",
    "fc2.weight":   "w2",   "fc2.bias":   "b2",
    "fc3.weight":   "w3",   "fc3.bias":   "b3",
}

# running stats live as buffers on the baseline but as Parameters on CNNModelQ, so they are
# looked up separately.
BASELINE_RUNNING_TO_Q = {
    "bn1.running_mean": "running_mean1", "bn1.running_var": "running_var1",
    "bn2.running_mean": "running_mean2", "bn2.running_var": "running_var2",
    "bn3.running_mean": "running_mean3", "bn3.running_var": "running_var3",
}

def paired_tensors(model_base, model_Q, include_running=True):
    # yield (name, baseline_tensor, quantized_tensor) for each mapped pair. shapes are
    # asserted so a rename or an architecture change fails loudly here rather than silently
    # comparing unrelated tensors, which is the failure the positional zip allowed.
    params_b = dict(model_base.named_parameters())
    buffers_b = dict(model_base.named_buffers())
    params_q = dict(model_Q.named_parameters())

    pairs = list(BASELINE_TO_Q.items())
    if include_running:
        pairs += list(BASELINE_RUNNING_TO_Q.items())

    for bname, qname in pairs:
        base = params_b.get(bname, buffers_b.get(bname))
        quant = params_q.get(qname)
        assert base is not None, f"baseline has no parameter/buffer named {bname}"
        assert quant is not None, f"quantized model has no parameter named {qname}"
        assert base.shape == quant.shape, \
            f"shape mismatch: {bname} {tuple(base.shape)} vs {qname} {tuple(quant.shape)}"
        yield bname, base, quant

def sync_models(model_base, model_Q):
    # copy the baseline's weights into the quantized model so both start from identical
    # values. without this the two are independently initialised and any comparison between
    # them measures the gap between two random inits, not quantization error. the baseline is
    # the reference, and it is float32 while CNNModelQ is float64 -- the widening cast is
    # exact, so the copy loses nothing.
    with torch.no_grad():
        for name, base, quant in paired_tensors(model_base, model_Q):
            quant.copy_(base.to(quant.dtype))

def calculateFinalParamError(model1, model2):
    # RMSE over every mapped parameter pair, normalised by the baseline's own RMS so the
    # result is a fraction of its scale rather than a raw magnitude (see rel_error). model1
    # is the baseline reference. running stats are excluded: they are not parameters on the
    # baseline. returns nan if the baseline params are all ~0, leaving nothing to be
    # relative to.
    total = 0
    total_ref = 0
    total_params = 0

    for name, base, quant in paired_tensors(model1, model2, include_running=False):
        ref = base.detach().cpu().to(torch.float64)
        q = quant.detach().cpu().to(torch.float64)
        total += torch.sum((ref - q)**2).item()
        total_ref += torch.sum(ref**2).item()
        total_params += ref.numel()

    rmse = math.sqrt(total / total_params)
    ref_rms = math.sqrt(total_ref / total_params)
    return rmse / ref_rms if ref_rms > 1e-12 else float("nan")

def time_function(f, *args):
    if not callable(f):
        print("Cannot time a non function.")
        return
    else:
        start = time.time()
        out = f(*args)
        end = time.time()
        print(f"Total time taken: {end - start} seconds.") # we can format this later, idc for now.
        return out


def forward_pass_only(model, input):
    def run_model():
        output = model(input)
        return output

    output = time_function(run_model)
    print(output)
    return output


# ** Utility functions

def to_fixed_no_shift(*tensors):
    return (torch.round(tensor.to(torch.int64)) for tensor in tensors) # less expensive to_fixed that correctly deals with prior float convolution mult for later truncation.

def to_float_no_shift(*tensors):
    return (tensor.to(torch.float64) for tensor in tensors) # less expensive to float for conv2D. don't shift so we can truncate correctly later.

def to_fixed(*tensors):
    return (torch.round(tensor * (2**BITs)).to(torch.int64) for tensor in tensors)

def to_float(*tensors):
    return (tensor.to(torch.float64) / 2**BITs for tensor in tensors)

def div_public(tensor, divisor):
    # divide a fixed-point tensor by a public integer, staying at the same scale.
    # the divisor is public, so its reciprocal costs no precision to hold -- the only
    # loss is rounding the quotient back onto the 2**-BITs grid. cast to float64 first:
    # int64 / int in torch yields float32, which drops bits once the operand is large.
    return torch.round(tensor.to(torch.float64) / divisor).to(torch.int64)


# ** accuracy reporting

def rel_error(approx, reference):
    # error of a quantized tensor against its float baseline, scaled by the baseline's own
    # magnitude. absolute diffs are not comparable across tensors: grad_weight is a sum over
    # the reduced elements, so |grad_weight| grows with N while |grad_input| does not -- the
    # same 1% error reads as 0.28 at N=16 and 19.1 at N=1024, which looks like a regression
    # that isn't there. relative error stays flat and can be read against the 2**-BITs floor.
    # normalise by mean |reference| rather than per element: an elementwise ratio explodes
    # wherever a reference element happens to sit near zero, which is common in gradients.
    approx, reference = approx.detach(), reference.detach()
    diff = torch.abs(approx - reference).to(torch.float64)
    scale = torch.abs(reference).to(torch.float64).mean()
    if scale.item() < 1e-12:
        return None # baseline is ~all zero, so there is nothing to be relative to
    avg = (diff.mean() / scale).item()
    mx = (diff.max() / scale).item()
    sd = (diff.std() / scale).item() if diff.numel() > 1 else 0.0
    return avg, mx, sd, scale.item()

def report_error(label, approx, reference):
    # print one comparison line: relative first, absolute kept alongside for continuity.
    absavg = torch.abs(approx.detach() - reference.detach()).to(torch.float64).mean().item()
    r = rel_error(approx, reference)
    if r is None:
        print(f"{label}: rel n/a (baseline ~0) | abs avg {absavg:.6g}")
        return
    avg, mx, sd, scale = r
    print(f"{label}: rel avg {avg:7.2%} | rel max {mx:7.2%} | rel std {sd:7.2%}"
          f" | abs avg {absavg:.4g} | |ref| {scale:.4g}")

def s_truncate(*tensors):

    output = []
    for tensor in tensors:
        if tensor.dtype != torch.int64:
            print("WARNING: tensor passed for truncation is not an integer!!")
        int_tensor = tensor # protect against bitwise shift, assume tensor is an int64.
        
        # stochastic ring value
        r = torch.round(torch.rand(tensor.shape, device = tensor.device) * (2**57)).to(torch.int64) # SPEEEEDD
        trunc_r = torch.bitwise_right_shift(r, BITs)

        # print("r val:",r)
        # print("trunc_r",trunc_r)
        # print("int_tensor",int_tensor)
        
        int_tensor = torch.where(int_tensor >= 0, int_tensor + r, int_tensor - r)  # this seems shitty
        # print("int_tensor with r:",int_tensor)
        
        int_tensor = torch.bitwise_right_shift(int_tensor, BITs)
        # print("int_tensor with trunc:",int_tensor)

        int_tensor = torch.where(int_tensor >= 0, int_tensor - trunc_r, int_tensor + trunc_r)  # this seems shitty
        # print("int_tensor with trunc and correction:",int_tensor)

        tensor = int_tensor.to(torch.float64) / 2**BITs
        # print("final tensor:",tensor)

        output.append(tensor) # this is very stupid... why is this stupid?
    
    return tuple(output)

def p_truncate(*tensors):

    # predef consts
    l = 57 # actual size
    k = 64 # repr ring
    m = BITs # truncation bits

    output = []

    for tensor in tensors:

        assert tensor.dtype == torch.int64, "p_truncation only works on integer tensors"

        # the simulated dtype is a 57-bit ring, but to_fixed hands us a signed int64, which is
        # not a ring element: a negative value carries bits 57..63 set. carry below then reads
        # 1 where it should read 0, and the tiebreaker injects 2**(l-m), so y returns ~2**43
        # garbage whenever R happens to land below |tensor| -- rate |tensor|/2**l, which is
        # negligible for small values but reaches 1e-6 by 2**37 and wrecks a whole layer.
        # reduce into Z_{2**l} first. this is free on a real instance rather than an extra
        # operation: shares are always stored reduced, so the value arrives in this form.
        tensor = tensor & ((1 << l) - 1)

        # masks
        r1 = torch.round(torch.rand(tensor.shape, device = tensor.device)*2**(l-m)).to(torch.int64)
        r2 = torch.round(torch.rand(tensor.shape, device = tensor.device)*2**(m)).to(torch.int64)
        b = torch.round(torch.rand(tensor.shape, device = tensor.device)).to(torch.int64) # this will be 1 or 0 uniform

        # masked values
        R = r1 * (2**m) + r2
        S = tensor + (b << l) + R

        # decomposition, rounding bits
        carry = (S >> l) & 1 # if S > l, these are bits above l
        low = S & ((2**l) - 1) # these are bits below l

        v = carry ^ b # tiebreaker

        # unmask
        a = (low >> m)
        b = a - r1
        c = b + ((v)<< (l-m))

        #compare ==> removed for training.

        # c is a Z_{2**(l-m)} element. the protocol itself never needs this step -- ring
        # arithmetic carries the sign implicitly and you decode only at reveal -- but this
        # returns a float for the rest of the simulation to consume, so decode it here.
        c = c & ((1 << (l - m)) - 1)
        c = torch.where(c >= (1 << (l - m - 1)), c - (1 << (l - m)), c)

        y = c/2**m

        output.append(y.to(torch.float64))

    return tuple(output)


# ** Modules and layers

class Conv2dQ(torch.autograd.Function):

    dtype = torch.float64

    @staticmethod
    def forward(ctx, input, kernel, bias, stride, padding, dilation): # in this case these are defaults so wtv.
        # TODO
        # print("=== Conv2dQ forward called ===")
        # print("conv2d info (float format):")
        # print("inputs:",input)
        # print("weights:",kernel)
        # print("bias:",bias)
        
        # float kernel and input
        input, kernel, bias = to_fixed(input, kernel, bias) # make it an integer
        
        ctx.save_for_backward(
            input, kernel
        ) # i hope this isn't by reference.

        ctx.padding = padding
        ctx.stride = stride
        ctx.dilation = dilation

        if isinstance(padding, int):
            padding = (padding, padding)
            
        # Same for stride and dilation
        if isinstance(stride, int):
            stride = (stride, stride)
            
        if isinstance(dilation, int):
            dilation = (dilation, dilation)


        # run conv2d and return the result

        # Debug info
        # print("Input shape:", input.shape)
        # print("Kernel shape:", kernel.shape)
        # print("Stride:", stride)
        # print("Padding:", padding)
        # print("Dilation:", dilation)

        # This will fail if input, kernel are not floats.
        input, kernel = to_float_no_shift(input, kernel) # numbers are just ints in floats lol.
        try:   
            output = F.conv2d(input, kernel, bias = None, stride=stride, padding=padding, dilation=dilation)
        except Exception as e:
            print(f"Conv2d error: {e}")
            input.to(torch.device("cpu"))
            kernel.to(torch.device("cpu"))
            output = F.conv2d(input, kernel, bias=None, stride=stride, padding=padding, dilation=dilation)

        # output will be a float64, to_fixed then truncate
        output, = to_fixed_no_shift(output) # this doesn't modify the value
        output, = p_truncate(output) # TESTTEST
        output, = to_fixed(output)
        output =  output + bias.view(1,-1,1,1)
        output, = to_float(output)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):

        # based on crypten implementation
        dtype = torch.float64

        #quant
        groups = 1
        #grad_output = torch.round(grad_output.to(dtype) * 2**BITs).to(dtype) # needed?

        input, kernel = ctx.saved_tensors # already fp
        grad_output, = to_fixed(grad_output) # bc grad_output will be a float.
        input, grad_output, kernel = to_float_no_shift(input, grad_output, kernel)

        batch_size = input.size(0)

        padding = ctx.padding
        stride = ctx.stride
        dilation = ctx.dilation

        if isinstance(padding, int):
            padding = (padding, padding)
            
        if isinstance(stride, int):
            stride = (stride, stride)
            
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        # to compute : 
        #   grad w.r.t. input
        #   grad w.r.t. kernel

        out_channels, in_channels, kernel_size_y, kernel_size_x = kernel.size()
        in_channels *= groups

        # same problem
        # kernel is an fp
        grad_input = F.conv_transpose2d( # this also requires floats. convert fp number here for calc
            grad_output,
            kernel,
            stride=stride,
            padding=padding,
            output_padding=0,
            groups=groups,
            dilation=dilation,
        )
        

        # compute gradient with respect to kernel:

        # print("grad output shape in conv:",grad_output.shape)
        # print("repeats:",inchannels//groups)
        
        grad_bias = grad_output.sum((0,2,3)) # grad bias?
        # print(grad_bias.shape)
        grad_output = grad_output.repeat(1, in_channels // groups, 1, 1)
        # this repeat is eating up memory maybe i need to clear memory more effectively as well.
        grad_output = grad_output.view(
            grad_output.size(0) * grad_output.size(1),
            1,
            grad_output.size(2),
            grad_output.size(3),
        )
        input = input.view(
            1, input.size(0) * input.size(1), input.size(2), input.size(3)
        ) # memory efficient reshape?
        # dilation and stride are swapped based on PyTorch's conv2d_weight implementation
        grad_kernel = F.conv2d(
            input = input,
            weight = grad_output,
            stride=dilation,
            padding=padding,
            dilation=stride,
            groups=in_channels * batch_size,
        )

        grad_kernel = grad_kernel.view(
            batch_size,
            grad_kernel.size(1) // batch_size,
            grad_kernel.size(2),
            grad_kernel.size(3),
        )
        grad_kernel = (
            grad_kernel.sum(0)
            .view(
                in_channels // groups,
                out_channels,
                grad_kernel.size(2),
                grad_kernel.size(3),
            )
            .transpose(0, 1)
        )
        grad_kernel = grad_kernel.narrow(2, 0, kernel_size_y)
        grad_kernel = grad_kernel.narrow(3, 0, kernel_size_x)

        # grad_input = quantize(grad_input)
        # grad_kernel = quantize(grad_kernel) not needed since calculated with quantized values.
        # this should be correct.
        grad_input, grad_kernel = to_fixed_no_shift(grad_input, grad_kernel) # this hopefully won't overflow?
        grad_input, grad_kernel = p_truncate(grad_input, grad_kernel) # this should work.
        grad_bias, = to_float(grad_bias) # cool

        return (grad_input, grad_kernel, grad_bias, None, None, None) # stride, padding, dilation are not needed in the backward pass.


class LinearQ(torch.autograd.Function):

    dtype = torch.float64

    @staticmethod
    def forward(ctx, inputs, weights, bias):

        # print("=== LinearQ forward called ===")

        # print((inputs @ weights.T) + bias)

        inputs, weights, bias = to_fixed(inputs,weights,bias) # make fp
        inputs, weights = to_float_no_shift(inputs,weights)

        # print("linear info (fixed-float format):")
        # print("inputs:",inputs)
        # print("weights:",weights)
        # print("bias:",bias)

        ctx.save_for_backward(inputs, weights, bias)

        result = inputs @ weights.T
        result, = to_fixed_no_shift(result) # dude this seems really fucking stupid.
        result, = p_truncate(result) # TESTTEST
        # print("truncated result:",result)
        result, = to_fixed(result)
        result = result + bias

        # quantize reuslt and return
        weights, = to_fixed_no_shift(weights)

        result, weights, bias = to_float(result, weights, bias) # go back

        # print(result)

        return result

    @staticmethod
    def backward(ctx, grad_output): # double check what values need to be quantized here?
        
        # quantize grad_output?
        # print("=== LinearQ backward called ===")

        inputs, weights, bias = ctx.saved_tensors # already expanded values

        grad_output, = to_fixed(grad_output) # expand grad_output
        
        grad_bias, = to_float(grad_output) 
        grad_output, = to_float_no_shift(grad_output)

        grad_weight = inputs.T @ grad_output

        grad_input = grad_output @ weights # double check this is right?
        # these will be floats

        grad_input, grad_weight = to_fixed_no_shift(grad_input, grad_weight)
        grad_input, grad_weight = p_truncate(grad_input, grad_weight)

        # print(grad_input.shape)
        # print(grad_weight.shape)

        return (grad_input, grad_weight.T, grad_bias)


class BatchNorm2dQ(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, running_mean, running_var, weight, bias):

        # print(x.shape)
        # print("=== BatchNorm2dQ forward called ===")
        # print("conv2d info (float format):")
        # print("inputs:",x)
        # print("running_mean:",running_mean)
        # print("running_var:",running_var)
        # print("weight:",weight)
        # print("bias:",bias)

        stats_dimensions = list(range(x.dim()))
        stats_dimensions.pop(1)

        broadcast_shape = [1] * x.dim()
        broadcast_shape[1] = x.shape[1]

        # print("x",x)
        gt_x = x
        gt_mean = gt_x.mean(stats_dimensions)
        gt_var = gt_x.var(stats_dimensions, unbiased=False) # torch normalises with the biased variance
        gt_x_norm = (gt_x - gt_mean.reshape(broadcast_shape)) / torch.sqrt(gt_var.reshape(broadcast_shape))

        x, weight, bias = to_fixed(x, weight, bias)

        # eps = 1 # in our fixed point system, this refers to 2^-6 ~= 0.015625.
        momentum = 0.1 # ok

        # print(x.shape)

        # determine dimensions over which means and variances are computed:

        # shape for broadcasting statistics with input:
        # print(x.dim())

        # compute mean and variance, track batch statistics:
        training = True
        if training:
            # mean requires float?
            x, = to_float_no_shift(x)

            mean = x.mean(stats_dimensions) # this math still works just fine. decimal result will be ok
            mean, = to_fixed_no_shift(mean)
            # torch normalises with the biased variance and feeds only the unbiased one
            # into running_var, so the two have to be tracked separately.
            variance = x.var(stats_dimensions, unbiased=False)# these don't need expansion.
            variance, = to_fixed(*p_truncate(*to_fixed_no_shift(variance)))
            # torch's 1e-5 eps rounds to 0 at BITs=7, so clamp to the smallest positive
            # fixed-point value instead: pseudosep takes log2(variance) and needs it > 0.
            variance = torch.clamp(variance, min=1)

            variance_unbiased = x.var(stats_dimensions, unbiased=True)
            variance_unbiased, = to_fixed(*p_truncate(*to_fixed_no_shift(variance_unbiased)))

            x, = to_fixed_no_shift(x) # to fp for later.

            if running_mean is not None and running_var is not None: 
                momentum = int(round(momentum * 2**BITs))
                alpha = (1 << BITs) - momentum

                rm_fixed, rv_fixed = to_fixed(running_mean, running_var)

                t1v = rv_fixed*alpha
                t2v = variance_unbiased * momentum
                t1m = rm_fixed * alpha
                t2m = mean * momentum

                rvar = p_truncate(t1v)[0] + p_truncate(t2v)[0] # not ok.
                rmean = p_truncate(t1m)[0] + p_truncate(t2m)[0] # these need truncation

                # running_var, running_mean = p_truncate(running_var, running_mean)
                running_var.set_(rvar)
                running_mean.set_(rmean)
                # print("Updated running mean:",running_mean)
                # print("Updated running var:",running_var)

        else:
            if running_mean is None or running_var is None:
                raise ValueError(
                    "Must provide running_mean and running_var when training is False"
                )
            # mean = quantize(running_mean)
            # variance = quantize(running_var)
            mean = running_mean
            variance = running_var

        if training or inv_var is None:
            # variance, = to_fixed(*p_truncate(variance))
            # print(variance)
            inv_var, = to_fixed(inverse_sqrt(variance))
            # print("inv_var:",*to_float(inv_var))
            gt_x, = to_float(x)
            gt_var = gt_x.var(stats_dimensions, unbiased=True)
            gt_inv_var = 1 / torch.sqrt(gt_var.to(torch.float64) + 1e-6)
            print("expected inv_var:",gt_inv_var)
            # inv_var, = to_fixed(*p_truncate(inv_var)) # small number correction
            # compute inverse variance:
        #     if torch.is_tensor(variance):
        #         inv_var = (1<<BITs) / torch.sqrt(variance + eps) # difficult for mpc -- we will need to run this approximation soon.
        #     else:
        #         inv_var = 1<<BITS * (variance + eps).rsqrt() # should work

        # # reshape shape (C) to broadcastable (1, C, 1, +):
        # # print(weight.shape)
        # # print(bias.shape)
        # inv_var, = to_fixed_no_shift(inv_var)
        # inv_var, = p_truncate(inv_var) # truncate from multiplication
        # inv_var, = to_fixed(inv_var) # back to int.

        mean = mean.reshape(broadcast_shape)
        inv_var = inv_var.reshape(broadcast_shape)

        weight = weight.reshape(broadcast_shape)
        bias = bias.reshape(broadcast_shape)

        # print(x.shape)
        # print(mean.shape, inv_var.shape)
        # print("batchnorm values",mean, inv_var)

        # compute z-scores:
        # print( x- mean )
        x_norm = (x - mean) * inv_var
        # print(mean)
        # print(inv_var)
        x_norm, = p_truncate(x_norm)
        print("x_norm",x_norm)
        print("expected x_norm:", gt_x_norm)
        report_error("x_norm", x_norm, gt_x_norm)

        # print("gt_x",gt_x)


        # print(x_norm)
        x_norm, = to_fixed(x_norm)

        # save context and return:
        ctx.save_for_backward(gt_x_norm, x_norm, weight, inv_var)
        ctx.training = training

        # print(weight.shape, bias.shape, x_norm.shape)
        # hack fix bc i messed something up.

        result = (x_norm * weight.reshape(broadcast_shape))
        result, = p_truncate(result)
        result, = to_fixed(result)
        result = result + bias
        result, = to_float(result)

        weight, bias, running_mean, running_var = to_float(weight, bias, running_mean, running_var) # needed?


        # print(result)

        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # retrieve context:
        # print(grad_output)

        # print("=== BatchNorm2dQ backward called ===")

        # TODO read through this.
        
        # these will also be quantized in the forward pass.
        gt_x_norm, x_norm, weight, inv_var = ctx.saved_tensors

        stats_dimensions = list(range(len(grad_output.shape)))
        stats_dimensions.pop(1)

        thing = grad_output.mul(gt_x_norm)
        thing = thing.sum(stats_dimensions)

        grad_output, = to_fixed(grad_output) # quantize grad output

        training = ctx.training

        # determine dimensions over which means and variances are computed:

        # shape for broadcasting statistics with output gradient:
        broadcast_shape = [1] * grad_output.dim()
        broadcast_shape[1] = grad_output.shape[1]

        # compute gradient w.r.t. weight:
        grad_weight = grad_output.mul(x_norm) 
        grad_weight, = p_truncate(grad_weight)
        grad_weight = grad_weight.sum(stats_dimensions)

        # print("calculated grad weight:",grad_weight)
        # print("expected grad weight:",thing)

        # compute gradient w.r.t. bias:
        grad_bias = grad_output.sum(stats_dimensions) # yeah 
        grad_bias, = to_float(grad_bias)

        # compute gradient with respect to the input:
        
        grad_output = grad_output.mul(weight) # is this a correct result
        grad_output, = p_truncate(grad_output)
        grad_output, = to_fixed(grad_output)
        
        grad_input = grad_output.mul(inv_var)
        grad_input, = p_truncate(grad_input)

        if training:
            # compute gradient term that is due to the mean:
            num_element = np.prod([grad_output.size(d) for d in stats_dimensions])
            grad_mean = grad_output.sum(stats_dimensions)
            grad_mean = grad_mean.reshape(broadcast_shape)

            # inv_var/num_element rounds to 0 at BITs=7 once num_element > 128, which
            # silently drops this whole term. apply inv_var while the sum is still large
            # and defer the division to the end, where there is magnitude to spare.
            grad_mean = grad_mean.mul(inv_var) # 2**14
            grad_mean, = to_fixed(*p_truncate(grad_mean)) # back to 2**7
            grad_mean = div_public(grad_mean, -num_element)
            grad_mean, = to_float(grad_mean)

            # compute gradient term that is due to the standard deviation:
            grad_std = x_norm.mul(grad_output) # this should be fp
            grad_std, = to_fixed(*p_truncate(grad_std))
            grad_std = grad_std.sum(stats_dimensions)

            grad_std = grad_std.reshape(broadcast_shape)
            grad_std = x_norm.mul(grad_std)

            grad_std, = to_fixed_no_shift(grad_std)
            grad_std, =to_fixed(* p_truncate(grad_std))

            grad_std = grad_std.mul(inv_var) # 2**14
            grad_std, = to_fixed(*p_truncate(grad_std)) # back to 2**7
            grad_std = div_public(grad_std, -num_element)
            grad_std, = to_float(grad_std) # final float form

            # put all the terms together:
            grad_input = grad_input.add(grad_mean).add(grad_std)
        
        # print(stats_dimensions)

        # return gradients:

        # running_mean and running_var don't require gradients -> return None for those slots
        return (grad_input, None, None, grad_weight, grad_bias)


class CNNModelQ(nn.Module):
    def __init__(self):
        # TODO
        b = 255 # MAGIC NUMBER WHEEEEEE

        super(CNNModelQ, self).__init__()

        # kernels to pass as input. we essentially store our parameters here
        # divide by dim of each mat/kernel
        self.k1 = nn.Parameter(torch.rand(64,3,8,8, dtype=torch.float64) / 128, requires_grad = True) 
        self.kb1 = nn.Parameter(torch.zeros(64, dtype=torch.float64)) 
        
        self.bnw1 = nn.Parameter(torch.rand(64, dtype=torch.float64) / 128, requires_grad = True) 
        self.bnb1 = nn.Parameter(torch.zeros(64, dtype=torch.float64), requires_grad = True) 

        self.k2 = nn.Parameter(torch.rand(256,64,8,8, dtype=torch.float64) / 128, requires_grad = True) 
        self.kb2 = nn.Parameter(torch.zeros(256 ,dtype=torch.float64)) 

        self.bnw2 = nn.Parameter(torch.rand(256, dtype=torch.float64) / 128, requires_grad = True) 
        self.bnb2 = nn.Parameter(torch.zeros(256, dtype=torch.float64), requires_grad = True) 

        self.k3 = nn.Parameter(torch.rand(1024,256,4,4, dtype=torch.float64) / 128, requires_grad = True) 
        self.kb3 = nn.Parameter(torch.zeros(1024, dtype=torch.float64)) 

        self.bnw3 = nn.Parameter(torch.rand(1024, dtype=torch.float64) / 128, requires_grad = True) 
        self.bnb3 = nn.Parameter(torch.zeros(1024, dtype=torch.float64) , requires_grad = True) 

        flatten_dim = 50176 

        # output final is fine. use relu!
        self.w1 = nn.Parameter(torch.rand(2048, 50176, dtype=torch.float64) / 128, requires_grad = True) 
        self.b1 = nn.Parameter(torch.zeros(2048, dtype=torch.float64), requires_grad = True) 
        
        self.w2 = nn.Parameter(torch.rand(2048, 2048, dtype=torch.float64) / 128, requires_grad = True) 
        self.b2 = nn.Parameter(torch.zeros(2048, dtype=torch.float64), requires_grad = True) 

        self.w3 = nn.Parameter(torch.rand(100, 2048, dtype=torch.float64) / 128, requires_grad = True) 
        self.b3 = nn.Parameter(torch.zeros(100, dtype=torch.float64), requires_grad = True) 

        # running mean and var -- these aren't listed as params in the baseline model, so for comparison keep them here!
        self.running_mean1 = nn.Parameter(torch.zeros(64, dtype=torch.float64), requires_grad=False)
        self.running_var1 = nn.Parameter(torch.ones(64, dtype=torch.float64), requires_grad=False)

        self.running_mean2 = nn.Parameter(torch.zeros(256, dtype=torch.float64))
        self.running_var2 = nn.Parameter(torch.ones(256, dtype=torch.float64))

        self.running_mean3 = nn.Parameter(torch.zeros(1024, dtype=torch.float64))
        self.running_var3 = nn.Parameter(torch.ones(1024, dtype=torch.float64))


        # layers as functions

        self.conv1 = Conv2dQ.apply
        self.bn1 = BatchNorm2dQ.apply
        self.conv2 = Conv2dQ.apply
        self.bn2 = BatchNorm2dQ.apply
        self.conv3 = Conv2dQ.apply
        self.bn3 = BatchNorm2dQ.apply
        
        self.flatten = nn.Flatten() # this has no params

        self.fc1 = LinearQ.apply
        self.fc2 = LinearQ.apply
        self.fc3 = LinearQ.apply # this needs a fix.
        # self.fc1 = nn.Linear(flatten_dim, 2048, dtype = torch.float64)  # TESTTEST
        # self.fc2 = nn.Linear(2048, 2048, dtype = torch.float64)
        # self.fc3 = nn.Linear(2048, 100, dtype = torch.float64) # TODO the outputs,



        # activations
        
        self.rl1 = nn.ReLU()
        self.rl2 = nn.ReLU()
        self.rl3 = nn.ReLU()

        self.rl4 = nn.ReLU()
        self.rl5 = nn.ReLU() # given that ReLU is sign based, shouldn't be a problem.

    def forward(self, x):
        # TODO
        # stride, padding, dilation
        x = self.conv1(x, self.k1, self.kb1, 2, 4, 1)
        # print("1:",x)
        x = self.bn1(x, self.running_mean1, self.running_var1, self.bnw1, self.bnb1)
        x = self.rl1(x)
        # print("2:",x)
        x = self.conv2(x, self.k2, self.kb2, 1, 0, 1)
        # print("3:",x)
        x = self.bn2(x, self.running_mean2, self.running_var2, self.bnw2, self.bnb2)
        x = self.rl2(x)
        # print("4:",x)
        x = self.conv3(x, self.k3, self.kb3, 1, 0, 1)
        # print("5:",x)
        x = self.bn3(x, self.running_mean3, self.running_var3, self.bnw3, self.bnb3)
        x = self.rl3(x)
        # print("6:",x)
        x = self.flatten(x)
        # print(x.shape)

        x = self.fc1(x, self.w1, self.b1)
        x = self.rl4(x)
        # print("7:",x)
        x = self.fc2(x, self.w2, self.b2)
        x = self.rl5(x)
        # print("8:",x)
        x = self.fc3(x, self.w3, self.b3)
        # x = self.rl6(x)
        # print("9:",x)



        return x


class CNNModelBaseline(nn.Module):
    def __init__(self):
        super(CNNModelBaseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (8,8), stride = 2, padding =4) # todo shapes for CIFAR
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 256, (8,8), stride = 1, padding = 0)
        self.bn2 = nn.BatchNorm2d(256, track_running_stats=True)
        self.conv3 = nn.Conv2d(256, 1024, (4,4), stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(1024, track_running_stats=True)
        
        self.flatten = nn.Flatten()

        flatten_dim = 50176# change as needed

        self.fc1 = nn.Linear(flatten_dim, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 100) # TODO the outputs, inputs of these NEED TO BE QUANTIZED (so do their weights, grads, etc)


        # activations
        
        self.rl1 = nn.ReLU()
        self.rl2 = nn.ReLU()
        self.rl3 = nn.ReLU()

        self.rl4 = nn.ReLU()
        self.rl5 = nn.ReLU()
        # self.rl6 = nn.ReLU()

        # for param in self.parameters():
        #     print(param.dtype)
        #     print(param.element_size()*param.nelement()

    def forward(self,x):
        x = self.conv1(x)
        # print("1:", x)
        x = self.bn1(x)
        x = self.rl1(x)
        # print("2:", x)

        x = self.conv2(x)
        # print("3:", x)
        x = self.bn2(x)
        x = self.rl2(x)
        # print("4:", x)

        x = self.conv3(x)
        # print("5:", x)
        x = self.bn3(x)
        x = self.rl3(x)
        # print("6:", x)

        x = self.flatten(x)
        # print(x.shape)

        x = self.fc1(x)
        x = self.rl4(x)
        # print("7:", x)

        x = self.fc2(x)
        x = self.rl5(x)
        # print("8:", x)

        x = self.fc3(x)
        # print("9:", x)
        return x


class one_hot:
    def __init__(self, num_classes):
        self.classes = num_classes
    
    def __call__(self, label):
        lvec = torch.zeros(self.classes)
        lvec[label] = 1
        return lvec


def load_data(batch_size = 250):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),  # mean
                             (0.2675, 0.2565, 0.2761))  # std
    ])

    target_transform = transforms.Compose([
        one_hot(100)
    ])

    # Load training dataset
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform, target_transform=target_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load test dataset
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform , target_transform=target_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

# gradient clipping

# TODO: Precision comparisons
def train_model(model, num_epochs, lr, max_batches = None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    bce = torch.nn.BCEWithLogitsLoss()
    # here we will use SGD
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9) # this ends up quantized in our backward pass.

    train_loader, test_loader = load_data(128)

    batch_times = []
    losses = [] # per-batch, returned so callers can compare trajectories and spot divergence

    # the model's own parameters set the working dtype: CNNModelQ is float64 while the loader
    # yields float32, and BCEWithLogitsLoss rejects a target whose dtype differs.
    model_dtype = next(model.parameters()).dtype

    # timer setup

    if(torch.cuda.is_available()):
        print("Running on GPU")
        print("GPU Type:",torch.cuda.get_device_name(0))

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)


    for epoch in range(num_epochs):
        # keep output to ttyl to a minimum since this tends to be time consuming

        batch_it = tqdm(enumerate(train_loader), unit = "batches", desc = "batch ", total = len(train_loader))

        for batch_idx, (inputs, labels) in batch_it :
            if max_batches is not None and batch_idx >= max_batches:
                break
            # start.record()
            inputs = inputs.to(device).to(model_dtype)
            labels = labels.to(device).to(model_dtype)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape)
            loss = bce(outputs, labels)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            batch_it.set_postfix({'loss': loss.item()})

            # end.record()

            torch.cuda.synchronize()
            # batch_times.append( start.elapsed_time(end) )

        # no point validating since we don't realistically care.

    return losses

def main():

    parser = argparse.ArgumentParser(description='Change evaluation parameters')

    parser.add_argument('--forward', action='store_true', default=False, help='whether to do a forward pass only (1) or train (0)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    args = parser.parse_args()

    model_base = CNNModelBaseline()
    model_Q = CNNModelQ()


    if args.forward:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)
        model_base.to(device)
        model_Q.to(device)
        sync_models(model_base, model_Q) # otherwise this compares two unrelated random inits
        print("Doing forward pass only...")
        base_in = torch.rand(1,3,32,32, dtype=torch.float32, device = device) # batch size 1
        print("test input:",base_in)
        print(
            "Baseline model forward pass:"
        )
        gt = forward_pass_only(model_base, base_in)
        print(
            "Quantized model forward pass:"
        )
        y_hat = forward_pass_only(model_Q, base_in)

        report_error("model forward", y_hat, gt)
        return

    # otherwise train the models

    # start both from identical weights, or the final param error measures the gap between
    # two random inits on top of whatever quantization actually cost.
    sync_models(model_base, model_Q)

    print("training baseline...")
    base_losses = time_function(train_model, model_base, args.epochs, args.lr)

    print("Training quantized...")
    q_losses = time_function(train_model, model_Q, args.epochs, args.lr)

    if base_losses and q_losses:
        print(f"baseline  loss: first {base_losses[0]:.4g} -> last {base_losses[-1]:.4g}")
        print(f"quantized loss: first {q_losses[0]:.4g} -> last {q_losses[-1]:.4g}")

    print("Comparing parameters")
    print(f"Final param RMSE (relative to baseline): {calculateFinalParamError(model_base, model_Q):.2%}")


if __name__ == "__main__":
    main()

