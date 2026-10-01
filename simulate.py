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

# Dedicated fixed-point scale for the batch-norm inverse std-dev, inv_var = 1/sqrt(var) ("Method A").
# inv_var is a small number that shares nothing else's dynamic range: on the global BITs grid it
# rounds to 0 once var > 2**(2*BITs) (the forward "precision floor", see BITS_REPORT.md sec 8).
# Holding it at 2**B_IV with B_IV > BITs pushes that floor out to var > 2**(2*B_IV), decoupling it
# from the global BITs. The scale is public (a bit-shift), so it is MPC-benign, but it is NOT free:
# the inv_var transient (2**(BITs+B_IV), and 2**(BITs+B_IV+log2 S) in the loss-scaled BN backward)
# grows the ring ~1 bit per extra bit and can become the ring bottleneck.
#
# EMPIRICAL RESULT (full-epoch CIFAR-100 sweeps, _sweep_AplusS.py / _sweep_extra.py): on THIS net
# the forward floor is never binding -- variance stays O(1-100), far below even BITs=8's floor of
# 2**16 -- so B_IV > BITs buys NO accuracy at BITs 8 or 10 and only costs ring. The floor-lowering
# to BITs=10 comes from loss scaling alone. B_IV is therefore left == BITs (Method A OFF, identical
# to the original path). Raise it (e.g. BITs+8) only for a deeper / higher-variance net where var
# actually approaches 2**(2*BITs); keep 1.5*BITs + B_IV < ~56 so g*m stays in the 57-bit ring.
B_IV = BITs

# Forward activation scaling for the softmax ("forward loss scale"). The softmax probabilities are
# the one tensor whose natural values legitimately fall below the 2**-BITs grid floor: once the model
# sharpens, off-target probs p_j collapse toward 0, and any p_j < 2**-BITs rounds to exactly 0 in the
# forward. Those zeros feed the CE seed (p - onehot)/N, so the wrong-class gradient signal is DESTROYED
# in the forward -- backward loss scaling S cannot rescue it (S lifts in backward; the value is already
# 0 before backward runs). This is the genuine forward floor that caps BITs=8 (BITS_REPORT sec 8; the
# gap accelerates late in training as more probs drop under the floor -- the underflow fingerprint).
#
# The fix mirrors backward loss scaling but on the forward path: carry the softmax internals (exp, and
# the final probs) at an ELEVATED scale 2**(BITs+A), so p_j down to 2**-(BITs+A) survive forward. A is
# a PUBLIC power-of-two exponent, so every scale/unscale is an exact shift (MPC-benign, no extra
# truncation round). The elevation is stripped in backward by dividing by N<<A instead of N, so the
# returned gradient is unchanged in scale -- only the small components that used to underflow now
# survive. Setting A = log2(S) - log2(N) makes the forward floor coincide with the backward floor, so
# the two move down together as S grows; A=0 exactly reproduces the un-scaled forward path.
SOFTMAX_EXTRA_BITS = 0

# Ring bit-width l of the simulated fixed-point dtype (values live in Z_{2**RING_BITS}). This is the
# TOTAL width -- integer + fractional + sign -- of the backing integer type a real deployment would use.
# It matters in exactly one place, ring_truncate: the stochastic truncation masks span the full ring, so
# its correctness-failure rate is ~ |x|/2**RING_BITS (a wrong carry when the mask lands below |x|). All
# other ops are exact int64 and, as long as the payload peak P < 2**(RING_BITS-1), wrap-free -- so lowering
# RING_BITS faithfully models a narrower dtype provided P stays under that bound (monitor it). The minimum
# viable RING_BITS = smallest fixed-point representation this training tolerates; sweep it down to find it.
RING_BITS = 57

# --- MPC op-count instrumentation ---------------------------------------------------------
# Wall-clock of this simulator is NOT a cost proxy for a real secure-MPC deployment (it runs
# locally with no communication). What DOES track MPC cost is the count of operations whose
# secure realisation is dominated by interactive communication:
#   * probabilistic truncations  -- one per fixed-point multiply group; the core online-round cost
#   * inverse-sqrt approximations -- nonlinear, needs bit-decomposition (BatchNorm)
#   * ReLU comparisons (DReLU)    -- one sign bit per element (computed in fwd, reused in bwd)
# 'calls' counts SIMD invocations (a tensor truncated in one shot ~ one round's worth of work,
# though true round count depends on the data-dependency graph); 'elems' counts total elements
# ~ bandwidth. Counting is always-on and negligibly cheap; snapshot around fwd/bwd to split.
OP_COUNTS = {
    'p_truncate_calls': 0, 'p_truncate_elems': 0,
    'inv_sqrt_calls':   0, 'inv_sqrt_elems':   0,
    'relu_calls':       0, 'relu_elems':       0,
}

def reset_op_counts():
    for key in OP_COUNTS:
        OP_COUNTS[key] = 0

def snapshot_op_counts():
    return dict(OP_COUNTS)

def attach_relu_counter(model):
    # ReLU is an nn.Module here, so count its comparisons with a forward hook. The sign bit is
    # computed once in the forward pass and reused in backward (no new comparison), so counting
    # forward invocations is the correct MPC round cost. Returns the hook handles.
    handles = []
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            def hook(mod, inp, out):
                OP_COUNTS['relu_calls'] += 1
                OP_COUNTS['relu_elems'] += out.numel()
            handles.append(module.register_forward_hook(hook))
    return handles

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
    OP_COUNTS['inv_sqrt_calls'] += 1
    OP_COUNTS['inv_sqrt_elems'] += x.numel()
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
    # hold the scale factor (and hence inv_var) at 2**B_IV, not 2**BITs. m_float carries the whole
    # 1/sqrt(var) magnitude, so quantizing it at 2**BITs is exactly what underflowed to 0 once
    # var > 2**(2*BITs) -- the forward floor. at 2**B_IV the floor moves out to var > 2**(2*B_IV).
    m = torch.round(m_float * (2**B_IV)).to(torch.int64)  # scale 2**B_IV
    g_mult = (g * m)                                      # g@2**BITs * m@2**B_IV -> 2**(BITs+B_IV)
    # g ~= 1/sqrt(u)

    result = ring_truncate(g_mult, BITs)                 # drop BITs -> inv_var int @ 2**B_IV

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
def paired_tensors(model_base, model_Q, include_running=True):
    # yield (name, baseline_tensor, quantized_tensor) for each mapped pair. The baseline and the
    # quantized model are built from the SAME block code (see _ResNet), so their parameter and
    # buffer NAMES are identical -- we pair by name. Shapes are asserted so any structural drift
    # fails loudly here rather than silently comparing unrelated tensors. Running stats live as
    # buffers on both; num_batches_tracked (baseline-only) is not a mapped quantity and is skipped.
    params_b = dict(model_base.named_parameters())
    params_q = dict(model_Q.named_parameters())
    buffers_b = dict(model_base.named_buffers())
    buffers_q = dict(model_Q.named_buffers())

    for name, base in params_b.items():
        quant = params_q.get(name)
        assert quant is not None, f"quantized model has no parameter named {name}"
        assert base.shape == quant.shape, \
            f"shape mismatch: {name} {tuple(base.shape)} vs {tuple(quant.shape)}"
        yield name, base, quant

    if include_running:
        for name, base in buffers_b.items():
            if not (name.endswith("running_mean") or name.endswith("running_var")):
                continue
            quant = params_q.get(name, buffers_q.get(name))
            if quant is None:
                continue
            assert base.shape == quant.shape, \
                f"shape mismatch: {name} {tuple(base.shape)} vs {tuple(quant.shape)}"
            yield name, base, quant

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

def ring_truncate(tensor, m):
    # protocol core: stochastic truncation of the low m bits of a 57-bit ring element. returns
    # an int64 whose scale is reduced by 2**m (i.e. round(tensor / 2**m) as a ring element). m is
    # PUBLIC, so any value is a free bit-shift on a real instance. p_truncate wraps this with the
    # /2**m rescale that turns a product of two 2**m-scaled operands back into a real value; call
    # ring_truncate directly when the two operands were scaled ASYMMETRICALLY -- e.g. an
    # activation at 2**BITs times inv_var at 2**B_IV, where the fractional bits to drop is B_IV,
    # not BITs -- and you want the result as an int at 2**BITs rather than a real.

    # predef consts
    l = RING_BITS # ring bit-width of the simulated fixed-point dtype (public, swept to find the minimum)

    assert tensor.dtype == torch.int64, "ring_truncate only works on integer tensors"

    OP_COUNTS['p_truncate_calls'] += 1
    OP_COUNTS['p_truncate_elems'] += tensor.numel()

    # the simulated dtype is a 57-bit ring, but to_fixed hands us a signed int64, which is
    # not a ring element: a negative value carries bits 57..63 set. carry below then reads
    # 1 where it should read 0, and the tiebreaker injects 2**(l-m), so the result returns
    # ~2**43 garbage whenever R happens to land below |tensor| -- rate |tensor|/2**l, which is
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
    # arithmetic carries the sign implicitly and you decode only at reveal -- but the sim
    # consumes a decoded value, so decode it here.
    c = c & ((1 << (l - m)) - 1)
    c = torch.where(c >= (1 << (l - m - 1)), c - (1 << (l - m)), c)

    return c


def p_truncate(*tensors):
    # truncate a product of two 2**BITs-scaled operands (scale 2**(2*BITs)) back to a real value:
    # drop BITs bits on the ring, then rescale the retained integer by 2**BITs.
    m = BITs # truncation bits
    output = []
    for tensor in tensors:
        c = ring_truncate(tensor, m)
        output.append((c.to(torch.float64) / 2**m))
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
        # conv_transpose2d must reproduce the forward's input spatial size. With stride>1 the forward
        # output size floors, so several input sizes map to the same output -- output_padding disambiguates.
        # Recover it per spatial dim from the saved input size; without this a stride-2 conv whose
        # (I + 2p - dilation*(k-1) - 1) is not divisible by stride returns a grad_input 1px too small.
        op_h = input.size(2) - ((grad_output.size(2) - 1) * stride[0] - 2 * padding[0]
                                + dilation[0] * (kernel_size_y - 1) + 1)
        op_w = input.size(3) - ((grad_output.size(3) - 1) * stride[1] - 2 * padding[1]
                                + dilation[1] * (kernel_size_x - 1) + 1)
        grad_input = F.conv_transpose2d( # this also requires floats. convert fp number here for calc
            grad_output,
            kernel,
            stride=stride,
            padding=padding,
            output_padding=(op_h, op_w),
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
            # torch's 1e-5 eps rounds to 0 whenever eps < 2**-BITs (true for all small BITs),
            # so clamp to the smallest positive fixed-point value instead: pseudosep takes
            # log2(variance) and needs it > 0.
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
            inv_var = inverse_sqrt(variance)  # already int @ 2**B_IV (see inverse_sqrt / B_IV)
            # print("inv_var:",*to_float(inv_var))
            gt_x, = to_float(x)
            gt_var = gt_x.var(stats_dimensions, unbiased=True)
            gt_inv_var = 1 / torch.sqrt(gt_var.to(torch.float64) + 1e-6)
            # print("expected inv_var:",gt_inv_var)
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
        x_norm = (x - mean) * inv_var            # (x-mean)@2**BITs * inv_var@2**B_IV -> 2**(BITs+B_IV)
        # print(mean)
        # print(inv_var)
        x_norm = ring_truncate(x_norm, B_IV)     # drop B_IV -> x_norm int @ 2**BITs
        # print("x_norm",x_norm)
        # print("expected x_norm:", gt_x_norm)
        # report_error("x_norm", x_norm, gt_x_norm)

        # print("gt_x",gt_x)


        # print(x_norm)
        # x_norm is already an int @ 2**BITs from ring_truncate above (no to_fixed needed)

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
        
        grad_input = grad_output.mul(inv_var)                    # grad_out@2**BITs * inv_var@2**B_IV -> 2**(BITs+B_IV)
        grad_input, = to_float(ring_truncate(grad_input, B_IV))  # drop B_IV -> int @ 2**BITs -> real

        if training:
            # compute gradient term that is due to the mean:
            num_element = np.prod([grad_output.size(d) for d in stats_dimensions])
            grad_mean = grad_output.sum(stats_dimensions)
            grad_mean = grad_mean.reshape(broadcast_shape)

            # inv_var/num_element rounds to 0 once num_element > 2**BITs, which
            # silently drops this whole term. apply inv_var while the sum is still large
            # and defer the division to the end, where there is magnitude to spare.
            grad_mean = grad_mean.mul(inv_var) # inv_var@2**B_IV -> scale 2**(BITs+B_IV)
            grad_mean = ring_truncate(grad_mean, B_IV) # drop B_IV -> int @ 2**BITs
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

            grad_std = grad_std.mul(inv_var) # inv_var@2**B_IV -> scale 2**(BITs+B_IV)
            grad_std = ring_truncate(grad_std, B_IV) # drop B_IV -> int @ 2**BITs
            grad_std = div_public(grad_std, -num_element)
            grad_std, = to_float(grad_std) # final float form

            # put all the terms together:
            grad_input = grad_input.add(grad_mean).add(grad_std)
        
        # print(stats_dimensions)

        # return gradients:

        # running_mean and running_var don't require gradients -> return None for those slots
        return (grad_input, None, None, grad_weight, grad_bias)


# =============================================================================================
# Fixed-point / MPC cross-entropy loss  (replaces torch.nn.CrossEntropyLoss, BITS_REPORT sec 11.1)
# ---------------------------------------------------------------------------------------------
# torch.nn.CrossEntropyLoss computes softmax + the seed gradient (softmax - onehot)/N in CLEAR
# float64. That seed gradient feeds fc3's backward, i.e. it is the top of the whole backward pass --
# the exact quantity sec 8/9 identify as the binding precision floor -- so leaving it un-quantized
# makes the low-BITs / loss-scaling conclusions optimistic. This is the fixed-point replacement.
#
# It follows the Conv2dQ / LinearQ / BatchNorm2dQ pattern: an explicit autograd.Function so torch
# never differentiates through the (secret, nonlinear) softmax internals -- backward() supplies the
# gradient directly. Softmax needs three primitives a real protocol runs on secret shares:
#   fixed_exp        -- range-reduced fixed-point exp (2^k * 2^f), IMPLEMENTED below.
#   fixed_reciprocal -- 1/sum(exp) via (1/sqrt)^2 reusing inverse_sqrt, IMPLEMENTED below.
#   secure_max       -- per-row max (stability shift); a comparison, so exact in fixed point (no
#                       quantization error) -- left as a clear max, which is faithful; only its DReLU
#                       *cost* (sec 7.2) is unmodelled. NOT an accuracy gap.
# The only clear-float step is the loss VALUE's log (monitoring-only, off the gradient path).
#
# SCALE CONVENTION (matches the rest of the file): a real value v is stored as int64 round(v*2^BITs);
# a product of two 2^BITs operands is at 2^(2*BITs) and is brought back with p_truncate.
# =============================================================================================

def secure_max(x, dim=1):
    # PRIMITIVE (STUB): per-row max of the logits, used only as the softmax numerical-stability
    # shift (softmax is invariant to subtracting a per-row constant). In MPC this is a max-reduction
    # over C elements -- a DReLU/comparison tree (already a counted primitive type, sec 7.2), scale-
    # preserving (no truncation). x is int64 @ 2^BITs; returns int64 @ 2^BITs, shape (N,1).
    # TODO: replace with the secure comparison-tree max.
    return x.max(dim=dim, keepdim=True).values


# 2^f minimax-ish polynomial coefficients on f in [0,1) (Taylor of 2^f = e^(f ln2) to cubic;
# max |err| ~0.6% at f->1, dominated by the fractional-part approximation). Bump the degree here if
# the softmax gradient needs it -- this is the one accuracy knob for fixed_exp.
_EXP2_COEFFS = [1.0, math.log(2.0), (math.log(2.0) ** 2) / 2.0, (math.log(2.0) ** 3) / 6.0]

def fixed_exp(x, extra_bits=0):
    # Fixed-point exp via range reduction (Keller & Sun 2022 / Aly & Smart 2019, BITS_REPORT sec 12):
    #   e^x = 2^(x*log2 e) = 2^k * 2^f,   t = x*log2 e = k + f,  k = floor(t) <= 0,  f in [0,1).
    # The integer part 2^k is an exact (public-amount-per-element) shift; the fractional part 2^f is a
    # polynomial on [0,1). Inputs are <= 0 (post max-shift) so k <= 0 and the shift is always right.
    # x is int64 @ 2^BITs; returns e^x as int64 @ 2^(BITs+extra_bits), in (0,1]. extra_bits>0 elevates
    # the OUTPUT scale so that small e^x (very negative x) survive to 2^-(BITs+extra_bits) instead of
    # underflowing at 2^-BITs -- the forward softmax-scaling path (see SOFTMAX_EXTRA_BITS). The left-
    # shift is applied BEFORE the final right-shift by -k, so the extra bits are retained through the
    # only lossy step; extra_bits=0 is the original 2^BITs path exactly.
    OP_COUNTS['inv_sqrt_calls'] += 1                 # reuse the nonlinear counter (exp ~ inv_sqrt cost)
    OP_COUNTS['inv_sqrt_elems'] += x.numel()

    t = torch.round(x.to(torch.float64) * math.log2(math.e)).to(torch.int64)  # t @ 2^BITs (public mult)
    k = t >> BITs                                    # integer part, floor (arith shift), <= 0
    f = t - (k << BITs)                              # fractional part @ 2^BITs, in [0, 2^BITs)

    # 2^f by Horner on the fixed-point grid: (((c3*f + c2)*f + c1)*f + c0
    coeffs = [torch.round(torch.tensor([c], dtype=torch.float64, device=x.device) * (2 ** BITs)).to(torch.int64)
              for c in _EXP2_COEFFS]
    p = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        p, = to_fixed(*p_truncate(p * f))            # p*f @ 2^(2*BITs) -> @ 2^BITs
        p = p + c
    # p ~ 2^f @ 2^BITs, in [1,2). multiply by 2^k = right shift by -k (k<=0), clamped (>=63 -> 0).
    # elevate by extra_bits FIRST (exact left-shift) so the right-shift by -k keeps extra_bits low bits.
    p = p << extra_bits                              # 2^f @ 2^(BITs+extra_bits)
    shift = torch.clamp(-k, max=63)
    result = torch.bitwise_right_shift(p, shift)     # e^x @ 2^(BITs+extra_bits)
    return result


def fixed_reciprocal(d):
    # Fixed-point reciprocal via 1/d = (1/sqrt(d))^2: reuse the tested inverse_sqrt, then square. d is
    # the softmax row-sum (>= 1). Squaring roughly doubles inverse_sqrt's relative error, so this is a
    # slightly PESSIMISTIC noise model vs a native secure reciprocal (Newton) -- the safe direction for
    # re-checking the BITs floor. Op-count-wise it charges one inv_sqrt + one multiply, not a native
    # reciprocal's round profile (fine: softmax runs once/sample, negligible bandwidth, BITS_REPORT
    # sec 7.2). d int64 @ 2^BITs -> 1/d int64 @ 2^BITs.
    # TODO (optional): swap for a native Newton reciprocal if an accurate round count is needed.
    r = inverse_sqrt(d)                              # 1/sqrt(d) @ 2^B_IV
    return ring_truncate(r * r, 2 * B_IV - BITs)     # (1/sqrt(d))^2 = 1/d @ 2^BITs


class CrossEntropyLossQ(torch.autograd.Function):
    # Fixed-point cross-entropy over raw logits (fc3 output). forward returns the scalar loss (for
    # logging); backward returns d loss / d logits = (softmax - onehot)/N, scaled by grad_output
    # (which carries the public loss-scale S from (S*loss).backward()).

    @staticmethod
    def forward(ctx, logits, targets):
        # logits: (N, C) float @ model dtype.  targets: (N,) int64 class indices.
        N, C = logits.shape
        x, = to_fixed(logits)                          # -> int64 @ 2^BITs

        # 1) numerical-stability shift: subtract per-row max (softmax is shift-invariant).
        x = x - secure_max(x, dim=1)                   # ring subtract, still @ 2^BITs, values <= 0

        # 2) exp, then 3) normalise by the reciprocal of the row-sum. The softmax internals are carried
        # at an ELEVATED scale 2^(BITs+A) (A = SOFTMAX_EXTRA_BITS) so small off-target probs survive the
        # forward instead of underflowing to 0 at the 2^-BITs grid floor (forward loss scaling; see the
        # SOFTMAX_EXTRA_BITS note). A=0 reproduces the original 2^BITs path exactly.
        A = SOFTMAX_EXTRA_BITS
        e = fixed_exp(x, extra_bits=A)                 # (N,C) @ 2^(BITs+A), in (0,1]; small e^x preserved
        denom = e.sum(dim=1, keepdim=True)             # (N,1) @ 2^(BITs+A)  (sum is linear -> free)
        # the row-sum is >= 1 (the max element contributes e^0=1), so it never underflows -- drop it back
        # to 2^BITs so fixed_reciprocal (built on inverse_sqrt @ 2^BITs) sees its expected input scale.
        denom_lo = denom if A == 0 else ring_truncate(denom, A)  # (N,1) @ 2^BITs
        inv_denom = fixed_reciprocal(denom_lo)         # (N,1) @ 2^BITs

        probs = e * inv_denom                          # @ 2^(2*BITs+A)  (2^(BITs+A) * 2^BITs)
        probs = ring_truncate(probs, BITs)             # -> int64 @ 2^(BITs+A), softmax probs (small kept)

        ctx.save_for_backward(probs, targets)
        ctx.N = N
        ctx.A = A

        # loss VALUE is monitoring-only: autograd does NOT differentiate through forward (custom
        # backward supplies the grad), so computing -log(p_target) in the clear here is harmless and
        # does not enter the gradient path. (A real run may skip the loss value entirely -- training
        # only needs the backward seed.) clamp guards log(0) at the 2^-BITs grid floor.
        pf = probs.to(torch.float64) / 2.0 ** (BITs + A)   # decode at the elevated softmax scale
        p_target = pf[torch.arange(N, device=logits.device), targets]
        loss = -torch.log(p_target.clamp_min(2.0 ** -(BITs + A))).mean()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # d loss / d logits = (softmax - onehot(target)) / N, then * grad_output (carries scale S).
        probs, targets = ctx.saved_tensors
        N = ctx.N
        A = ctx.A                                      # softmax forward elevation (probs @ 2^(BITs+A))

        onehot = torch.zeros_like(probs)
        onehot[torch.arange(N, device=probs.device), targets] = (1 << (BITs + A))  # 1.0 @ 2^(BITs+A)
        grad = probs - onehot                          # @ 2^(BITs+A), ring subtract, values in [-1, 1]

        # The seed gradient is S*(softmax - onehot)/N. S (the public loss scale, grad_output), 1/N, and
        # the forward elevation 2^A are all PUBLIC scalars, but the ORDER of application decides whether
        # the small off-target components (~p_j/N) survive quantization. Two floors combine here:
        #   * FORWARD floor -- fixed above by carrying probs at 2^(BITs+A): p_j down to 2^-(BITs+A)
        #     survive the softmax instead of underflowing to 0 before backward ever runs.
        #   * BACKWARD floor -- fixed by applying S FIRST (an exact power-of-two shift that lifts), THEN
        #     rounding the combined /(N*2^A) on the grid at the lifted scale. The single truncation is
        #     round((p-y)*S / N) performed while the value is large, so small components land on the grid.
        # Doing the /N before the *S rounds an already-underflowed intermediate -- that was the original
        # bug. Choosing A = log2(S) - log2(N) lines the two floors up so they descend together as S grows.
        # With A=0 and S=1 this reproduces both underflow floors exactly (sec 8), which is why forward
        # scaling (A>0) AND loss scaling (S>1) are both needed to push BITs below 10.
        grad = grad * grad_output.to(torch.int64)      # * S first (public shift), exact, @ 2^(BITs+A)
        grad = div_public(grad, N << A)                # round((p-y)*S / N) back to 2^BITs; strips the A
        #                                                elevation (N*2^A) AND the /N in one rounding, done
        #                                                at the lifted scale so small components survive.
        grad, = to_float(grad)                         # decode @ 2^BITs; fc3 (LinearQ.backward) requantizes

        # gradients must line up with forward's inputs (logits, targets); targets needs none.
        return grad, None


def cross_entropy_q(logits, targets):
    # convenience wrapper so it drops into the training loop exactly like torch's CE:
    #   loss = cross_entropy_q(model(x), targets)   ->   (S * loss).backward()
    return CrossEntropyLossQ.apply(logits, targets)


# ============================================================================================
# Residual CNN + MLP architecture, shared between the float baseline and the fixed-point model.
# Both are built from the SAME block code via a `q` flag, so their submodule/parameter NAMES are
# identical -- which is what lets paired_tensors/sync_models pair them by name (no hand map).
#
# Faithfulness of the new ops in the quantized path:
#   * residual add (out + identity)  -- a ring add of two on-grid values: exact, free in MPC.
#   * projection shortcut (1x1 conv) -- a Conv2dQ: already faithful.
#   * global average pool            -- sum (on grid) then div_public by the public spatial count:
#                                        one rounding onto the grid, same primitive as div_public.
# ============================================================================================

# Toggle for the classifier head. True -> global average pool before the MLP (small classifier,
# eases the fc weight-grid floor, less overfitting). False -> full flatten (large first FC).
# Read at model construction; set simulate.GLOBAL_AVG_POOL before building the models to change it.
GLOBAL_AVG_POOL = True

# Architecture depth (read at model construction; set before building models to change).
#   STAGE_BLOCKS -- number of residual basic blocks per conv stage (3 stages, widths WIDTHS).
#   MLP_BLOCKS   -- number of residual MLP blocks in the classifier head.
# e.g. [2,2,2] -> ~13 conv (ResNet-ish), [3,3,3] -> ~19, [5,5,5] -> ~31.
STAGE_BLOCKS = [2, 2, 2]
STAGE_WIDTHS = [64, 128, 256]
MLP_BLOCKS = 2


# ---- fixed-point layer wrappers: torch-matching parameter names, compute via the Q autograd fns ----

class QConv2d(nn.Module):
    def __init__(self, cin, cout, k, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cout, cin, k, k, dtype=torch.float64))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))     # init is overwritten by sync_models
        if bias:
            self.bias = nn.Parameter(torch.zeros(cout, dtype=torch.float64))
        else:
            self.register_parameter('bias', None)
        self.stride, self.padding = stride, padding

    def forward(self, x):
        bias = self.bias
        if bias is None:                                          # bias-free conv (pre-BN): pass zeros
            bias = torch.zeros(self.weight.shape[0], dtype=x.dtype, device=x.device)
        return Conv2dQ.apply(x, self.weight, bias, self.stride, self.padding, 1)


class QBatchNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(c, dtype=torch.float64))
        self.register_buffer('running_mean', torch.zeros(c, dtype=torch.float64))
        self.register_buffer('running_var', torch.ones(c, dtype=torch.float64))

    def forward(self, x):
        return BatchNorm2dQ.apply(x, self.running_mean, self.running_var, self.weight, self.bias)


class QLinear(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(fout, fin, dtype=torch.float64))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))     # overwritten by sync_models
        self.bias = nn.Parameter(torch.zeros(fout, dtype=torch.float64))

    def forward(self, x):
        return LinearQ.apply(x, self.weight, self.bias)


def _factory(q):
    # return (Conv, BN, Lin) constructors with a uniform signature, float64+Q or float32+torch.
    if q:
        Conv = lambda ci, co, k, s, p, b: QConv2d(ci, co, k, s, p, b)
        BN   = lambda c: QBatchNorm2d(c)
        Lin  = lambda fi, fo: QLinear(fi, fo)
    else:
        Conv = lambda ci, co, k, s, p, b: nn.Conv2d(ci, co, k, stride=s, padding=p, bias=b)
        BN   = lambda c: nn.BatchNorm2d(c, track_running_stats=True)
        Lin  = lambda fi, fo: nn.Linear(fi, fo)
    return Conv, BN, Lin


class BasicBlock(nn.Module):
    # ResNet basic block: conv-bn-relu-conv-bn (+ identity or 1x1-projection shortcut), then relu.
    def __init__(self, cin, cout, stride, q):
        super().__init__()
        Conv, BN, _ = _factory(q)
        self.conv1 = Conv(cin, cout, 3, stride, 1, False)
        self.bn1   = BN(cout)
        self.conv2 = Conv(cout, cout, 3, 1, 1, False)
        self.bn2   = BN(cout)
        self.relu  = nn.ReLU()
        self.has_down = (stride != 1 or cin != cout)
        if self.has_down:                                        # projection shortcut on shape change
            self.down_conv = Conv(cin, cout, 1, stride, 0, False)
            self.down_bn   = BN(cout)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.has_down:
            identity = self.down_bn(self.down_conv(x))
        return self.relu(out + identity)                        # residual add (ring add: exact/free)


class MLPBlock(nn.Module):
    # residual MLP block on a fixed width: fc-relu-fc (+ identity), then relu.
    def __init__(self, dim, q):
        super().__init__()
        _, _, Lin = _factory(q)
        self.fc1 = Lin(dim, dim)
        self.fc2 = Lin(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return self.relu(out + x)                               # residual add


class _ResNet(nn.Module):
    # Shared backbone. q=True -> fixed-point (CNNModelQ); q=False -> float baseline.
    # Depth is configurable via the module globals STAGE_BLOCKS / STAGE_WIDTHS / MLP_BLOCKS (read here):
    # stem -> 3 conv stages (downsample at stage starts) -> head [GAP or flatten] -> fc->512 ->
    # MLP_BLOCKS residual MLP blocks -> fc->100.

    def __init__(self, q):
        super().__init__()
        self.q = q
        Conv, BN, Lin = _factory(q)
        self.relu = nn.ReLU()
        blocks, widths = STAGE_BLOCKS, STAGE_WIDTHS

        # stem: 32x32 -> 16x16
        self.stem_conv = Conv(3, 64, 3, 2, 1, False)
        self.stem_bn   = BN(64)

        # residual stages (stage>0 downsamples by 2 at its first block): 16 -> 8 -> 4
        self.stages = nn.ModuleList()
        cin = 64
        for si, (nb, cout) in enumerate(zip(blocks, widths)):
            stage = nn.ModuleList()
            for bi in range(nb):
                stride = 2 if (bi == 0 and si > 0) else 1
                stage.append(BasicBlock(cin, cout, stride, q))
                cin = cout
            self.stages.append(stage)

        # head
        self.use_gap = GLOBAL_AVG_POOL
        feat_spatial = 4
        fc_in = cin if self.use_gap else cin * feat_spatial * feat_spatial
        self.fc_in_dim = fc_in
        self.head_fc0 = Lin(fc_in, 512)
        self.mlp = nn.ModuleList([MLPBlock(512, q) for _ in range(MLP_BLOCKS)])
        self.head_fc_out = Lin(512, 100)

    def _gap(self, x):
        # Global average pool = sum / (public spatial count). Use torch's mean so the op stays
        # differentiable (a hand-rolled to_fixed/div_public severs the autograd graph). Faithfulness
        # is preserved because the pooled vector immediately enters head_fc0 (LinearQ), whose forward
        # re-quantizes its input with to_fixed -- snapping any off-grid mean back onto the 2^-BITs grid.
        # The /N gradient (1/N, a public scalar) is exact, matching the fixed-point average's backward.
        return x.mean(dim=(2, 3))

    def forward(self, x):
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        for stage in self.stages:
            for block in stage:
                x = block(x)
        x = self._gap(x) if self.use_gap else x.flatten(1)
        x = self.relu(self.head_fc0(x))
        for blk in self.mlp:
            x = blk(x)
        return self.head_fc_out(x)


class CNNModelQ(_ResNet):
    def __init__(self):
        super().__init__(q=True)


class CNNModelBaseline(_ResNet):
    def __init__(self):
        super().__init__(q=False)


class one_hot:
    def __init__(self, num_classes):
        self.classes = num_classes
    
    def __call__(self, label):
        lvec = torch.zeros(self.classes)
        lvec[label] = 1
        return lvec


def augment_batch(x, pad=4):
    # Standard CIFAR train-time augmentation, applied on-GPU each epoch (the training loop caches
    # normalized images on device, so augmentation must be re-sampled per epoch, not baked into the
    # cache). Per-sample random horizontal flip + per-sample reflect-pad-and-random-crop back to 32x32.
    N, C, H, W = x.shape
    # random horizontal flip (per sample)
    flip = torch.rand(N, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    # random crop: reflect-pad by `pad`, then take a random HxW window per sample
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    Wp = W + 2 * pad
    oy = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
    ar = torch.arange(H, device=x.device)
    rows = oy[:, None] + ar[None, :]                                   # (N, H)
    cols = ox[:, None] + ar[None, :]                                   # (N, W)
    xr = torch.gather(xp, 2, rows[:, None, :, None].expand(N, C, H, Wp))
    xc = torch.gather(xr, 3, cols[:, None, None, :].expand(N, C, H, W))
    return xc


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
    # CrossEntropyLoss over class indices: fc3 emits raw logits (no final activation) and CE
    # applies log_softmax internally, giving the network a real gradient toward the correct
    # class. BCEWithLogitsLoss over 100 one-hot targets minimised to a trivial "predict absent
    # everywhere" solution (loss fell but accuracy stayed at chance -- see BITS_REPORT.md).
    ce = torch.nn.CrossEntropyLoss()
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
            # labels arrive one-hot (N, 100); CrossEntropyLoss wants class indices (N,).
            targets = labels.to(device).argmax(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.shape)
            loss = ce(outputs, targets)

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

