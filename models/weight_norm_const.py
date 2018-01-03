"""
Weight Normalization from https://arxiv.org/abs/1602.07868
"""
import torch
from torch.nn.parameter import Parameter


def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


class WeightNormConst(object):
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim
        self.eps = 1e-10
    def rep_mean(self, w):
        v_mean = w.mean(dim = 1).mean(dim = 1).mean(dim=1)
        v_mean = v_mean.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1,w.size(1),w.size(2),w.size(3))
        return v_mean
    def compute_weight(self, module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        b = getattr(module, self.name + '_b')
        c = getattr(module, self.name + '_c')
        b = torch.clamp(b, min = -5.0, max = 5.0)
        w = v - 2.0 * torch.nn.functional.tanh(b) * self.rep_mean(v) + 0.01 * torch.nn.functional.tanh(c) 
        w = w / (_norm( w , self.dim) + self.eps)
       
        return w * g #w * (g / (_norm( w , self.dim) + self.eps))
    @staticmethod
    def apply(module, name, dim):
        fn = WeightNormConst(name, dim)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]
        eps = 1e-10
        # add g, b and v as new parameters and express w as g/||v|| * v
        w_mean = weight.mean(dim = 1).mean(dim = 1).mean(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        module.register_parameter(name + '_b', Parameter(torch.zeros(w_mean.size())))
        module.register_parameter(name + '_c', Parameter(torch.zeros(w_mean.size())))
        module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
        module.register_parameter(name + '_v', Parameter(weight.data / _norm(weight, dim).data))
        #module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_b']
        del module._parameters[self.name + '_v']
        del module._parameters[self.name + '_c']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def weight_norm_const(module, name='weight', dim=0):
    """Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by `name` (e.g. "weight") with two parameters: one specifying the magnitude
    (e.g. "weight_g") and one specifying the direction (e.g. "weight_v").
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with `dim=0`, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    `dim=None`.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm_const(nn.Linear(20, 40), name='weight')
        Linear (20 -> 40)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNormConst.apply(module, name, dim)
    return module


def remove_weight_norm_const(module, name='weight'):
    """Removes the weight normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm_const(nn.Linear(20, 40))
        >>> remove_weight_norm_const(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNormConst) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm_const of '{}' not found in {}"
                     .format(name, module))
