import torch

from .types import BaseTracer
from .module_utils import module_analysis, module_get_fullname

HOOK_CACHE = {}
EVENT_COUNT = 0
TOTAL_COUNT = 0

from .mem_tracer import MemTracer

def install_hooks(
    m: torch.nn.Module = None,
    opt: torch.optim.Optimizer = None,
    tracer: BaseTracer = None,
):
    if tracer is None:
        return

    global HOOK_CACHE
    if m is not None:
        if id(m) in HOOK_CACHE:
            return
        module_analysis(m)
        h1 = m.register_forward_pre_hook(tracer.pre_forward_hook)
        h2 = m.register_forward_hook(tracer.post_forward_hook)
        module_name = module_get_fullname(m)
        if not module_name.endswith("FusedScaleMaskSoftmax"):
            h3 = m.register_full_backward_pre_hook(tracer.pre_backward_hook)
            h4 = m.register_full_backward_hook(tracer.post_backward_hook)
            HOOK_CACHE[id(m)] = (h1, h2, h3, h4)
        else:
            HOOK_CACHE[id(m)] = (h1, h2)
        for s in m.children():
            install_hooks(s, tracer=tracer)

    if opt is not None:
        h1 = opt.register_step_pre_hook(tracer.pre_step_hook)
        h2 = opt.register_step_post_hook(tracer.post_step_hook)
        HOOK_CACHE[opt] = (h1, h2)
