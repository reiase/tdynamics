import time

import torch

from .types import BaseTracer, TensorDef
from .module_utils import module_name
from ..step import next_layer, reset_layer, layer
from ..step import next_step, reset_step, step

def get_mem_info():
    return {
        "allocated": torch.cuda.memory_allocated() / 1024 / 1024,
        "cached": torch.cuda.memory_reserved() / 1024 / 1024,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024 / 1024,
        "max_cached": torch.cuda.max_memory_reserved() / 1024 / 1024,
    }

class MemTracer(BaseTracer):
    def __init__(self, logfile=None, tracepy=False, logtime=False):
        self.need_wait = torch.cuda.is_available()
        self.logtime = logtime
        if logfile is not None:
            self.logfile = logfile
        else:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            self.logfile = open(f"mem_trace_rank_{rank}.log", "w")
        if tracepy:
            import sys
            sys.settrace(self.pytrace)
        super().__init__()
        
    def pytrace(self, frame, event, arg):
        if event == "exception":
            exception, value, traceback = arg
            if isinstance(value, RuntimeError):
                print(f"Exception: {exception}, Value: {value}", file=self.logfile)
        return self.pytrace
    
    def log(self, stage, m):
        info = {
            "step": step(),
            "module": module_name(m),
            "stage": stage,
            "mem": get_mem_info(),
        }
        if self.logtime:
            info["time"] = time.time()
        print(info, file=self.logfile)

    def pre_forward_hook(self, m, i):
        if self.need_wait:
            torch.cuda.synchronize()
        self.log("pre forward", m)
        return super().pre_forward_hook(m, i)
    
    def post_forward_hook(self, m, i, o):
        if self.need_wait:
            torch.cuda.synchronize()
        self.log("post forward", m)
        return super().post_forward_hook(m, i, o)
    
    def pre_backward_hook(self, m, i):
        if self.need_wait:
            torch.cuda.synchronize()
        self.log("pre backward", m)
        return super().pre_backward_hook(m, i)
    
    def post_backward_hook(self, m, i, o):
        if self.need_wait:
            torch.cuda.synchronize()
        self.log("post backward", m)
        return super().post_backward_hook(m, i, o)
    
    def pre_step_hook(self, m, i):
        if self.need_wait:
            torch.cuda.synchronize()
        self.log("pre step", m)
        return super().pre_step_hook(m, i)
    
    def post_step_hook(self, m, i):
        if self.need_wait:
            torch.cuda.synchronize()
        self.log("post step", m)
        next_step()
        return super().post_step_hook(m, i)