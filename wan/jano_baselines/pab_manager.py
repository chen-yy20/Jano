import torch
from utils.envs import GlobalEnv

class PabManager:
    
    def __init__(self, num_inference_steps: int):
        self.warmup = 5
        self.cooldown = 5
        self.num_inference_steps = num_inference_steps
        self.self_range = GlobalEnv.get_envs("self_range")
        self.cross_range = GlobalEnv.get_envs("cross_range")
        
        self.self_calc = True
        self.cross_calc = True
        
        self.self_attn_cache = {}
        self.cross_attn_cache = {}
        
    def check_calc(self, step):
        if step < self.warmup or step > self.num_inference_steps - self.cooldown:
            self.self_calc = True
        else:
            self.self_calc =  ((step - self.warmup) % self.self_range == 0)
    
        if step < self.warmup or step > self.num_inference_steps - self.cooldown:
            self.cross_calc = True
        else:
            self.cross_calc =  ((step - self.warmup) % self.cross_range == 0)
        print(f"step {step} | PAB: self: {self.self_calc}, cross {self.cross_calc}", flush=True)
        

def init_pab_manger(num_steps) -> PabManager:
    pab_manager = PabManager(num_steps)
    GlobalEnv.set_envs("pab_manager", pab_manager)
    return pab_manager
    
def get_pab_manager() -> PabManager:
    return GlobalEnv.get_envs("pab_manager")