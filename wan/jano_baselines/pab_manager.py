import torch
from utils.envs import GlobalEnv

class PabManager:
    
    def __init__(self, num_inference_steps, self_range, cross_range, warmup):
        self.warmup = warmup
        self.cooldown = 5
        self.num_inference_steps = num_inference_steps
        self.self_range = self_range
        self.cross_range = cross_range
        
        self.self_calc = True
        self.cross_calc = True
        
        self.self_attn_cache = {}
        self.cross_attn_cache = {}
        
    def check_calc(self, step):
        if step < self.warmup or step > self.num_inference_steps - self.cooldown:
            self.self_calc = True
            self.should_store = False
        else:
            self.self_calc =  ((step - self.warmup) % self.self_range == 0)
            self.should_store = True
    
        if step < self.warmup or step > self.num_inference_steps - self.cooldown:
            self.cross_calc = True
            self.should_store = False
        else:
            self.cross_calc =  ((step - self.warmup) % self.cross_range == 0)
            self.should_store = True
            
        print(f"step {step} | PAB: self: {self.self_calc}, cross {self.cross_calc}, store {self.should_store}", flush=True)
        

def init_pab_manger(num_steps, self_range, cross_range, warmup) -> PabManager:
    pab_manager = PabManager(num_steps, self_range, cross_range, warmup)
    GlobalEnv.set_envs("pab_manager", pab_manager)
    return pab_manager
    
def get_pab_manager() -> PabManager:
    return GlobalEnv.get_envs("pab_manager")