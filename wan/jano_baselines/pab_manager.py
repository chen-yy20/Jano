import torch
from utils.envs import GlobalEnv

class PabManager:
    
    def __init__(self, num_inference_steps, self_range, cross_range, warmup, layer_interval=1):
        self.warmup = warmup
        self.cooldown = 5
        self.layer_interval = layer_interval
        self.num_inference_steps = num_inference_steps
        self.self_range = self_range
        self.cross_range = cross_range
        
        self.self_calc = True
        self.cross_calc = True
        
        self.self_attn_cache = {}
        self.cross_attn_cache = {}
        
        self.print_once = False
        
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
        # 控制print
        # if step == self.warmup + 2:
        #     self.print_once = True
            
            
        print(f"step {step} | PAB: self: {self.self_calc}, cross {self.cross_calc}, store {self.should_store}", flush=True)
    
    def is_target_layer(self, layer_id):
        if layer_id % self.layer_interval == 0:
            # if not self.print_once:
            #     print(f"pab {layer_id=}, should store!", flush=True)
            return True
        return False

def init_pab_manger(num_steps, self_range, cross_range, warmup, layer_interval) -> PabManager:
    pab_manager = PabManager(num_steps, self_range, cross_range, warmup, layer_interval)
    GlobalEnv.set_envs("pab_manager", pab_manager)
    return pab_manager
    
def get_pab_manager() -> PabManager:
    return GlobalEnv.get_envs("pab_manager")