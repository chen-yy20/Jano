from utils.envs import GlobalEnv

class PabManager:
    
    def __init__(self, num_inference_steps: int, self_range: int = 2, warmup: int = 5):
        self.warmup = warmup
        self.cooldown = 5
        self.num_inference_steps = num_inference_steps
        self.self_range = self_range        
        self.self_calc = True
        self.self_store = False
        self.self_attn_cache = {}     
            
    def check_calc(self, step):
        if step < self.warmup or step > self.num_inference_steps - self.cooldown:
            self.self_calc = True
            self.self_store = False
        else:
            self.self_calc =  ((step - self.warmup) % self.self_range == 0) # 确保第一次是calc
            self.self_store = True and self.self_calc
            
        print(f"step {step+1} | PAB-Attn: Calc:{self.self_calc}, Store:{self.self_store}", flush=True)
    

def init_pab_manger(num_steps, self_range, warmup) -> PabManager:
    pab_manager = PabManager(num_steps, self_range, warmup)
    GlobalEnv.set_envs("pab_manager", pab_manager)
    return pab_manager
    
def get_pab_manager() -> PabManager:
    return GlobalEnv.get_envs("pab_manager")