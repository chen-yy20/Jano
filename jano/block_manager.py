import torch
import torch.nn.functional as F
from utils.envs import GlobalEnv

class BlockManager:
    def __init__(self, T, H, W, C):
        self.T = T
        self.H = H
        self.W = W
        self.C = C
        self.latent_shape = (C, T, H, W)
        self.block_size = GlobalEnv.get_envs('analyze_block_size')
        if self.T == 1:
            assert self.block_size[0] == 1, f"Unsupported block_size: {self.block_size}"
            
        # Padded size
        bt, bh, bw = self.block_size
        self.padded_T = ((T + bt - 1) // bt) * bt # 最小倍数
        self.padded_H = ((H + bh - 1) // bh) * bh
        self.padded_W = ((W + bw - 1) // bw) * bw
        
        # Block number on each dimension
        self.nt = self.padded_T // bt
        self.nh = self.padded_H // bh
        self.nw = self.padded_W // bw
        self.total_blocks = self.nt * self.nh * self.nw
        
    def _pad_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        对tensor进行padding以适应block_size
        Args:
            tensor: [step, C, T, H, W]
        Returns:
            padded_tensor: [step, C, padded_T, padded_H, padded_W]
        """
        step, C, T, H, W = tensor.shape
        
        # 计算padding大小
        pad_T = self.padded_T - T
        pad_H = self.padded_H - H
        pad_W = self.padded_W - W
        
        # 使用F.pad进行padding，顺序是(左,右,上,下,前,后)
        # 对于5D tensor [step, C, T, H, W]，padding顺序是 [W, H, T]
        # 左侧补0，右侧补pad
        padded_tensor = F.pad(tensor, (0, pad_W, 0, pad_H, 0, pad_T), mode='constant', value=0)
        
        return padded_tensor
    
    def block_3d(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        对profile输入的blocks进行3D数据分块（支持padding）
        Args:
            tensor: [step, C, T, H, W]
        Returns:
            blocked: [step, total_blocks, bt, bh*bw, C]
        """
        # 先进行padding
        padded_tensor = self._pad_tensor(tensor)
        step, C, T, H, W = padded_tensor.shape
        bt, bh, bw = self.block_size
        
        # 重新排列维度便于分块
        tensor = padded_tensor.permute(0, 2, 3, 4, 1)  # [step, T, H, W, C]
        
        # 分块重排
        blocked = tensor.view(step, self.nt, bt, self.nh, bh, self.nw, bw, C)
        blocked = blocked.permute(0, 1, 3, 5, 2, 4, 6, 7)  # [step, nt, nh, nw, bt, bh, bw, C]
        blocked = blocked.reshape(step, self.total_blocks, bt, bh*bw, C)
        
        return blocked
    

BM = None     
    
def init_block_manager(T, H, W, C) -> BlockManager:
    global BM
    BM = BlockManager(T, H, W, C)
    return BM

def get_block_manager() -> BlockManager:
    global BM
    assert BM is not None
    return BM