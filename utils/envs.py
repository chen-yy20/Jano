class GlobalEnv:
    _instance = None
    _envs = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalEnv, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_envs(cls, name):
        """获取全局变量"""
        var =  cls._envs.get(name)
        if var is not None:
            return var
        else:
            raise KeyError(f"Did not found '{name}' in GlobalEnv.")
    
    @classmethod
    def set_envs(cls, name, value):
        """设置全局变量"""
        cls._envs[name] = value
        
    @classmethod
    def clear_envs(cls):
        """清空所有全局变量"""
        cls._envs.clear()
        
    @classmethod
    def list_envs(cls):
        """列出所有全局变量名"""
        return list(cls._envs.keys())
    