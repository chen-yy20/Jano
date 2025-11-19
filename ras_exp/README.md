本目录中代码用于比较RAS和JANO。

`ras_sd3.py`是原始版本的ras启动代码。

`jano_sd3.py`中集成了origin，ras和jano的运行控制和时间、质量测试代码。

修改`model_path`后，通过设置：
```
ENABLE_JANO = 0或1
ENABLE_RAS = 0或1
```

来进行实验。
