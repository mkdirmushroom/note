# 书生·浦语大模型 OpenCompass 笔记

## 安装

### 环境安装

```bash

conda create --name opencompass --clone=/root/share/conda_envs/internlm-base
source activate opencompass
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

### 数据准备

```bash
# 解压评测数据集到 data/ 处
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip

# 将会在opencompass下看到data文件夹
```

### 查看支持的数据集和模型 

```bash
# 列出所有跟 internlm 及 ceval 相关的配置
python tools/list_configs.py internlm ceval
```

### 启动评测

```bash
python run.py --datasets ceval_gen --hf-path /share/temp/model_repos/internlm-chat-7b/ --tokenizer-path /share/temp/model_repos/internlm-chat-7b/ --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 2048 --max-out-len 16 --batch-size 4 --num-gpus 1 --debug
```

命令解析
```bash
--datasets ceval_gen \
--hf-path /share/temp/model_repos/internlm-chat-7b/ \  # HuggingFace 模型路径
--tokenizer-path /share/temp/model_repos/internlm-chat-7b/ \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
--model-kwargs device_map='auto' trust_remote_code=True \  # 构建模型的参数
--max-seq-len 2048 \  # 模型可以接受的最大序列长度
--max-out-len 16 \  # 生成的最大 token 数
--batch-size 4  \  # 批量大小
--num-gpus 1  # 运行模型所需的 GPU 数量
--debug
```

如果一切正常，您应该看到屏幕上显示 “Starting inference process”：

```bash
[2024-01-12 18:23:55,076] [opencompass.openicl.icl_inferencer.icl_gen_inferencer] [INFO] Starting inference process...
```

除了通过命令行配置实验外，OpenCompass 还允许用户在配置文件中编写实验的完整配置，并通过 `run.py` 直接运行它。配置文件是以 Python 格式组织的，并且必须包括 `datasets` 和 `models` 字段。

```python
from mmengine.config import read_base

with read_base():
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .datasets.winograd.winograd_ppl import winograd_datasets
    from .models.opt.hf_opt_125m import opt125m
    from .models.opt.hf_opt_350m import opt350m

datasets = [*siqa_datasets, *winograd_datasets]
models = [opt125m, opt350m]
```

运行任务时，我们只需将配置文件的路径传递给 `run.py`：

```bash
python run.py configs/eval_demo.py
```


OpenCompass 提供了一系列预定义的模型配置，位于 `configs/models` 下。

```python
# 使用 `HuggingFaceCausalLM` 评估由 HuggingFace 的 `AutoModelForCausalLM` 支持的模型
from opencompass.models import HuggingFaceCausalLM

# OPT-350M
opt350m = dict(
       type=HuggingFaceCausalLM,
       # `HuggingFaceCausalLM` 的初始化参数
       path='facebook/opt-350m',
       tokenizer_path='facebook/opt-350m',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto'),
       # 下面是所有模型的共同参数，不特定于 HuggingFaceCausalLM
       abbr='opt350m',               # 结果显示的模型缩写
       max_seq_len=2048,             # 整个序列的最大长度
       max_out_len=100,              # 生成的最大 token 数
       batch_size=64,                # 批量大小
       run_cfg=dict(num_gpus=1),     # 该模型所需的 GPU 数量
    )
```

使用配置时，我们可以通过命令行参数 `--models` 指定相关文件，或使用继承机制将模型配置导入到配置文件中的 `models` 列表中。


与模型类似，数据集的配置文件也提供在 `configs/datasets` 下。用户可以在命令行中使用 `--datasets`，或通过继承在配置文件中导入相关配置

下面是来自 `configs/eval_demo.py` 的与数据集相关的配置片段：

```python
from mmengine.config import read_base  # 使用 mmengine.read_base() 读取基本配置

with read_base():
    # 直接从预设的数据集配置中读取所需的数据集配置
    from .datasets.winograd.winograd_ppl import winograd_datasets  # 读取 Winograd 配置，基于 PPL（困惑度）进行评估
    from .datasets.siqa.siqa_gen import siqa_datasets  # 读取 SIQA 配置，基于生成进行评估

datasets = [*siqa_datasets, *winograd_datasets]       # 最终的配置需要包含所需的评估数据集列表 'datasets'
```

数据集配置通常有两种类型：'ppl' 和 'gen'，分别指示使用的评估方法。其中 `ppl` 表示辨别性评估，`gen` 表示生成性评估。

```bash
python run.py --models hf_llama_7b --datasets base_medium
```
