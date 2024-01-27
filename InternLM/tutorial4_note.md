# 书生·浦语大模型 Lmdeploy 笔记

## 1 环境配置

```bash
$ /root/share/install_conda_env_internlm_base.sh lmdeploy
```

然后激活环境。

```bash
$ conda activate lmdeploy
```
安装源码

```bash
# 解决 ModuleNotFoundError: No module named 'packaging' 问题
pip install packaging
# 使用 flash_attn 的预编译包解决安装过慢问题
pip install /root/share/wheels/flash_attn-2.4.2+cu118torch2.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

```bash
pip install 'lmdeploy[all]==v0.1.0'
```

## 2 服务部署

### 2.1 模型转换

#### 2.1.1 在线转换

```bash
# 需要能访问 Huggingface 的网络环境
lmdeploy chat turbomind internlm/internlm-chat-20b-4bit --model-name internlm-chat-20b
lmdeploy chat turbomind Qwen/Qwen-7B-Chat --model-name qwen-7b
```

直接启动本地的 Huggingface 模型，如下所示。

```bash
lmdeploy chat turbomind /share/temp/model_repos/internlm-chat-7b/  --model-name internlm-chat-7b
```

#### 2.1.2 离线转换

使用官方提供的模型文件，就在用户根目录执行，如下所示。

```bash
lmdeploy convert internlm-chat-7b  /root/share/temp/model_repos/internlm-chat-7b/
```

执行完成后将会在当前目录生成一个 `workspace` 的文件夹。这里面包含的就是 TurboMind 和 Triton “模型推理”需要到的文件。

### 2.2  TurboMind 推理+命令行本地对话

执行命令如下：

```bash
# Turbomind + Bash Local Chat
lmdeploy chat turbomind ./workspace
```

### 2.3 TurboMind推理+API服务
启动API
```bash
# ApiServer+Turbomind   api_server => AsyncEngine => TurboMind
lmdeploy serve api_server ./workspace \
	--server_name 0.0.0.0 \
	--server_port 23333 \
	--instance_num 64 \
	--tp 1
```
调用API
```bash
# ChatApiClient+ApiServer（注意是http协议，需要加http）
lmdeploy serve api_client http://localhost:23333
```

### 2.4 网页 Demo 演示


#### 2.4.1 TurboMind 服务作为后端

直接启动作为前端的 Gradio：

```bash
# Gradio+ApiServer。必须先开启 Server，此时 Gradio 为 Client
lmdeploy serve gradio http://0.0.0.0:23333 \
	--server_name 0.0.0.0 \
	--server_port 6006 \
	--restful_api True
```

#### 2.4.2 TurboMind 推理作为后端


```bash
# Gradio+Turbomind(local)
lmdeploy serve gradio ./workspace
```

## 3 模型量化

### 3.1 KV Cache 量化

#### 3.1.1 量化步骤

KV Cache 量化是将已经生成序列的 KV 变成 Int8，使用过程一共包括三步：

第一步：计算 minmax。主要思路是通过计算给定输入样本在每一层不同位置处计算结果的统计情况。

- 对于 Attention 的 K 和 V：取每个 Head 各自维度在所有Token的最大、最小和绝对值最大值。对每一层来说，上面三组值都是 `(num_heads, head_dim)` 的矩阵。这里的统计结果将用于本小节的 KV Cache。
- 对于模型每层的输入：取对应维度的最大、最小、均值、绝对值最大和绝对值均值。每一层每个位置的输入都有对应的统计值，它们大多是 `(hidden_dim, )` 的一维向量，当然在 FFN 层由于结构是先变宽后恢复，因此恢复的位置维度并不相同。这里的统计结果用于下个小节的模型参数量化，主要用在缩放环节（回顾PPT内容）。

第一步执行命令如下：

```bash
# 计算 minmax
lmdeploy lite calibrate \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --calib_dataset "c4" \
  --calib_samples 128 \
  --calib_seqlen 2048 \
  --work_dir ./quant_output
```

在这个命令行中，会选择 128 条输入样本，每条样本长度为 2048，数据集选择 C4，输入模型后就会得到上面的各种统计值。值得说明的是，如果显存不足，可以适当调小 samples 的数量或 sample 的长度。

> 这一步由于默认需要从 Huggingface 下载数据集，国内经常不成功。所以我们导出了需要的数据，大家需要对读取数据集的代码文件做一下替换。共包括两步：
>
> - 第一步：复制 `calib_dataloader.py` 到安装目录替换该文件：`cp /root/share/temp/datasets/c4/calib_dataloader.py  /root/.conda/envs/lmdeploy/lib/python3.10/site-packages/lmdeploy/lite/utils/`
> - 第二步：将用到的数据集（c4）复制到下面的目录：`cp -r /root/share/temp/datasets/c4/ /root/.cache/huggingface/datasets/` 

第二步：通过 minmax 获取量化参数。主要就是利用下面这个公式，获取每一层的 K V 中心值（zp）和缩放值（scale）。

```bash
zp = (min+max) / 2
scale = (max-min) / 255
quant: q = round( (f-zp) / scale)
dequant: f = q * scale + zp
```

有这两个值就可以进行量化和解量化操作了。具体来说，就是对历史的 K 和 V 存储 quant 后的值，使用时在 dequant。

第二步的执行命令如下：

```bash
# 通过 minmax 获取量化参数
lmdeploy lite kv_qparams \
  --work_dir ./quant_output  \
  --turbomind_dir workspace/triton_models/weights/ \
  --kv_sym False \
  --num_tp 1
```

在这个命令中，`num_tp` 的含义前面介绍过，表示 Tensor 的并行数。每一层的中心值和缩放值会存储到 `workspace` 的参数目录中以便后续使用。`kv_sym` 为 `True` 时会使用另一种（对称）量化方法，它用到了第一步存储的绝对值最大值，而不是最大值和最小值。

第三步：修改配置。也就是修改 `weights/config.ini` 文件，这个我们在《2.6.2 模型配置实践》中已经提到过了（KV int8 开关），只需要把 `quant_policy` 改为 4 即可。

这一步需要额外说明的是，如果用的是 TurboMind1.0，还需要修改参数 `use_context_fmha`，将其改为 0。

接下来就可以正常运行前面的各种服务了，只不过咱们现在可是用上了 KV Cache 量化，能更省（运行时）显存了。


### 3.2 W4A16 量化

#### 3.2.1 量化步骤

W4A16中的A是指Activation，保持FP16，只对参数进行 4bit 量化。使用过程也可以看作是三步。

第一步：同 1.3.1。

第二步：量化权重模型。利用第一步得到的统计值对参数进行量化，具体又包括两小步：

- 缩放参数。主要是性能上的考虑（回顾 PPT）。
- 整体量化。

第二步的执行命令如下：

```bash
# 量化权重模型
lmdeploy lite auto_awq \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --w_bits 4 \
  --w_group_size 128 \
  --work_dir ./quant_output 
```

命令中 `w_bits` 表示量化的位数，`w_group_size` 表示量化分组统计的尺寸，`work_dir` 是量化后模型输出的位置。这里需要特别说明的是，因为没有 `torch.int4`，所以实际存储时，8个 4bit 权重会被打包到一个 int32 值中。所以，如果你把这部分量化后的参数加载进来就会发现它们是 int32 类型的。

最后一步：转换成 TurboMind 格式。

```bash
# 转换模型的layout，存放在默认路径 ./workspace 下
lmdeploy convert  internlm-chat-7b ./quant_output \
    --model-format awq \
    --group-size 128
```

这个 `group-size` 就是上一步的那个 `w_group_size`。如果不想和之前的 `workspace` 重复，可以指定输出目录：`--dst_path`，比如：

```bash
lmdeploy convert  internlm-chat-7b ./quant_output \
    --model-format awq \
    --group-size 128 \
    --dst_path ./workspace_quant
```
