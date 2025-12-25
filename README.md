# [HBFormer： A Hybrid-Bridge Transformer for Microtumor and Miniature Organ Segmentation](https://doi.org/10.48550/arXiv.2512.03597)

# 🛎 Citation
# HBFormer
Accepted by IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2025)

---

If you find our work helpful for your research, please cite:

```bib
@article{zheng2025hbformer,
  title={HBFormer: A Hybrid-Bridge Transformer for Microtumor and Miniature Organ Segmentation},
  author={Zheng, Fuchen and Chen, Xinyi and Li, Weixuan and Li, Quanjun and Zhou, Junhua and Guo, Xiaojiao and Chen, Xuhang and Pun, Chi-Man and Zhou, Shoujun},
  journal={arXiv preprint arXiv:2512.03597},
  year={2025}
}
```
# 📋HBFormer
HBFormer: A Hybrid-Bridge Transformer for Microtumor and Miniature Organ Segmentation

**University of Macau & SIAT CAS**

# 项目结构
```
HBFormer/
├── configs/                     # 各数据集训练配置
│   ├── config_setting_synapse.py
│   ├── config_setting_lits2017.py
│   ├── config_setting_ACDC.py
│   └── config_setting_bladder.py
├── datasets/                    # 数据集加载与增广
│   ├── dataset.py               # Synapse（.npz 切片 / .npy.h5 体）
│   ├── dataset_lits2017.py
│   ├── dataset_ACDC.py
│   └── dataset_bladder.py
├── models/                      # 模型实现
│   ├── DWSegNet.py
│   ├── HBFormer.py
│   ├── DZZMamba.py
│   └── vmunet/
├── engine_synapse.py            # 训练/验证/评估（Synapse专用）
├── engine_lits2017.py           # LiTS2017 训练/验证
├── engine_ACDC.py               # ACDC 训练/验证
├── train_synapse.py             # Synapse 训练入口
├── train_lits2017.py            # LiTS2017 训练入口
├── train_ACDC.py                # ACDC 训练入口
├── train_bladder.py             # Bladder 训练入口
├── test_synapse.py              # Synapse 体数据评估（如有）
├── pre_trained_weights/         # 预训练/外部权重
├── results/                     # 输出：日志/权重/评估
├── 3D_Vis/                      # 3D/切片查看工具
└── utils.py                     # 通用工具、日志、优化器/调度器构建
```
- **多数据集**: Synapse（8器官）、LiTS2017（肝脏/肿瘤）、ACDC（心脏3类）、Bladder（膀胱/肿瘤）。
- **两种验证模式（Synapse）**: 体数据验证（.npy.h5）与切片验证（.npz，与训练一致）。
- **可视化**: 预测对比、注意力/激活热图、3D/切片交互查看。
- **日志与记录**: 训练/验证 CSV 记录、best/latest 权重保存、可配置保存间隔。**

---

## 环境安装

### 推荐（Python 3.8 + PyTorch 1.13，CUDA 11.7）
```bash
conda create -n hbformer python=3.8 -y
conda activate hbformer
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging timm==0.4.12 pytest chardet yacs termcolor
pip install submitit tensorboardX triton==2.0.0
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy
pip install opencv-python pandas tqdm nibabel scipy einops
```

可选：
```bash
pip install ptflops  # 计算FLOPs/参数量
```

也可参考：`ui/vmunet_environment.yml` 与 `ui/vmunet_conda_env.txt`。

---

## 数据集准备

### Synapse（8类）
目录要求（与代码默认一致）：
```
./data/Synapse/
├── lists/
│   └── lists_Synapse/
│       ├── all.lst
│       ├── train.txt
│       ├── test_vol.txt        # 体数据验证/测试
│       └── test_slice.txt      # 切片验证（与训练一致的数据预处理）
├── train_npz/                  # 切片级 .npz（训练 & 切片验证使用）
│   └── caseXXXX_sliceYYY.npz   # 含键 image/label
└── test_vol_h5/                # 体数据 .npy.h5（体验证使用）
    └── caseXXXX.npy.h5         # 含键 image/label
```

获取方式：可参考 Swin-UNet 的 Synapse 数据下载与预处理，或使用你已有的预处理结果。若需自制 `.npz`，可参考 `data_proprecessing.py` 示例将 NIfTI 切片为 `.npz`（确保 `.npz` 内含键 `image` 与 `label`，并与 `test_slice.txt` 命名一致）。

`test_slice.txt` 样例（每行一个切片名，不带扩展名）：
```text
case0001_slice001
case0001_slice002
...
```

### LiTS2017（肝脏/肿瘤）
```
./data/LITS2017_nii/
├── CT/                       # volume-*.nii
└── seg/                      # segmentation-*.nii
```

### ACDC（心脏3类）
```
./data/ACDC/
└── database/
    ├── training/patient*/patient*_frame*.nii.gz
    └── testing/patient*/
```

### Bladder（膀胱/肿瘤）
```
./data/Bladder/
├── images ...
└── masks  ...
```

---

## 使用与训练

所有训练入口脚本会读取对应配置文件（`configs/`），日志/权重输出目录位于 `results/{Model}_{Dataset}/`。

### 选择/切换模型（以 Synapse 为例）
在 `configs/config_setting_synapse.py` 中：
- `available_models = ['DWSegNet', 'AFFSegNet', 'vmunet', 'HBFormer']`
- 默认 `network = 'HBFormer'`
- 可通过类方法切换：
```python
from configs.config_setting_synapse import setting_config
setting_config.set_model('DWSegNet')  # 或 'HBFormer'、'vmunet'、'AFFSegNet'
```

### 训练命令
```bash
# Synapse（包含切片验证）
python train_synapse.py

# LiTS2017（支持交叉验证）
python train_lits2017.py

# ACDC
python train_ACDC.py

# Bladder
python train_bladder.py
```

### 验证与测试
- Synapse：
  - 切片验证由 `train_synapse.py` 内调用 `val_one_epoch_slice` 自动完成（读 `train_npz/` 与 `test_slice.txt`）。
  - 体验证（可选）使用 `engine_synapse.val_one_epoch`（读 `test_vol_h5/` 与 `test_vol.txt`）。
- 其他数据集：对应 `engine_*.py` 中的验证逻辑与 `test_*.py` 测试脚本（如提供）。

训练过程会在 `results/.../log/` 写入训练/验证日志，并将指标记录到 CSV：
- `train_record.csv`（loss、avg_dice、mIoU、各器官Dice）
- `val_record.csv`（avg_dice、avg_hd95、mIoU、各器官Dice/HD95）

Test Result:

```bash
### Synapse（8类）test result follow SwinUnet‘s test framework
================================================================================
 测试完成！
================================================================================
 Slice评估 Dice: 0.9856, HD95: 0.0000, mIoU: 0.9743
 Volume评估 Dice: 0.8782, HD95: 6.9185
 结果保存路径: test_result/HBFormer_synapse
️  预测可视化: test_result/HBFormer_synapse/prediction_visualization
 注意力热图: test_result/HBFormer_synapse/attention_heatmaps
⚡ 激活热图: test_result/HBFormer_synapse/activation_heatmaps
================================================================================
```


权重保存：
- `checkpoints/best.pth` / `best_dice.pth`（最佳）
- `checkpoints/latest.pth`（最近）
- 支持 `save_interval` 定期保存（见各配置）。
- Download Synapse result of HBFormer in [HuggingFace](https://huggingface.co/lzeeorno666/HBFormer-Synapse-Multi-Organ-Segmentation/tree/main/results/HBFormer_Synapse/checkpoints)

---

## 预训练权重

目录：`pre_trained_weights/`

提示：we use the swin Transformer's '[swin_tiny_patch4_window7_224.pth](https://huggingface.co/lzeeorno666/HBFormer-Synapse-Multi-Organ-Segmentation/tree/main/pre_trained_weights)'


---

## 可视化与3D查看
- 训练/验证过程支持保存预测对比、注意力/激活热图（见 `engine_synapse.py` 相关函数）。
- 目录 `3D_Vis/` 与 `Pathplanning/` 提供三维与切片查看工具，可按需运行其中的 `run.py`/`main.py` 类型脚本进行交互式浏览。

---

## 常见问题（FAQ）
- 切片验证报错尝试访问 `.npy.h5`？请确认 `datasets/dataset.py` 的逻辑：`split in ["train","test_slice"]` 读取 `.npz`，`split in ["test_vol","val_vol"]` 读取 `.npy.h5`，并确保 `list_dir` 下存在对应的 `test_slice.txt`/`test_vol.txt`。
- CUDA/依赖冲突：优先使用上文给定版本；或参考 `ui/vmunet_environment.yml` 创建一致环境。
- 日志/权重找不到：检查 `results/{Model}_{Dataset}/` 是否已创建；脚本会自动创建必要目录。


---

## 引用
若本项目对您的研究有帮助，欢迎引用或在论文中致谢本仓库。

---

## HBFormer 性能优化与调试过程总结

> 本节总结从“优化 HBFormer 性能”开始，到后续结构调整、SwinBackbone 兼容性修改以及多轮 Bug 排查直至 `train_synapse.py` 能够在 `seg` 环境中正常启动训练的全过程，便于后续复现和维护。

### 1. 问题与修改动机概览

- **目标一：提升 HBFormer 在 Synapse 上的性能与稳定性**  
  - 希望在保持/提升 Dice 指标的前提下，改进特征表达与训练稳定性，并支持 FLOPs/参数量的可视化统计。
- **目标二：统一与现代 timm Swin 实现的兼容性**  
  - 代码最初针对较老版本 `timm` 的 SwinTransformer 实现，随着库版本升级，Swin 的接口（特别是 PatchEmbed 输出格式、Stage/Block 的输入形状约定）发生变化，导致 HBFormer 中的 Swin 作为编码器时出现维度不匹配与运行错误。
- **目标三：保留多尺度特征、兼容 HBFormer 原有解码结构**  
  - HBFormer 依赖 Swin backbone 输出多尺度特征 (如 C1~C4)，这些特征需要满足特定的 `B x C x H x W` 形状及通道数，以便后续 `proj_enc*` 等 1×1 卷积投影层能正确工作。

基于上述目标，整个优化过程大致分为三类问题：

1. **SwinBackbone 与 timm.SwinTransformer 的接口/张量形状不匹配**。
2. **HBFormer 投影层与 Swin 输出通道数不一致**。
3. **FLOPs 统计（thop）阶段暴露的前向推理 Bug 与维度问题**。

下面按时间线梳理主要问题、排查思路与最终解决方案。

### 2. SwinBackbone 接口不匹配与维度问题

#### 2.1 初始报错与问题类型

- 在 `seg` 环境下运行：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate seg
cd /home/ipprlab/zfc/AFFSegNnet_VMUnetVis
python train_synapse.py
```

- 首轮报错主要集中在 `models/swin_backbone.py`：
  - **错误一：解包维度错误**  
    - 报错信息类似：`B, L, C = x.shape` 时出现 “too many values to unpack (expected 3)” 或 “not enough values to unpack”。  
    - 暗示 **Swin PatchEmbed 或某一 stage 的输出维度 (rank) 与预期不符**。
  - **错误二：reshape 超出张量 size**  
    - 例如：`x.transpose(1, 2).reshape(B, C, h, w)` 时提示目标形状与 `numel` 不一致。  
    - 表明 **我们通过 `sqrt(L)` 或硬编码的 `H=W=int(L**0.5)` 恢复空间尺寸的方式，在某些版本的 Swin 中已经不再成立**。

#### 2.2 根本原因分析

- 新版 `timm` 中 `SwinTransformer` 相关模块（特别是 `SwinTransformerStage`、`SwinTransformerBlock` 和 `PatchEmbed`）**对输入/输出形状有严格约定**：
  - `PatchEmbed` 输出往往是 **4D 张量**：`(B, H, W, C)` 或旧式 `3D (B, L, C)`，取决于实现版本。
  - `SwinTransformerStage` 和 `SwinTransformerBlock` 内部的 `_attn` 等函数一般要求 **输入为 `(B, H, W, C)`**，并在内部完成 window 分块与还原。
  - 原仓库中的 `SwinBackbone` 假定 Swin 所有层都是 `(B, L, C)` 形式并通过 `sqrt(L)` 恢复 `H, W`，这与当前 timm 版本存在偏差。

因此：

- **问题类型 A：PatchEmbed 输出形状变更导致的维度不匹配。**
- **问题类型 B：Swin Stage/Block 期望 4D HWC 输入，但 backbone 手动 flatten 变成 `B×L×C` 再传入，触发 layer norm 或 `_attn` 内部报错。**

#### 2.3 排查与修改思路

1. **直接 probe PatchEmbed 输出形状**  
   - 在 `seg` 环境下编写最小脚本，打印 `self.backbone.patch_embed(x)` 的形状：

   ```python
   from models.swin_backbone import SwinBackbone
   import torch

   m = SwinBackbone(pretrained_path=None, img_size=256, window_size=8)
   x = torch.randn(1, 3, 256, 256)
   pe_out = m.backbone.patch_embed(x)
   print('patch_embed out shape:', pe_out.shape)
   ```

   - 结果表明：**在当前 timm 版本中，PatchEmbed 输出为 `(B, H, W, C)`，例如 `(1, 64, 64, 96)`**。

2. **对照 timm 源码理解 Swin 正确的前向期望**  
   - 通过终端 `sed` 查看 `timm/models/swin_transformer.py` 内 `SwinTransformerBlock` 和 `SwinTransformerStage` 的 `forward`：
     - `SwinTransformerBlock.forward` 的第一行通常是：`B, H, W, C = x.shape`。  
     - `SwinTransformerStage.forward` 则基本是：
       ```python
       def forward(self, x):
           x = self.downsample(x)
           x = self.blocks(x)  # blocks 内部同样用 HWC
           return x
       ```
   - 由此确认：**整个 Swin 主干期望始终保持 4D HWC 形式**，而不是以前常见的 3D `B×L×C`。

3. **重新设计 SwinBackbone.forward 的数据流**  
   - 改造目标：
     - 输入：`(B, 3, H, W)`。
     - 经过 patch embed：`(B, H_pe, W_pe, C)`，整个 Swin 主干内部一直以 HWC 形式前向。  
     - 在我们需要输出多尺度特征时，再将 HWC 转成 HBFormer 需要的 `B×C×H×W`。  
   - 修改要点：
     1. **取消对 Swin Stage 的手动 flatten/reshape 操作**，直接把 4D `x` 传入 `layer(x)`。  
     2. 对于输出特征：
        - 在每个 stage（`for idx, layer in enumerate(self.backbone.layers)`）之后，`x` 的形状是 `(B, H_i, W_i, C_i)`。  
        - 通过 `x.permute(0, 3, 1, 2).contiguous()` 得到 `feat_i`，作为 HBFormer 的多尺度特征。  
     3. 关于 padding/cropping：
        - 仍在进入 Swin 前对输入图像做边缘 pad，使其尺寸可被 window 大小整除。  
        - 每个 stage 输出后，根据原始 pad 记录按比例裁剪回原分辨率，以避免 decode 阶段出现越界问题。

4. **兼容 window_size 为 tuple 的情况**  
   - timm 中 window size 可能为 `int` 或 `tuple`，如果要基于 window 对 H/W 做取模或 pad，需要统一成标量：
     ```python
     win = window_size[0] if isinstance(window_size, (tuple, list)) else window_size
     ```
   - 后续以 `win` 计算需要的 pad，避免 `TypeError: unsupported operand type(s) for %: 'int' and 'tuple'`。

#### 2.4 已遇 Bug 与对应思路

在多轮调整过程中出现了典型的几类报错：

1. **`ValueError: too many/not enough values to unpack`**  
   - 出现场景：试图用 `B, L, C = x.shape` 或 `B, H, W, C = x.shape` 时，`x.dim()` 与期望不一致。  
   - 排查方式：在关键位置插入 shape 打印或使用最小脚本，如：
     ```python
     print('before layer', idx, x.shape)
     ```
   - 解决策略：统一约定 Swin 内部所有 stage 只接收/返回 4D HWC 张量，任何 flatten 操作只在导出特征时局部使用。

2. **`RuntimeError: shape '...' is invalid for input of size ...`**  
   - 出现场景：将 `B×L×C` 误 reshape 成 `B×C×H×W`，而 `H×W≠L`。  
   - 根因：仍然沿用老版本 “通过 `sqrt(L)` 推测 H、W” 的做法。  
   - 最终方案：抛弃 `sqrt(L)` 推断，**改为在 PatchEmbed 处显式记录 `h, w`，并在整个 forward 流中沿用/更新这两个变量**。

3. **LayerNorm 维度错误：`Given normalized_shape=[96], expected input with shape [*, 96], but got [1, 64, 96, 64]`**  
   - 出现场景：向 `SwinTransformerBlock` 传入了 `B×H×C×W` 或 `B×W×H×C` 等错误顺序的 4D 张量。  
   - 根因：在尝试在 Swin 内部手动 permute（如 `x.transpose(1,2)`）之后再调用 `layer(x)`，破坏了 block 对输入为 `[*, C]` 的 LayerNorm 假设。  
   - 解决：**确保进入 timm 的 Swin block/stage 的张量顺序完全遵循源码约定 `(B, H, W, C)`，不要额外 permute C 轴**。

通过一轮轮修正，SwinBackbone 最终被简化为：

- 只在以下三个位置做形状处理：
  1. 输入前的 pad（`F.pad`）。
  2. PatchEmbed 输出后的 HWC→BCHW 或 BCHW→HWC 转换。  
  3. 每个 stage 输出时，将 HWC 转为 BCHW 作为多尺度特征，并根据原始 pad 进行裁剪。

### 3. HBFormer 通道数与投影层不匹配

#### 3.1 报错与问题类型

在解决完 SwinBackbone 的大部分 shape 问题后，`train_synapse.py` 在进行 FLOPs 统计（`cal_params_flops` 调用 `thop.profile`）或前向时出现：

- 报错示例：

```text
RuntimeError: Given groups=1, weight of size [96, 192, 1, 1], expected input[1, 96, 64, 64] to have 192 channels, but got 96 channels instead
```

这类报错典型地说明：

- **卷积权重 `weight` 的 in_channels 与输入特征图的通道数不一致**。  
  - 以 `weight of size [96, 192, 1, 1]` 为例：
    - 该 1×1 卷积期望 `in_channels=192, out_channels=96`。  
    - 实际输入特征的通道为 96，因此触发错误。

结合 HBFormer 代码，可以判断这些卷积层多为 encoder 特征通道投影层，例如：`self.proj_enc2(c2)`、`self.proj_enc3(c3)` 等。

#### 3.2 原因分析

- 预期设计：
  - 不同 Swin stage 输出的通道数为 `C1=96, C2=192, C3=384, C4=768`（对应 Tiny/Small 不同配置）。
  - HBFormer 中 `proj_enc2` 的卷积 weight 通常设为 `[C_target, C2, 1, 1]`。  
  - 因此只要 SwinBackbone 正确维护 stage 通道数，就不应有通道不匹配问题。

- 实际情况：
  - 在最初几轮手动 reshape / permute 调整中，曾错误地将某些 stage 的输出通道与空间维度混淆，  
    或在裁剪/flatten 后错误地再 reshape 导致通道数被压缩/展开。  
  - 最终表现为：**Swin C2 输出通道“被”处理成了 96，而卷积仍然以 192 为输入通道数初始化**。

#### 3.3 排查与解决思路

1. **打印多尺度特征的 shape**  
   - 在 HBFormer 中临时打印 SwinBackbone 输出（如 `c1, c2, c3, c4`）的 shape：

   ```python
   feats = self.MWAencoder(x_in)
   for i, f in enumerate(feats):
       print(f"feat[{i}] shape:", f.shape)
   ```

   - 期望：
     - `feat[0]: (B, 96, H/4,  W/4)`
     - `feat[1]: (B, 192, H/8,  W/8)`
     - `feat[2]: (B, 384, H/16, W/16)`
     - `feat[3]: (B, 768, H/32, W/32)`（根据具体 Swin 配置）。

2. **若发现通道不对，应优先检查 SwinBackbone 中对应 stage 的 permute/裁剪逻辑**  
   - 通道数是由 timm Swin 的 `embed_dim` 和 `out_dim` 决定，几乎不会在 stage 间自行“变成” 96；  
   - 更可能是我们在某个环节错误地做了：
     - `.view(B, H, W, C)` → 通道/空间轴顺序搞错；
     - `.flatten(2)` 与 `.transpose(1, 2)` 的顺序使用不当，导致 C 与 HW 维互换。  
   - 因此解决方案仍然回到：**将 SwinBackbone 中所有中间表示统一为 HWC，并只在导出特征时用简单的 `permute(0, 3, 1, 2)` 转回 BCHW**。

3. **避免临时性的“改卷积通道数”补丁**  
   - 在调试中，最直接的“修法”是把 `proj_enc2` 的 in_channels 改为 96；但这只是在现有错误特征通道基础上的“将错就错”，  
     会破坏模型结构设计初衷且影响性能。  
   - 正确做法应是 **修正出错的特征流**，而不是围绕错误的特征重新设计投影层。

最终，伴随 SwinBackbone 的 shape 流顺利梳理清楚，多尺度特征通道恢复正常后，上述卷积通道数错误随之消失。

### 4. FLOPs 统计阶段的特殊注意

- `train_synapse.py` 在 `main` 函数中会调用：

```python
cal_params_flops(model, config.input_size_w, logger)
```

- 其内部通过 `thop.profile(model, inputs=(input,))` 对整个 HBFormer+SwinBackbone 进行一次前向推理，用于统计参数量与 FLOPs。

- 这一步非常“严格”，因为：
  - 即使正常训练过程中某些维度错误被掩盖（例如从未访问到那条分支），在 FLOPs 统计中也会被完整跑一遍。  
  - 因此我们在调试过程中，**实际上是让 `thop.profile` 帮我们提前扫清了一些潜在的 shape/接口问题**。

调试建议：

1. 若只想先确认训练主流程是否可跑通，可以临时注释 `cal_params_flops` 调用，待模型结构稳定后再恢复统计。  
2. 当 FLOPs 统计报错时，应优先关注：
   - 该报错是否出现在 SwinBackbone 中的边缘分支（例如某些未在实际训练中使用的 path）；
   - 是否是由于我们在 `cal_params_flops` 中构造的 dummy input 尺寸与真实训练尺寸不一致导致的。

本次调试中，我们保持了 FLOPs 调用不变，通过修正 SwinBackbone 与 HBFormer 的接口，使其能够顺利完成完整前向，保证统计结果可用。

### 5. 总结与经验教训

1. **尽量遵循第三方库的“官方数据流”**  
   - 对于 `timm` 这类不断演进的模型库，强行沿用早期版本中 `B×L×C` 的习惯做法非常容易在升级后出问题。  
   - 更稳妥的方式是：
     - 仔细阅读当前版本源码，确认各模块 `forward` 的输入/输出约定（如 HWC vs BCHW）；
     - 在自定义 `Backbone` 封装中**尽可能少地做 shape hack**，优先使用库内部的 `set_input_size` 等接口。

2. **记录并显式维护空间尺寸 H/W，而非依赖 `sqrt(L)`**  
   - 当模型中存在多次下采样、pad/crop、window partition 等复杂操作时，靠 `sqrt(L)` 自动推断空间尺寸非常脆弱。  
   - 更好的实践是：
     - 在进入网络前记录原始 `H, W` 以及 pad 量；
     - 每次下采样/升采样时显式更新这两个变量；
     - 仅在需要 reshape 时使用它们。

3. **在 encoder-decoder 结构中优先保证“特征通道语义”的稳定性**  
   - 对于 UNet/Transformer encoder-decoder，某个 stage 输出的通道数通常与模型设计紧密相关（如 96/192/384/768），  
     下游很多模块都按此假设构建。  
   - 一旦中途改动导致通道数变化，应统一更新所有依赖模块；否则宁可去修正中间特征流，而不是给每个卷积单独“改口径”。

4. **利用 FLOPs/参数统计作为“结构体检”工具**  
   - `thop.profile` 虽然只是用来算 FLOPs，但其“全网前向一次”的特性，  
     实际上可以当作结构一致性测试：**只要统计能跑完，大概率多尺度连接/分支都没有 shape 问题**。

本节记录的 HBFormer + SwinBackbone 兼容性修复与性能调优过程，旨在作为后续结构修改和 `timm` 升级时的参考。若未来更换为新的 Swin 变体或其他 Transformer backbone，推荐沿用类似的调试思路：

1. 先搞清楚 backbone 官方的输入/输出形状；
2. 编写最小脚本单独验证 PatchEmbed 和每个 stage；
3. 再逐步与 HBFormer 的编码/解码结构对齐，并用 FLOPs 统计作为最终一致性检查。
