# mambaBEV：基于 MMDetection3D 的 mambaBEV 论文复现

本项目基于 **MMDetection3D v1.4.0** 与 **Mamba（mamba-ssm）**，  
复现 **mambaBEV** 论文在 **nuScenes v1.0-trainval** 数据集上的实验结果。

实现方式采用 **MMDetection3D Plugin（插件）形式**，  
训练与测试统一使用官方脚本：

```
mmdetection3d/tools/train.py
mmdetection3d/tools/test.py
```

当前已支持并对齐论文中的两个配置：

- **mambaBEV-Tiny**（R50，800×450，3 帧历史）
- **mambaBEV-Base**（R101，1600×900，4 帧历史）

---

## 1. 环境与依赖安装

### 1.1 系统环境

本项目在以下环境下验证通过：

- 操作系统：Ubuntu 20.04  
- Python：3.8  
- PyTorch：2.0.0  
- CUDA：11.8  
- MMDetection3D：v1.4.0  
- Mamba：mamba-ssm 2.2.2  

> **AutoDL 用户说明**  
> 使用 AutoDL 时可直接勾选 PyTorch 2.0 + CUDA 11.8 + Python 3.8，  
> **无需使用 conda 创建环境**。

---

### 1.2 Python 依赖安装

```bash
# 安装 MMDetection3D（推荐 editable 模式）
cd mmdetection3d
pip install -e .

# 安装 mamba
pip install mamba-ssm==2.2.2
```

（可选但推荐）

```bash
pip install -U openmim
mim install mmengine mmcv
```

---

### 1.3 项目目录结构

```text
(root)
├── mambaBEV
│   ├── mambabev
│   ├── configs
│   ├── tools
│   └── README.md
└── mmdetection3d
    ├── data
    │   └── nuscenes
    ├── projects
    │   └── DETR3D
    ├── mmdet3d
    └── tools
```

---

## 2. 数据集准备（nuScenes）

### 2.1 下载数据集

下载 **nuScenes v1.0-trainval**，并解压到：

```text
mmdetection3d/data/nuscenes/
```

---

### 2.2 生成 nuScenes Info 文件（必须）

该步骤 **只需执行一次**：

```bash
cd mmdetection3d

python tools/create_data.py nuscenes \
  --root-path data/nuscenes \
  --out-dir data/nuscenes \
  --extra-tag nuscenes
```

生成文件示例：

```text
nuscenes_infos_train.pkl
nuscenes_infos_val.pkl
```

---

## 3. 让项目运行起来

### 3.1 设置 PYTHONPATH（必须）

由于 **mambaBEV 以插件形式实现**，需加入 `PYTHONPATH`。

在 **mmdetection3d 根目录**执行：

```bash
export PYTHONPATH=$(pwd)/../mambaBEV:$(pwd):$PYTHONPATH
```

或直接安装为 editable 包（可选）：

```bash
pip install -e mmdetection3d
pip install -e mambaBEV
```

---

### 3.2 模块注册说明

在配置文件中已显式调用：

```python
import mambabev
mambabev.register_all_modules()
```

确保以下模块被正确注册：

- `MambaBEVDetector`
- `TemporalMamba`
- `MambaDETR3DHead`

---

## 4. 训练模型

统一使用 MMDetection3D 官方脚本：

```
mmdetection3d/tools/train.py
```

---

### 4.1 训练 mambaBEV-Tiny

```bash
cd mmdetection3d

python tools/train.py \
  ../mambaBEV/configs/mambabev_tiny_r50_800x450_3f.py
```

**Tiny 设置（论文对齐）**：

- Backbone：ResNet-50  
- 图像分辨率：800 × 450  
- BEV 网格：50 × 50  
- 历史帧数：3  
- Batch size：1  
- 优化器：AdamW（lr=2e-4，weight_decay=0.01）  
- 无额外数据增强  

---

### 4.2 训练 mambaBEV-Base

```bash
cd mmdetection3d

python tools/train.py \
  ../mambaBEV/configs/mambabev_base_r101_1600x900_4f.py
```

**Base 设置（论文对齐）**：

- Backbone：ResNet-101  
- 图像分辨率：1600 × 900  
- BEV 网格：200 × 200  
- 历史帧数：4  
- Encoder 层数：6  

---

### 4.3 多卡训练（可选）

```bash
cd mmdetection3d

torchrun --nproc_per_node=8 tools/train.py \
  ../mambaBEV/configs/mambabev_tiny_r50_800x450_3f.py \
  --launcher pytorch
```

---

## 5. 测试与评估

### 5.1 在 nuScenes 验证集上测试

```bash
cd mmdetection3d

python tools/test.py \
  ../mambaBEV/configs/mambabev_tiny_r50_800x450_3f.py \
  work_dirs/mambabev_tiny_r50_800x450_3f/latest.pth \
  --eval bbox
```

Base 模型测试：

```bash
python tools/test.py \
  ../mambaBEV/configs/mambabev_base_r101_1600x900_4f.py \
  work_dirs/mambabev_base_r101_1600x900_4f/latest.pth \
  --eval bbox
```

---

### 5.2 论文结果参考（nuScenes Val）

| 模型 | Backbone | mAP | NDS |
| ---- | -------- | --- | --- |
| mambaBEV-Tiny | R50 | ≈0.368 | ≈0.262 |
| mambaBEV-Base | R101 | ≈0.410 | ≈0.508 |

> 实际结果可能因随机种子与硬件略有差异。

---

## 6. 注意事项

- **Batch size 必须为 1**（论文设置）
- **不要启用 GridMask / 翻转 / 旋转等增强**
- 若启动失败，请检查：
  - `PYTHONPATH` 是否正确  
  - `projects/DETR3D` 是否可被 import  
  - nuScenes info 文件是否已生成  
- 推荐首次运行做最小测试：

```bash
python tools/train.py <config> --cfg-options train_cfg.max_iters=1
```

---

## 7. 致谢

- MMDetection3D  
  https://github.com/open-mmlab/mmdetection3d  
- nuScenes 数据集  
  https://www.nuscenes.org/  
- Mamba / mamba-ssm  
  https://github.com/state-spaces/mamba

