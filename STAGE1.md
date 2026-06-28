# 内容一：表格结构+内容识别（Qwen2.5-VL-3B LoRA）操作手册

## 数据布局（data/ 不进 Git，留本地/服务器）

```
data/
  synth/serif/
    images/   3000 张 PNG
    labels/   3000 份逐图 JSON（含 bbox，保留给内容二）
    manifest.json
  jsonl/      train/val/test.jsonl（由 build_dataset.py 生成，训练直接喂这个）
  splits/     train/val/test.txt（样本 id）
```

## 代码

- `src/build_dataset.py`：标注 → ms-swift jsonl（目标=HTML，含 rowspan/colspan/空格；不含坐标），并做分层划分。
- `src/checker.py`：结构合法性校验（占用矩阵无重叠/越界/漏覆盖 + 表头边引用）。
- `scripts/train_stage1.sh`：LoRA 微调（smoke / full）。

## 本地已完成

- 数据已就位 `data/synth/serif/`（3000 配对，校验 0 问题）。
- 已生成 `data/jsonl/{train,val,test}.jsonl` = 2390 / 290 / 320，HTML 与标注单元格数比对一致。

## 服务器执行步骤

1. **传数据到服务器**（data/ 不走 Git，需单独传）：
   ```powershell
   # 本地 PowerShell，把数据传到服务器项目目录
   scp -r E:\tableReco0623\data\synth ywj@10.200.97.195:/home/ywj/projects/tableReco0623/data/
   ```
2. **拉代码 + 建环境**（首次）：
   ```bash
   cd /home/ywj/projects/tableReco0623
   git pull origin main
   bash scripts/server_bootstrap.sh
   ```
3. **生成 jsonl**（服务器上重建，路径才正确）：
   ```bash
   python src/checker.py --data-root data/synth/serif
   python src/build_dataset.py --data-root data/synth/serif --out data/jsonl
   ```
4. **冒烟验证再正式训练**：
   ```bash
   bash scripts/train_stage1.sh smoke   # 20 步跑通
   bash scripts/train_stage1.sh full    # 正式
   ```

## 备注

- `MAX_PIXELS` 默认约 1.5M，用于压住 3508×2480 高分辨率的显存（内容二会专门做表格感知压缩替代它）。
- 训练目标当前是纯 HTML（结构+内容）。内容一的"表头层级/合并/空"多任务监督、约束解码，后续在此基线上加。
