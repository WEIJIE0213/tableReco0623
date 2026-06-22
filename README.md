# tableReco0623 — 社保表格识别（Qwen2.5-VL-3B + ms-swift）

本地开发（Windows，由 Claude 编写代码）+ 服务器训练（4×RTX 4090，SSH 登录运行）。

## 协作与同步模型

- **代码**:通过 Git 在「本地 ↔ GitHub ↔ 服务器」之间同步。小、可版本管理、可复现。
- **数据**:不走 git，留在服务器本地（`data/` 已被 `.gitignore` 忽略）。需要时用 scp/rsync 传。
- **Checkpoint/日志**:同样留服务器本地（`checkpoints/`、`runs/` 已忽略）。

## 每轮迭代怎么走

1. Claude 在本地 `E:\tableReco0623` 改代码。
2. 你双击 `scripts\sync.bat` → 自动 commit & push 到 GitHub。
3. 服务器上跑 `bash scripts/update_and_train.sh` → 自动 `git pull` 后开训。
4. 你把服务器的报错/日志复制回来给 Claude → Claude 改 → 回到第 1 步。

## 首次配置

见 `SETUP.md`（创建仓库、配置 SSH 密钥、服务器初始化、首次同步自测）。

## 目录

```
data/              # 数据（不入 git，留服务器）
configs/           # ms-swift 训练配置
src/               # 模型 / 数据处理 / 脚本
scripts/           # 同步与训练脚本
  sync.bat               # 本地一键 push（Windows）
  update_and_train.sh    # 服务器一键 pull + 训练
  server_bootstrap.sh    # 服务器一次性环境初始化
SETUP.md           # 首次配置步骤
```
