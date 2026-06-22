# tableReco0623

社保表格识别项目，目标是基于 Qwen2.5-VL-3B 与 ms-swift 做表格结构识别、宽表视觉增强、以及后续领域自适应训练。

## 当前协作方式

- 本地开发目录：`E:\tableReco0623`
- GitHub 仓库：`git@github.com:WEIJIE0213/tableReco0623.git`
- 服务器代码目录：`~/tableReco0623`
- 服务器配置：4 x RTX 4090 24GB
- 服务器 conda：`/usr/local/anaconda3/bin/conda`
- 服务器 GitHub 访问：HTTPS 443 被拦截，SSH 22 可用

代码同步优先走 Git：

```text
本地修改 -> git push 到 GitHub -> 服务器 git pull
```

服务器已验证可以通过 SSH 访问 GitHub，所以 `~/tableReco0623` 是正式 Git clone 目录。`scripts/push_to_server.ps1` 只作为 GitHub SSH 不可用时的备用 scp/tar 同步方案。

## 常用命令

本地提交并推送：

```powershell
git add -A
git commit -m "your message"
git push
```

服务器拉取并执行训练入口：

```bash
cd ~/tableReco0623
bash scripts/update_and_train.sh
```

服务器首次初始化训练环境：

```bash
cd ~/tableReco0623
bash scripts/server_bootstrap.sh
```

备用同步方案，本地直接上传代码到服务器：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\push_to_server.ps1
```

## 数据和模型产物

以下内容不进 Git：

- `data/`
- `checkpoints/`
- `runs/`
- `outputs/`
- `wandb/`
- 大模型权重和 checkpoint 文件

真实数据、合成数据、训练输出默认留在服务器本地。需要迁移少量样本时再单独用 `scp` 或 `rsync`。

## 目录

```text
configs/                 ms-swift 训练配置
data/                    数据目录，不进 Git
scripts/
  server_bootstrap.sh    服务器环境初始化
  update_and_train.sh    服务器 git pull + 训练入口
  sync.bat               本地 commit + push 辅助脚本
  push_to_server.ps1     备用 scp/tar 同步脚本
src/                     后续模型、数据处理、评估代码
SETUP.md                 首次配置和排障记录
```
