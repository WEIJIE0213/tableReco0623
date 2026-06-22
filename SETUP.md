# 首次配置与服务器同步记录

本文档记录当前已经验证通过的同步方式，以及后续需要执行的服务器初始化命令。

## 已验证结论

服务器 `10.200.97.195` 的网络状态：

| 测试项 | 结果 |
| --- | --- |
| DNS 解析 | 正常 |
| ICMP Ping | 正常 |
| TCP 80 | 正常 |
| TCP 22 | 正常 |
| TCP 443 | 超时/不通 |

结论：

- GitHub HTTPS 访问会失败，例如 `https://github.com/...` 或 `curl -I https://github.com`。
- GitHub SSH 访问可用，因此服务器必须使用 SSH URL：`git@github.com:WEIJIE0213/tableReco0623.git`。
- 当前服务器已经可以通过 SSH 认证到 GitHub，`git ls-remote` 和 `git pull` 均已验证通过。

## 当前路径

本地：

```text
E:\tableReco0623
```

GitHub：

```text
git@github.com:WEIJIE0213/tableReco0623.git
```

服务器：

```text
/home/ywj/projects/tableReco0623
```

旧的 scp/tar 临时同步目录已备份为：

```text
~/tableReco0623_scp_backup_before_git
```

旧路径仍保留为软链接，方便兼容早期命令：

```text
/home/ywj/tableReco0623 -> /home/ywj/projects/tableReco0623
```

## 本地日常同步

推荐直接用 Git：

```powershell
cd E:\tableReco0623
git status
git add -A
git commit -m "your message"
git push
```

也可以双击：

```text
scripts\sync.bat
```

## 服务器日常更新

服务器执行：

```bash
cd /home/ywj/projects/tableReco0623
bash scripts/update_and_train.sh
```

当前自检输出应包含：

```text
[update] git pull origin main
已经是最新的。
[train] sync path is working; training config will be added in the next phase
```

如果还没有创建 `tablereco` 环境，会看到：

```text
[env] conda env 'tablereco' does not exist yet; run bash scripts/server_bootstrap.sh before real training
```

这是正常的，说明代码同步链路已经通了，只是训练环境还没初始化。

## 初始化服务器训练环境

服务器 conda 路径已确认：

```text
/usr/local/anaconda3/bin/conda
```

首次安装训练依赖：

```bash
cd /home/ywj/projects/tableReco0623
bash scripts/server_bootstrap.sh
```

该脚本会：

- 自动找到 conda
- 创建 `tablereco` 环境
- 使用 Python 3.10
- 安装 `ms-swift`、`transformers`、`accelerate`、`deepspeed`、`vllm`、`qwen-vl-utils`、`datasets` 等依赖
- 检查 PyTorch CUDA 与 GPU 数量

## 备用同步方案

如果以后 GitHub SSH 也不可用，使用本地 scp/tar 备用方案：

```powershell
cd E:\tableReco0623
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\push_to_server.ps1
```

这个方案会把当前 Git HEAD 打包上传到服务器 `/home/ywj/projects/tableReco0623`，但不会保留服务器目录里的 `.git`。现在 Git SSH 已通，默认不使用它。

## 注意事项

- 数据、checkpoint、训练日志不进 Git。
- 服务器的 `data/`、`checkpoints/`、`runs/`、`outputs/` 默认留在服务器本地。
- `.sh` 文件必须保持 LF 换行；项目已通过 `.gitattributes` 强制处理。
- 如果服务器 `git pull` 报 GitHub 认证问题，先测试：

```bash
ssh -T git@github.com
git ls-remote git@github.com:WEIJIE0213/tableReco0623.git HEAD
```
