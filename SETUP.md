# 首次配置：打通本地 ↔ GitHub ↔ 服务器

按顺序做一遍，最后做一次同步自测。涉及账号/密钥的步骤只有你能做。

---

## 步骤 0：在 GitHub 创建空仓库

1. GitHub → New repository → 名字填 `tableReco0623` → 选 **Private** → **不要**勾选 README/.gitignore/license（保持空）→ Create。
2. 记下仓库地址。本文示例用 SSH 形式：`git@github.com:WEIJIE0213/tableReco0623.git`
   （若用 HTTPS 形式则是 `https://github.com/WEIJIE0213/tableReco0623.git`）

---

## 步骤 1：本地（Windows 桌面）配置

> 需要先装好 Git for Windows。

### 1a. 配置 GitHub SSH 密钥（推荐，一次性）
打开 Git Bash：
```bash
ssh-keygen -t ed25519 -C "weijie-desktop"   # 一路回车
cat ~/.ssh/id_ed25519.pub                   # 复制输出的整行公钥
```
把公钥粘到 GitHub → Settings → SSH and GPG keys → New SSH key → 保存。
测试：
```bash
ssh -T git@github.com    # 出现 "Hi WEIJIE0213! ..." 即成功
```

### 1b. 初始化本地仓库并首次推送
在 `E:\tableReco0623` 打开 Git Bash（或 PowerShell）：
```bash
cd /e/tableReco0623
git init
git branch -M main
git remote add origin git@github.com:WEIJIE0213/tableReco0623.git
git add -A
git commit -m "init: sync scaffold"
git push -u origin main
```

---

## 步骤 2：服务器（4×4090，SSH 登录）配置

### 2a. 配置 GitHub SSH 密钥（在服务器上同样做一次）
```bash
ssh-keygen -t ed25519 -C "weijie-server"
cat ~/.ssh/id_ed25519.pub      # 复制，加到 GitHub 的 SSH keys（可与桌面共用一个账号，加第二把 key 即可）
ssh -T git@github.com
```
> 若服务器在内网、连不上 github.com：告诉我，我们改用「桌面 push → 桌面 rsync 到服务器」的备用方案。

### 2b. clone + 初始化环境
```bash
cd ~                       # 或你想放项目的目录
git clone git@github.com:WEIJIE0213/tableReco0623.git
cd tableReco0623
bash scripts/server_bootstrap.sh    # 建 conda 环境、装 ms-swift 等
```

---

## 步骤 3：同步自测（确认链路打通）

1. 让 Claude 在本地改一个小文件（比如往 README 加一行）。
2. 你双击 `scripts\sync.bat` → 应看到 push 成功。
3. 服务器上执行：
   ```bash
   cd ~/tableReco0623
   bash scripts/update_and_train.sh
   ```
   应看到 `git pull` 拉到刚才的改动，并打印「同步链路已通，等待 Claude 补训练配置」。

看到这句话，就说明 **本地→GitHub→服务器** 全链路通了。

---

## 数据怎么传（不走 git）

- 服务器自己生成的合成数据：留服务器本地 `data/synth/`。
- 你本地标注的真实数据 → 传服务器：
  ```bash
  rsync -avz -e ssh ./data/real/  用户名@服务器IP:~/tableReco0623/data/real/
  ```
- 需要把少量样本数据传给 Claude 调试时：scp 下来放到本地，再发我即可。
