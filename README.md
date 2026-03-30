
# 远程服务器开发环境配置指南 (GitHub 模板版)

本指南旨在帮助开发者快速完成从 Docker 容器初始化、SSH 远程连接、到 TensorBoard 监控及 VS Code 调试环境的完整搭建流程。

---

## 1. 基础环境初始化

在进入容器或新服务器后，首先完成基础网络标识与 SSH 服务的开启。

### 1.1 获取实例标识
```bash
# 查看当前实例的内部 IP/主机名
hostname -i
```

### 1.2 启动 SSH 服务与权限设置
```bash
# 1. 启动服务
service ssh start

# 2. 设置 root 密码 (用于 VS Code 初次连接认证)
passwd root
# 按照提示输入两次密码并妥善记录

# 3. 允许 Root 登录与密码认证 (修改配置文件)
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
```

### 1.3 配置免密登录
将本地主机的公钥（通常为 `~/.ssh/id_rsa.pub`）添加到服务器中：
```bash
vim ~/.ssh/authorized_keys
# 将公钥内容粘贴至此处保存
```

---

## 2. VS Code 远程连接配置

### 2.1 本地 SSH Config 配置
在本地电脑修改 `~/.ssh/config` 文件，添加跳板机或目标机配置：
```text
Host target_server
    HostName [SERVER_IP]
    User root
    Port [MAPPED_PORT]
```

### 2.2 验证与调试
如果连接失败，可以在本地终端通过详细模式排查端口映射或认证问题：
```bash
ssh target_server -vvv
```
> **注意**：如连接不通，请优先检查 `/etc/ssh/sshd_config` 中的端口是否与容器映射端口一致。

---

## 3. TensorBoard 安装与远程监控

### 3.1 离线安装
若服务器处于内网环境，可将安装包传输至服务器后执行：
```bash
pip install --no-index --find-links=. tensorboard
```

### 3.2 端口转发
在本地执行 SSH 指令，将服务器端口映射至本地：
```bash
ssh -L 6006:localhost:6006 target_server
```

### 3.3 启动服务
使用 `nohup` 后台运行 TensorBoard：
```bash
# 建议 logdir 指定到子目录以区分不同实验
nohup tensorboard --logdir=/path/to/logs --port=6006 --bind_all > tb_run.log 2>&1 &

# 管理进程
ps -ef | grep tensorboard  # 查看进程 ID
kill -9 [PID]              # 停止服务
```

---

## 4. 高级调试与代理配置

### 4.1 方案 A：VS Code Server 自动转发 (推荐)
通过 `RemoteForward` 实现反向代理，使服务器共用本地科学上网环境。

**SSH 配置：**
```text
Host target_server
    HostName [SERVER_IP]
    User root
    RemoteForward [REMOTE_PORT] 127.0.0.1:[LOCAL_PROXY_PORT]
```
**server config**
```text
# 配置服务器端端口
export http_proxy=http://127.0.0.1:[LOCAL_PROXY_PORT]
export https_proxy=http://127.0.0.1:[LOCAL_PROXY_PORT]
```
**VS Code 设置：**
1. 搜索 `Http: Proxy`，填入 `http://127.0.0.1:[REMOTE_PORT]`。
2. 取消勾选 `Http: Proxy Strict SSL`。
3. 设置 `Http: Proxy Support` 为 `override`。

### 4.2 方案 B：手动部署 VS Code Server (离线场景)
1. 创建目录：`mkdir -p ~/.vscode-server/bin`
2. 解压预下载的服务端包：
   ```bash
   tar -zxf vscode_all_servers.tar.gz -C ~/.vscode-server/bin/
   ```

---

## 5. 开发与推理实践

### 5.1 跨服务器文件传输
在本地执行，利用 `-3` 参数在两个远程服务器间中转数据：
```powershell
scp -3 -r root@[IP_A]:/path/to/source root@[IP_B]:/path/to/destination
```

### 5.2 运行环境准备
推理时指定 `PYTHONPATH` 与显卡：
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd) && \
export CUDA_VISIBLE_DEVICES=0,1 && \
python ./path/to/inference_script.py
```

---

## 附录：TensorBoard 可视化工具类
项目中可封装如下 Logger 类，用于在训练过程中记录 Loss 并自动将 Latent 解码为视频/图片。

```python
import torch
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """独立的 TensorBoard 日志记录器，支持 Loss 与视频解码可视化"""
    def __init__(self, log_dir="/path/to/log", vis_every=500):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.vis_every = vis_every
        self.global_step = 0

    def log_loss(self, loss_val):
        self.writer.add_scalar("train/loss", loss_val, self.global_step)
        self.global_step += 1

    @torch.no_grad()
    def log_visualization(self, pipe, noise_pred, training_target, latents, timestep):
        step = self.global_step - 1
        if step % self.vis_every != 0 or step == 0:
            return
        
        try:
            # 此处实现从 Latent 到图像/视频的解码逻辑
            # ... (代码逻辑详见源码)
            self.writer.add_video(f"train/pred_video", vid_tensor, step, fps=8)
        except Exception as e:
            print(f"[TBLogger] Visualization failed: {e}")

    def close(self):
        self.writer.close()
```

---
**提示**：请根据实际项目路径替换文档中的 `[SERVER_IP]`、`[REMOTE_PORT]` 等占位符。

---
