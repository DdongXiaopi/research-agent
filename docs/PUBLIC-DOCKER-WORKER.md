# 在本机启用渲染工具

渲染 worker 是可选能力，提供单文件 pdfLaTeX 和结构化数值绘图。部署需要运行中的本地 Docker Linux 引擎、可下载 Debian 软件包的网络和足够磁盘空间。运行任务时容器不使用网络、主机挂载或任意脚本执行。

在公开仓库根目录、PowerShell 和已安装项目依赖的 Python 3.12 环境中运行：

```powershell
$env:RSI_HOME = Join-Path $PWD 'outputs/docker-worker/runtime'
docker context ls
docker --context desktop-linux pull debian:bookworm-slim
.\.venv\Scripts\python.exe scripts/deploy_docker_worker.py --context desktop-linux --output-config outputs/docker-worker/profile.yaml
```

`desktop-linux` 是 Docker Desktop Linux 引擎的常见 context 名称；其他安装应从 `docker context ls` 选择指向本地 Unix socket 或 Windows named pipe 的 context，并在两条命令中使用同一名称。部署从本地镜像的 registry digest 构建，保存构建记录，并实际检查容器隔离、PDF 与三种图像格式。全部检查通过后才生成配置；已存在的输出配置不会被覆盖。环境变量占位符保留在生成配置中，使用时仍须提供对应环境变量。

生成的配置与部署收据绑定本机镜像、代码和路径。示例将它们放在默认被 Git 忽略的 `outputs/`；自定义输出位置时，应自行避免提交机器路径、私有配置和收据。代码或镜像变化后需要重新部署，不能复制另一台机器的收据当作本机部署证明。

将下列输入保存为 `outputs/docker-worker/plot.json`：

```json
{"kind":"plot","payload":{"title":"Synthetic example","xlabel":"x","ylabel":"y","series":[{"label":"sample","x":[0,1,2],"y":[0,1,4]}],"formats":["png","pdf","svg"]}}
```

先生成计划，再用输出的精确哈希执行，最后验证返回的运行目录。若打开了新的 PowerShell 窗口，先重新设置上面的 `RSI_HOME`，让核验入口找到同一运行目录：

```powershell
.\.venv\Scripts\python.exe scripts/research_assistant.py tool --config outputs/docker-worker/profile.yaml --input outputs/docker-worker/plot.json
.\.venv\Scripts\python.exe scripts/research_assistant.py tool --config outputs/docker-worker/profile.yaml --input outputs/docker-worker/plot.json --execute --expected-plan-sha256 <计划哈希>
.\.venv\Scripts\python.exe scripts/research_assistant.py verify-tool --run-dir <返回的运行目录>
```

`verify-tool` 的 `result_ok` 表示工具是否成功，验证状态表示保存的结果与收据身份是否一致。需要编译时，把输入改为 `{"kind":"latex_compile","payload":{"source":"\\documentclass{article}\\begin{document}Hello.\\end{document}"}}`。该接口不支持多文件 TeX、XeLaTeX 或任意 Python；详情见[公开能力范围](PUBLIC-RELEASE-SCOPE.md)。
