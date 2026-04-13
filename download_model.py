from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

api = HubApi()
api.login('ms-ae02ff7d-db79-48c3-b0b4-36967e8cdbdf')

repo_id = 'jibaixu/Metric'

# 下载整个模型仓库
local_dir = snapshot_download(
    repo_id,
    cache_dir='/data_jbx/Codes/Diffsynth_Wan2.2/diffsynth/core/metric/',  # 下载到本地目录
    revision='master',  # 分支名，默认为 master
    # 可选参数：
    # allow_file_pattern=['*.bin', '*.json'],  # 只下载匹配的文件
    # ignore_file_pattern=['*.log', '.git'],   # 忽略的文件
)

print(f"模型已下载到: {local_dir}")
