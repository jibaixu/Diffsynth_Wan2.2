from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility

api = HubApi()
api.login('ms-ae02ff7d-db79-48c3-b0b4-36967e8cdbdf')

repo_id = f"jibaixu/Wan_ATM"

try:
    api.get_model(repo_id)
except Exception:
    print(f"创建新模型库: {repo_id}")
    api.create_model(
        model_id=repo_id,
        visibility=ModelVisibility.PUBLIC,  # PUBLIC(5) 或 PRIVATE(1)
        license=Licenses.APACHE_V2,         # 许可证类型
        chinese_name="你的模型中文名"        # 可选
    )

# 4. 上传模型文件夹
api.upload_folder(
    repo_id=repo_id,
    folder_path='/data_jbx/Codes/Diffsynth_Wan2.2/Ckpt/wan_atm001B_480_640_202604101603/epoch-36',  # 本地模型文件夹绝对路径
    commit_message="Initial model upload",
    # 可选参数：
    # path_in_repo='subfolder',           # 上传到仓库的子目录
    # allow_patterns=['*.bin', '*.json'], # 只上传匹配的文件
    # ignore_patterns=['*.log', '.git'],  # 忽略的文件
    # max_workers=8,                      # 上传线程数
    # revision='master'                   # 分支名
)