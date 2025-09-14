# filepath: /home/zlx/chat-wukong/modelscope_download.py
from modelscope.hub.snapshot_download import snapshot_download
import os

# ModelScope 上的模型 ID
repo_id = 'qwen/Qwen2.5-7B-Instruct'
# 你希望保存到的本地路径
target_dir = '/home/zlx/chat-wukong/model/Qwen2.5-7B-Instruct'

print(f"🚀 开始从 ModelScope 下载: {repo_id} -> {target_dir}")

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

model_dir = snapshot_download(
    repo_id,
    # 建议为 modelscope 单独设缓存
    cache_dir='/home/zlx/chat-wukong/model/.cache_ms',
    local_dir=target_dir,
    # revision='master' # 可指定分支
)

print(f"✅ 模型已下载到: {model_dir}")