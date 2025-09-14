# /home/zlx/chat-wukong/generation_dataset/DashScope_LLM.py

import dashscope
import os
import time
import random
from langchain.llms.base import LLM
from typing import Dict, List, Optional, Any
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# 阿里云百炼API配置
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")

def get_completion(prompt: str, 
                   model: str = "qwen-plus", 
                   temperature: float = 0.7, 
                   max_retries: int = 5,
                   top_p: float = 0.8,
                   max_tokens: int = 2048) -> str:
    '''
    调用DashScope API获取文本生成结果
    
    Args:
        prompt: 输入提示词
        model: 调用的模型，可选: qwen-turbo, qwen-plus, qwen-max
        temperature: 温度系数，控制输出的随机程度 (0-2.0)
        max_retries: 最大重试次数
        top_p: nucleus sampling参数 (0-1.0)
        max_tokens: 最大生成token数
    
    Returns:
        str: 生成的文本内容
    '''
    if not DASHSCOPE_API_KEY:
        raise ValueError("未设置 DASHSCOPE_API_KEY 环境变量")
    
    # 设置API密钥
    dashscope.api_key = DASHSCOPE_API_KEY
    
    for attempt in range(max_retries):
        try:
            # 添加随机延时，避免频率限制
            if attempt > 0:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"🔄 第{attempt+1}次重试，等待 {wait_time:.1f} 秒...")
                time.sleep(wait_time)
            
            # 调用阿里云百炼API
            response = dashscope.Generation.call(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=1.1,  # 减少重复
                seed=42  # 固定种子以获得一致结果
            )
            
            # 检查响应状态
            if response.status_code == 200:
                return response.output.text
            else:
                error_msg = f"API调用失败: {response.status_code}, {response.message}"
                if attempt < max_retries - 1:
                    print(f"⚠️  {error_msg}，准备重试...")
                    continue
                else:
                    raise Exception(error_msg)
                    
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  请求失败: {str(e)}，准备重试...")
                continue
            else:
                raise Exception(f"API请求失败: {str(e)}")

class DashScope_LLM(LLM):
    """阿里云百炼平台的大语言模型接口"""
    
    model_name: str = "qwen-turbo"  # 默认使用qwen-turbo，速度快且稳定
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 2048
    max_retries: int = 5

    def __init__(self, 
                 model_name: str = "qwen-turbo", 
                 temperature: float = 0.7,
                 top_p: float = 0.8,
                 max_tokens: int = 2048,
                 max_retries: int = 5):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    @property
    def _llm_type(self) -> str:
        return "dashscope"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        调用语言模型生成文本
        
        Args:
            prompt: 输入提示词
            stop: 停止词列表
            **kwargs: 其他参数
            
        Returns:
            str: 生成的文本
        """
        try:
            res = get_completion(
                prompt=prompt,
                model=self.model_name,
                temperature=self.temperature,
                max_retries=self.max_retries,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )
            return res
        except Exception as e:
            raise Exception(f"模型调用失败: {str(e)}")
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """获取标识参数"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries
        }

# 使用示例
if __name__ == "__main__":
    # 创建模型实例
    llm = DashScope_LLM(
        model_name="qwen-plus",
        temperature=0.7,
        max_tokens=1024
    )
    
    # 测试调用
    try:
        result = llm("你好，请介绍一下你自己")
        print("生成结果:", result)
    except Exception as e:
        print(f"调用失败: {e}")