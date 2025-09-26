# Chat-WuKong: 悟空对话大模型

本项目旨在通过 LoRA 高效微调技术，训练一个能以《西游记》中孙悟空的口吻和性格进行对话的大语言模型。项目完整地展示了从原始文本处理、自动化指令数据集构建，到模型微调和最终推理对话的全过程。

## 项目特点

*   **自动化数据集构建**：使用 `kor` 框架结合大模型 API（如通义千问），从《西游记》白话文原著中自动抽取出对话，生成指令微调数据集。
*   **高效微调**：采用 `PEFT` 库的 LoRA (Low-Rank Adaptation) 技术，仅用少量计算资源即可在消费级显卡上完成对 7B 级别大模型的微调。
*   **完整的技术栈**：项目代码覆盖了数据处理、模型训练、模型推理等环节，是一个端到端的 LLM 微调实践案例。
*   **模型基础**：基于 `Qwen/Qwen2.5-7B-Instruct` 模型进行微调。

## 项目结构

```
/home/zlx/chat-wukong/
├── dataset/                # 数据集存放目录
│   ├── input/wukong/       # 存放原始小说文本，如 西游记白话文.txt
│   └── train/lora/         # 存放处理好的、用于训练的 JSON 格式数据集
├── generation_dataset/     # 数据集生成模块
│   ├── main.py             # 自动化抽取对话、生成数据集的主脚本
│   └── DashScope_LLM.py    # 对接通义千问 API 的封装
├── model/                    # 存放下载的预训练基础模型
│   └── Qwen2.5-7B-Instruct/
├── output/                   # 训练输出目录
│   └── qwen2.5-7B-instruct_lora/ # 存放训练好的 LoRA 适配器权重
├── model_download.py         # 下载预训练模型的脚本
├── train.py                  # LoRA 微调训练脚本
└── chat.py                   # 与微调后模型进行对话的脚本
```

## 环境准备

1.  **克隆项目**
    ```bash
    git clone https://github.com/chenchong911/Chat-WuKong.git
    cd Chat-WuKong
    ```

2.  **安装依赖**
    建议创建一个 conda 虚拟环境，然后安装所需依赖。
    ```bash
    # 假定你已创建并激活了名为 wukong 的环境
    pip install torch transformers datasets peft accelerate bitsandbytes
    pip install tqdm pandas
    # 用于数据集生成
    pip install kor langchain dashscope python-dotenv
    ```

3.  **配置 API Key** (仅在需要重新生成数据集时需要)
    在 `generation_dataset/` 目录下创建一个 `.env` 文件，并填入你的阿里云 DashScope API Key。
    ```
    # filepath: /home/zlx/chat-wukong/generation_dataset/.env
    DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

## 执行流程

### 步骤一：下载预训练模型

运行 `model_download.py` 脚本，它将从 Hugging Face 或 ModelScope 下载 `Qwen2.5-7B-Instruct` 模型到 `model/` 目录下。

```bash
python model_download.py
```
> **注意**：脚本内置了国内镜像加速。如果下载失败，请检查网络或尝试脚本内的其他下载方案。

### 步骤二：生成微调数据集 (可选)

如果你想从自己的文本（如其他小说）生成数据集，请按以下步骤操作。项目已提供一份从《西游记》中提取的数据集，你可以跳过此步。

1.  将你的原始 `.txt` 文件放入 `dataset/input/wukong/` 目录。
2.  修改 `generation_dataset/main.py` 中的 `path` 变量，指向你的文本文件。
3.  运行脚本，它将调用大模型 API 抽取对话，并最终在 `dataset/train/lora/` 目录下生成 `xxx.json` 格式的训练文件。
    ```bash
    cd generation_dataset
    python main.py
    ```

### 步骤三：执行 LoRA 微调

运行 `train.py` 脚本开始训练。脚本会自动加载基础模型和处理好的数据集，进行 LoRA 微调。

```bash
python train.py
```
训练完成后，最终的 LoRA 适配器权重会被保存在 `output/qwen2.5-7B-instruct_lora/final` 目录下。

### 步骤四：与“悟空”对话

运行 `chat.py` 脚本，它会加载基础模型和训练好的 LoRA 适配器，然后你就可以在命令行中与扮演孙悟空的模型进行互动了。

```bash
python chat.py
```
你可以直接在 `chat.py` 文件中修改 `messages` 列表来测试不同的对话输入。

## 自定义配置

*   **微调参数**：可以在 `train.py` 中调整 `LoraConfig` (如 `r`, `lora_alpha`) 和 `TrainingArguments` (如 `learning_rate`, `num_train_epochs`) 来探索不同的训练效果。
*   **模型与数据**：所有脚本中的文件路径（如模型路径、数据路径）都以变量形式定义在脚本开头，方便根据你的实际情况进行修改。
*   **System Prompt**：在 `train.py` 和 `chat.py` 中，你可以修改 `system_prompt` 来改变模型的角色设定。请确保训练和推理时的人设一致。


