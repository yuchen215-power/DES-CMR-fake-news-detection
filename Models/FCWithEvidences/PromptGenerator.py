# prompt.py

import torch
import torch.nn as nn
from typing import List               # ← 新增
from transformers import AutoModel, AutoTokenizer
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _resolve_pretrained_model(pretrained_model: str):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = []
    if os.path.isabs(pretrained_model):
        candidates.append(pretrained_model)
    else:
        candidates.append(os.path.abspath(pretrained_model))
        candidates.append(os.path.join(repo_root, pretrained_model))

    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            return candidate, True
    return pretrained_model, False


class PromptGenerator(nn.Module):
    """
    动态提示向量生成器（Soft Prompt），基于预训练 BERT。
    将每条文本的 [CLS] 向量映射到指定的 prompt_dim。
    """

    def __init__(
        self,
        prompt_dim: int,
        pretrained_model: str = "bert-base-cased",
        freeze_bert: bool = True,
        hidden_proj_dim: int = None,
    ):
        super().__init__()
        model_source, local_files_only = _resolve_pretrained_model(pretrained_model)
        self.bert = AutoModel.from_pretrained(model_source, local_files_only=local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_files_only)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 通常是 768
        layers = []
        if hidden_proj_dim is not None:
            layers += [nn.Linear(hidden_size, hidden_proj_dim), nn.ReLU()]
            proj_in = hidden_proj_dim
        else:
            proj_in = hidden_size
        layers += [nn.Linear(proj_in, prompt_dim)]
        self.proj = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(prompt_dim)

    def forward(self, texts: List[str]) -> torch.Tensor:  # ← 改为 List[str]
        """
        Args:
            texts: List[str]，长度 = batch_size
        Returns:
            Tensor of shape (batch_size, prompt_dim)
        """
        # 1) Tokenize (CPU)
        encoding = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        # 1.1) 把 encoding 全部搬到和 BERT 同样的 device
        device = next(self.bert.parameters()).device
        for k, v in encoding.items():
            encoding[k] = v.to(device)
        # 2) BERT 编码
        outputs = self.bert(**encoding)
        cls_embed = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
        # 3) MLP 投影
        prompt = self.proj(cls_embed)                   # (B, prompt_dim)
        # 4) 归一化
        prompt = self.norm(prompt)
        return prompt

if __name__ == "__main__":
    # 测试代码
    model = PromptGenerator(prompt_dim=128, hidden_proj_dim=64)
    sample_texts = ["Hello world!"]
    prompts = model(sample_texts)
    print(prompts.shape)
    print(prompts)  # 打印生成的提示向量
