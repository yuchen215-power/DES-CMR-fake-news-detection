import os
import torch
import torch.nn as nn
from typing import List
from transformers import T5ForConditionalGeneration, T5Tokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class PromptGenerator(nn.Module):
    """
    使用完整的 T5ForConditionalGeneration，只取它的 shared embedding 及 encoder。
    """
    def __init__(
        self,
        prompt_dim: int,
        pretrained_model: str = "./t5-large",
        freeze_bert: bool = True,
        hidden_proj_dim: int = None,
    ):
        super().__init__()
        # 1) 加载完整的 T5 模型（共享 embedding + encoder + decoder）
        full_model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        # 2) 只保留 encoder 部分
        self.encoder = full_model.encoder
        # 3) 分词器也要用同一个 vocab
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model)

        if freeze_bert:
            for p in self.encoder.parameters():
                p.requires_grad = False

        d_model = full_model.config.d_model  # T5-Large 默认为 1024
        layers = []
        if hidden_proj_dim is not None:
            layers += [nn.Linear(d_model, hidden_proj_dim), nn.ReLU()]
            proj_in = hidden_proj_dim
        else:
            proj_in = d_model
        layers += [nn.Linear(proj_in, prompt_dim)]
        self.proj = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(prompt_dim)

    def forward(self, texts: List[str]) -> torch.Tensor:
        encoding = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )
        device = next(self.encoder.parameters()).device
        for k, v in encoding.items():
            encoding[k] = v.to(device)

        # 只跑 encoder，不会触发 decoder
        outputs = self.encoder(**encoding)
        cls_embed = outputs.last_hidden_state[:, 0, :]  # (B, d_model)
        prompt = self.proj(cls_embed)                  # (B, prompt_dim)
        return self.norm(prompt)

if __name__ == "__main__":
    model = PromptGenerator(prompt_dim=128, hidden_proj_dim=64)
    sample_texts = ["Hello world!", "今天天气不错。"]
    prompts = model(sample_texts)
    print(prompts.shape)  # torch.Size([2, 128])
    print(prompts)
