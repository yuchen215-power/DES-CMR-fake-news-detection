import torch
import torch.nn as nn

class DynamicEvidenceSelector(nn.Module):
    def __init__(self, embedding_dim, top_k=5, num_heads=4, temperature=1.0, router_dropout=0.1):
        super(DynamicEvidenceSelector, self).__init__()
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.temperature = max(float(temperature), 1e-6)
        self.query_encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.evidence_encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.router = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.GELU(),
            nn.Dropout(router_dropout),
            nn.Linear(embedding_dim, 1),
        )

    @staticmethod
    def _masked_mean(sequence, mask=None):
        sequence = sequence.permute(1, 0, 2)
        if mask is None:
            return sequence.mean(dim=1)

        mask = mask.to(device=sequence.device, dtype=sequence.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (sequence * mask).sum(dim=1) / denom

    @staticmethod
    def _masked_sparsemax(scores, mask=None):
        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.to(device=scores.device, dtype=torch.bool)

        valid_rows = mask.any(dim=1, keepdim=True)
        safe_scores = torch.where(mask, scores, torch.full_like(scores, torch.finfo(scores.dtype).min))
        max_scores = torch.where(
            valid_rows,
            safe_scores.max(dim=1, keepdim=True)[0],
            torch.zeros_like(scores[:, :1]),
        )
        safe_scores = torch.where(mask, safe_scores - max_scores, torch.zeros_like(safe_scores))

        sorted_scores, _ = torch.sort(safe_scores, dim=1, descending=True)
        steps = torch.arange(
            1,
            scores.size(1) + 1,
            device=scores.device,
            dtype=scores.dtype,
        ).view(1, -1)
        support = 1 + steps * sorted_scores > sorted_scores.cumsum(dim=1)
        support_size = support.sum(dim=1, keepdim=True).clamp(min=1)
        tau = (
            sorted_scores.cumsum(dim=1).gather(1, support_size - 1) - 1
        ) / support_size.to(scores.dtype)

        probs = torch.clamp(safe_scores - tau, min=0.0)
        probs = probs * mask.to(probs.dtype)
        normalizer = probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
        probs = torch.where(valid_rows, probs / normalizer, torch.zeros_like(probs))
        return probs

    def forward(
        self,
        query,
        evidences,
        evidence_mask=None,
        query_mask=None,
        evidence_token_mask=None,
        return_idx=False,
    ):
        query_emb = self.query_encoder(query)
        query_repr = self._masked_mean(query_emb, query_mask)

        scores = []
        routed_evidences = []
        for idx, evidence in enumerate(evidences):
            evidence_emb = self.evidence_encoder(evidence)
            token_mask = None
            if evidence_token_mask is not None:
                token_mask = evidence_token_mask[:, idx, :]
            evidence_repr = self._masked_mean(evidence_emb, token_mask)

            pair_repr = torch.cat(
                [
                    query_repr,
                    evidence_repr,
                    query_repr * evidence_repr,
                    torch.abs(query_repr - evidence_repr),
                ],
                dim=-1,
            )
            score = self.router(pair_repr).squeeze(-1) / self.temperature
            scores.append(score)
        scores = torch.stack(scores, dim=1)

        selector_weights = self._masked_sparsemax(scores, evidence_mask)
        if evidence_mask is not None:
            evidence_mask = evidence_mask.to(device=scores.device, dtype=torch.bool)
            empty_rows = selector_weights.sum(dim=1, keepdim=True) <= 0
            has_valid_rows = evidence_mask.any(dim=1, keepdim=True)
            if torch.any(empty_rows & has_valid_rows):
                fallback = evidence_mask.to(selector_weights.dtype)
                fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
                selector_weights = torch.where(empty_rows & has_valid_rows, fallback, selector_weights)

        ranking_scores = scores
        if evidence_mask is not None:
            ranking_scores = scores.masked_fill(~evidence_mask, torch.finfo(scores.dtype).min)
        k = min(self.top_k, scores.size(1))
        top_scores, top_indices = torch.topk(ranking_scores, k, dim=1)

        if return_idx:
            return selector_weights, top_indices, top_scores

        for idx, evidence in enumerate(evidences):
            routed_evidences.append(evidence * selector_weights[:, idx].view(1, -1, 1))
        return routed_evidences
