import time
import torch.nn as nn
import torch
import torch.nn.functional as F

from Models.FCWithEvidences.basic_fc_model import BasicFCModel
import torch_utils
from setting_keywords import KeyWordSettings
from Models.BiDAF.wrapper import GGNN, GGNN_with_GSL, GGNN2_with_GSL, Linear
from thirdparty.two_branches_attention import *
import numpy as np
import torch_utils as my_utils
from Models.FCWithEvidences.PromptGenerator import PromptGenerator
from Models.FCWithEvidences.DynamicEvidenceSelector import DynamicEvidenceSelector
torch.set_printoptions(profile="full")

class GETRAL(BasicFCModel):
    """ Hierarchical Multi-Head Attention Network for Fact-Checking (MAC)"""

    def __init__(self, params):
        super(GETRAL, self).__init__(params)
        self._params = params
        self.embedding = self._make_default_embedding_layer(params)
        self.num_classes = self._params["num_classes"]
        self.fixed_length_right = self._params["fixed_length_right"]
        self.fixed_length_left = self._params["fixed_length_left"]
        self.use_claim_source = self._params["use_claim_source"]
        self.use_article_source = self._params["use_article_source"]
        self._use_cuda = self._params["cuda"]
        # number of attention heads
        self.num_att_heads_for_words = self._params["num_att_heads_for_words"]
        self.num_att_heads_for_evds = self._params["num_att_heads_for_evds"]
        self.dropout_gnn = self._params['dropout_gnn']
        self.dropout_left = self._params["dropout_left"]
        self.dropout_right = self._params["dropout_right"]
        self.hidden_size = self._params["hidden_size"]
        self.output_size = self._params['output_size']
        self.gsl_rate = self._params["gsl_rate"]
        self.n_kernel = self._params["n_kernel"]
        self.alpha = self._params["alpha"]
        self.tau = self._params["tau"]
        self.topk = self._params["topk"]
        self.prompt_dim = self._params["prompt_dim"]
        if self.use_claim_source:
            self.claim_source_embs = self._make_entity_embedding_layer(
                self._params["claim_source_embeddings"], freeze=False)  # trainable
            self.claim_emb_size = self._params["claim_source_embeddings"].shape[1]

        if self.use_article_source:
            self.article_source_embs = self._make_entity_embedding_layer(
                self._params["article_source_embeddings"], freeze=False)  # trainable
            self.article_emb_size = self._params["article_source_embeddings"].shape[1]
            source_hidden_size = max(32, self.article_emb_size // 2)
            self.source_credibility = nn.Sequential(
                nn.Linear(self.article_emb_size, source_hidden_size),
                nn.Tanh(),
                nn.Linear(source_hidden_size, self.num_att_heads_for_evds),
            )
            self.source_prior_scale = 0.1
        D = self._params["embedding_output_dim"]
        self.prompt_gen = PromptGenerator(self.prompt_dim,hidden_proj_dim=64, freeze_bert=False)
        self.prompt_proj = nn.Linear(D + self.prompt_dim,
                                     D)
        # 动态证据选择
        self.evd_selector = DynamicEvidenceSelector(
            embedding_dim=D,
            top_k=self.topk,
            num_heads=self.num_att_heads_for_evds,
        )
        # Graph Gated Neural Network with structural learning
        self.ggnn4claim_1 = GGNN(in_features=D, out_features=self.hidden_size)
        
        # self.ggnn_with_gsl = GGNN_with_GSL(input_dim=D, hidden_dim=self.hidden_size, output_dim=self.hidden_size, rate=self.gsl_rate, dropout=self.dropout_gnn)
        self.ggnn_with_gsl2 = GGNN2_with_GSL(input_dim=D, hidden_dim=self.hidden_size, 
                                    output_dim=self.hidden_size, rate=self.gsl_rate, dropout=self.dropout_gnn,
                                    n_kernel=self.n_kernel, alpha=self.alpha)
        self.trans = Linear(2*self.hidden_size, self.hidden_size)

        # 证据级注意力熵正则化，避免单一证据过度主导
        self.entrop_lambda = 0.01
        self.router_lambda = 0.005

        # mapping query vector + claim's source vector if possible. Experiments show that without using claims'
        # src, Politifact dataset has lower performance
        dim = self.hidden_size # the dimension of the output of representation models (e.g., ggnn, bilstm)
        claim_size = dim
        if self.use_claim_source: 
            claim_size += self.claim_emb_size
        self.evidence_feature_size = dim * self.num_att_heads_for_words
        if self.use_article_source:
            self.evidence_feature_size += self.article_emb_size
        evd_size = self.evidence_feature_size * self.num_att_heads_for_evds
        input_size = claim_size + evd_size

        self._get_word_attention_func(dim=dim)
        self._get_evd_attention_func(dim=dim)
        self.evidence_relation_heads = 4
        while self.evidence_feature_size % self.evidence_relation_heads != 0 and self.evidence_relation_heads > 1:
            self.evidence_relation_heads -= 1
        self.evidence_relation_attn = nn.MultiheadAttention(
            embed_dim=self.evidence_feature_size,
            num_heads=self.evidence_relation_heads,
            dropout=self.dropout_right,
            batch_first=True,
        )
        self.evidence_relation_norm1 = nn.LayerNorm(self.evidence_feature_size)
        self.evidence_relation_norm2 = nn.LayerNorm(self.evidence_feature_size)
        self.evidence_relation_dropout = nn.Dropout(self.dropout_right)
        self.evidence_relation_ffn = nn.Sequential(
            nn.Linear(self.evidence_feature_size, self.evidence_feature_size),
            nn.GELU(),
            nn.Dropout(self.dropout_right),
            nn.Linear(self.evidence_feature_size, self.evidence_feature_size),
        )
        self.evidence_relation_gate = nn.Linear(self.evidence_feature_size * 2, self.evidence_feature_size)

        self.out = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.output_size)
        )
        self.evidence_relation_ffn.apply(torch_utils.init_weights)
        self.evidence_relation_gate.apply(torch_utils.init_weights)
        nn.init.zeros_(self.evidence_relation_attn.out_proj.weight)
        if self.evidence_relation_attn.out_proj.bias is not None:
            nn.init.zeros_(self.evidence_relation_attn.out_proj.bias)
        nn.init.zeros_(self.evidence_relation_ffn[-1].weight)
        nn.init.zeros_(self.evidence_relation_ffn[-1].bias)
        nn.init.zeros_(self.evidence_relation_gate.weight)
        nn.init.constant_(self.evidence_relation_gate.bias, -4.0)
        if self.use_article_source:
            self.source_credibility.apply(torch_utils.init_weights)
            nn.init.zeros_(self.source_credibility[-1].weight)
            nn.init.zeros_(self.source_credibility[-1].bias)
        self.out[0].apply(torch_utils.init_weights)
        self.out[1].apply(torch_utils.init_weights)

    def _pad_right_tensor2(self, tsr: torch.Tensor, **kargs):
        fixed_n = kargs[KeyWordSettings.FIXED_NUM_EVIDENCES]  # int
        # EvidenceCountPerQuery 是 Tensor(shape=(B,)), 记录每个 query 有多少条证据
        evd_counts = kargs[KeyWordSettings.EvidenceCountPerQuery]  # Tensor of shape (B,)
        B = evd_counts.size(0)
        n = fixed_n
        # tsr is (sum_i evd_counts[i], R, D)
        R, D = tsr.size(1), tsr.size(2)

        padded = tsr.new_zeros((B, n, R, D))
        ptr = 0
        for i, cnt in enumerate(evd_counts.tolist()):
            padded[i, :cnt] = tsr[ptr:ptr+cnt]
            ptr += cnt
        return padded

    
    def forward(self, query: torch.Tensor, document: torch.Tensor, verbose=False, **kargs):
        """
        query and document have shaped as described. Each query is assumed to have `n = 30` evidences. If a query has
        less than 30 evidences, I pad them with all zeros. The length of all-zeros evidence is 0. However, PyTorch
        does not allow empty sequences input to RNN. Therefore, I have to use
        `kargs[KeyWordSettings.QueryContentNoPaddingEvidence]` and `kargs[KeyWordSettings.DocContentNoPaddingEvidence]`
        with shape (n1 + n2 + ... + nx, L) and (n1 + n2 + ... + nx, R) respectively.
        Parameters
        ----------
        query: `torch.Tensor`  (B, L)
        document: `torch.Tensor` (B, n = 30, R)
        """
        prompt_vecs = self.prompt_gen(kargs["claim_texts"])  # (B, prompt_dim)
        embed_q = self.embedding(query.long())  # (B, L, D)
        B, L, D = embed_q.shape
        p_expanded = prompt_vecs.unsqueeze(1).expand(B, L, -1)  # (B, L, prompt_dim)
        cat = torch.cat([p_expanded, embed_q], dim=-1)  # (B, L, D + prompt_dim)
        embed_q = self.prompt_proj(cat)  # (B, L, D)
        kargs["enhanced_query_embed"] = embed_q  # (B, L, D)
        assert KeyWordSettings.Query_lens in kargs and KeyWordSettings.Doc_lens in kargs
        _, L = query.size()
        D = self._params["embedding_output_dim"]
        assert query.size(0) == document.size(0)
        batch_size, n, R = document.size()  # batch_size = 32 which is real batch_size of each of mini-batches
        assert n == 30
        # for documents
        d_new_indices, d_restoring_indices, d_lens = kargs[KeyWordSettings.DocLensIndices]
        assert KeyWordSettings.DocContentNoPaddingEvidence in kargs
        doc = kargs[KeyWordSettings.DocContentNoPaddingEvidence]  # (n1 + n2 + n3 + .. n_b, R)
        doc_mask = (doc >= 1)  # (B1, R) 0 is for padding word
        doc_adj = kargs[KeyWordSettings.Evd_Docs_Adj].float()  # (n1 + n2 + n3 + .. n_b, R, R)
        embed_doc = self.embedding(doc.long())  # (n1 + n2 + n3 + .. n_b, R, D)
        assert d_lens.shape[0] == embed_doc.size(0)

        # ggnn for query
        query_repr_word, query_repr = self._generate_query_repr_gnn(query, **kargs)  # output's shape is always (B1, self.hidden_size)
        
        # ggnn for doc
        query_mask = self._pad_left_tensor((query > 0), **kargs)        # (n1+...+nk, 30)
        query_repr_word = self._pad_left_tensor2(query_repr_word, **kargs)  # (n1+...+nk, 30, 300)

        # doc_out_ggnn = self.ggnn_with_gsl(doc_adj, embed_doc)     
        doc_out_ggnn = self.ggnn_with_gsl2(doc_adj, embed_doc, query_repr_word,doc_mask, query_mask)

        # 1) 恢复成 (B, n, R, D)
        padded_doc_out = self._pad_right_tensor2(doc_out_ggnn, **kargs)  # (B, n, R, D)

        # 2) Flatten and dynamically select evidences
        B, n, R, D = padded_doc_out.size()
        query_trans = embed_q.permute(1, 0, 2)                         # (L, B, D)
        evidences = [padded_doc_out[:, i].permute(1, 0, 2) for i in range(n)]
        real_evidence_mask = (document.sum(dim=-1) >= 1)
        routing_weights, _, _ = self.evd_selector(
            query_trans,
            evidences,
            evidence_mask=real_evidence_mask,
            query_mask=(query > 0),
            evidence_token_mask=(document > 0),
            return_idx=True,
        )  # (B, n)

        # Step 1: word-level attention
        doc_avg, word_att_weights = self._word_level_attention(left_tsr=query_repr, right_tsr=doc_out_ggnn,
                                                           right_mask=doc_mask, **kargs)
        # Step 2: evidence-level attention. We will override this function in sub-classes
        if self.use_claim_source:
            query_source_idx = kargs[KeyWordSettings.QuerySources]
            claim_embs = self.claim_source_embs(query_source_idx.long())  # (B, 1, D)
            claim_embs = claim_embs.squeeze(1)  # (B, D)
            claim_embs = self._pad_left_tensor(claim_embs, **kargs)
            query_repr = torch.cat([claim_embs, query_repr], dim=-1)  # (B, 2D + D)
        evd_avg, evd_att_weight = self._evidence_level_attention_new(
            query_repr,
            doc_avg,
            document,
            routing_weights=routing_weights,
            **kargs
        )
        att = torch.nan_to_num(evd_att_weight, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(1e-9)
        ent = -(att * att.log()).sum(dim=1).mean()
        routing = torch.nan_to_num(routing_weights, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(1e-9)
        routing_ent = -(routing * routing.log()).sum(dim=1).mean()
        aux_losses = {
            "entropy_loss": self.entrop_lambda * ent,
            "routing_entropy_loss": self.router_lambda * routing_ent,
        }
        
        output = self._get_final_repr(left_tsr=query_repr, right_tsr=evd_avg, **kargs)
        phi = self.out(output)  # (B, )
        if kargs.get(KeyWordSettings.OutputRankingKey, False): 
            return phi, (word_att_weights, evd_att_weight)
        if kargs.get(KeyWordSettings.OutputRepr, False):
            # output representation of evidence and claim+evidence
            label = kargs["labels"]

            noise_mask = self._generate_noise_mask(evd_att_weight, **kargs)                   # only perturbate the most important evidence
            noise_doc_avg = self._generate_adversarial_instance(phi, doc_avg, label, mask=noise_mask, noise_norm=1.5)            # local evidence perturbation
            noise_evd_avg, _ = self._evidence_level_attention_new(
                query_repr,
                noise_doc_avg,
                document,
                routing_weights=routing_weights,
                **kargs
            )
            noise_output2 = self._get_final_repr(left_tsr=query_repr, right_tsr=noise_evd_avg, **kargs)
            noise_phi2 = self.out(noise_output2)

            return phi, output, noise_phi2, noise_output2, aux_losses

        return phi

    def _generate_query_repr(self, query: torch.Tensor, **kargs):
        q_new_indices, q_restoring_indices, q_lens = kargs[KeyWordSettings.QueryLensIndices]
        query_mask = (query > 0).unsqueeze(2)  # (B, L, 1)
        query_lens = kargs[KeyWordSettings.Query_lens]  # (B, )
        query_lens = query_lens.unsqueeze(-1)  # (B, 1)

        # embed_query = self.embedding(query.long())  # (B, L, D)
        embed_query = kargs.get("enhanced_query_embed", self.embedding(query.long()))  # (B, L, D)
        # bilstm for query
        query_gru_hiddens = torch_utils.auto_rnn(self.query_bilstm, input_feats=embed_query, lens=q_lens,
                                                 new_indices=q_new_indices, restoring_indices=q_restoring_indices,
                                                 max_len=self.fixed_length_left)  # (B, L, 2*D)
        query_repr = torch.sum(query_gru_hiddens * query_mask.float(), dim=1) / query_lens.float()  # (B, D)

        query_repr = self._pad_left_tensor(query_repr, **kargs)  # (n1 + n2 + n3 + .. + nx, H)
        return query_repr

    def _generate_query_repr_gnn(self, query: torch.Tensor, **kargs):
        query_mask = (query > 0).unsqueeze(2)  # (B, L, 1)
        query_lens = kargs[KeyWordSettings.Query_lens]  # (B, )
        query_lens = query_lens.unsqueeze(-1)  # (B, 1)

        adj = kargs[KeyWordSettings.Query_Adj].float()  # (B, L, L)
        # embed_query = self.embedding(query.long())  # (B, L, D)
        embed_query = kargs.get("enhanced_query_embed", self.embedding(query.long()))  # (B, L, D)
        query_gnn_hiddens = self.ggnn4claim_1(adj, embed_query)

        query_repr = torch.sum(query_gnn_hiddens * query_mask.float(), dim=1) / query_lens.float()  # (B,2*D)
        query_repr = self._pad_left_tensor(query_repr, **kargs)  # (n1 + n2 + n3 + .. + nx, H)
        return query_gnn_hiddens, query_repr            #  repr after ggnn

    def _get_article_source_embeddings(self, **kargs):
        doc_source_idx = kargs[KeyWordSettings.DocSources]  # (B, n = 30)
        mask = (doc_source_idx == -1)
        # when doc_src has negative values, the exception will be thrown.
        doc_source_idx = doc_source_idx.masked_fill(mask, 0)
        article_embs = self.article_source_embs(doc_source_idx.long())  # (B, n, D)
        return article_embs

    def _use_article_embeddings(self, article_repr: torch.Tensor, **kargs):
        """
        Using article embeddings with articles' representations
        Parameters
        ----------
        article_repr: `torch.Tensor` (B, n, H)
        kargs
        """
        article_embs = self._get_article_source_embeddings(**kargs)
        article_repr = torch.cat([article_repr, article_embs], dim=-1)  # (B, n, 2D + D)
        return article_repr

    def _compute_source_credibility_prior(self, article_embs: torch.Tensor, mask: torch.Tensor):
        prior = self.source_credibility(article_embs)
        prior = self.source_prior_scale * torch.tanh(prior)
        prior = prior * mask.unsqueeze(-1).to(prior.dtype)
        return prior

    def _refine_evidence_relations(self, evidence_repr: torch.Tensor, mask: torch.Tensor):
        valid_mask = mask.unsqueeze(-1).to(evidence_repr.dtype)
        normalized = self.evidence_relation_norm1(evidence_repr)
        attn_out, _ = self.evidence_relation_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~mask,
        )
        attn_out = self.evidence_relation_dropout(attn_out) * valid_mask
        context = evidence_repr + attn_out

        ffn_input = self.evidence_relation_norm2(context)
        ffn_out = self.evidence_relation_ffn(ffn_input) * valid_mask
        context = context + ffn_out

        gate = torch.sigmoid(self.evidence_relation_gate(torch.cat([evidence_repr, context], dim=-1)))
        gate = gate * valid_mask
        refined = evidence_repr + gate * (context - evidence_repr)
        refined = refined * valid_mask
        return refined

    def _word_level_attention(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor, right_mask: torch.Tensor, **kargs):
        """
            Compute word-level attention of evidences.
        Parameters
        ----------
        left_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, H). It represents claims' representation
        right_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, R, H). Doc's representations.
        right_mask: `torch.Tensor` (n1 + n2 + ... + nx, R)
        kargs
        Returns
        -------
            Representations of each of evidences of each of claim in the mini-batch of shape (B1, X)
        """
        # for reproducing results in the report
        B1, R, H = right_tsr.size() # [n1+n2..., 100, 300]
        assert left_tsr.size(0) == B1 and len(left_tsr.size()) == 2
        # new_left_tsr = left_tsr.unsqueeze(1).expand(B1, R, -1)
        avg, att_weight = self.self_att_word(left_tsr, right_tsr, right_mask)
        avg = torch.flatten(avg, start_dim=1)  # (n1 + n2 + n3 + ... + nx, n_head * 4D)
        # avg = torch.cat([left_tsr, avg], dim=-1)  # (B1, 2D + D)
        return avg, att_weight  # (n1 + n2 + n3 + ... + nx, R)

    def _routing_aware_evidence_attention(
        self,
        left_tsr: torch.Tensor,
        right_tsr: torch.Tensor,
        mask: torch.Tensor,
        routing_weights: torch.Tensor = None,
        source_prior: torch.Tensor = None,
    ):
        B, L, _ = right_tsr.size()
        left_tmp = left_tsr.unsqueeze(1).expand(B, L, -1)
        tsr = torch.cat([left_tmp, right_tsr], dim=-1)
        tmp = torch.tanh(self.self_att_evd.linear1(tsr))
        linear_out = self.self_att_evd.linear2(tmp)

        if routing_weights is not None:
            prior = torch.log(routing_weights.clamp_min(1e-8)).unsqueeze(-1)
            linear_out = linear_out + prior
        if source_prior is not None:
            linear_out = linear_out + source_prior

        doc_mask = (~mask).unsqueeze(-1).expand(B, L, self.num_att_heads_for_evds)
        linear_out = linear_out.masked_fill(doc_mask, -np.inf)
        attention_weights = F.softmax(linear_out, dim=1)
        attended = torch.bmm(right_tsr.permute(0, 2, 1), attention_weights)
        return attended, attention_weights

    def _evidence_level_attention_new(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor,
                                      full_padded_document: torch.Tensor, routing_weights: torch.Tensor = None, **kargs):
        """
        compute evidence-level attention
        Parameters
        ----------
        left_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, D)
        right_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, D)
        full_padded_document: `torch.Tensor` (B, R). Note, B != (n1 + n2 + ... + nx)

        Returns
        -------
            a tensor of shape (B, _) which stands for representation of `batch_size = B` claims in each of mini-batches
        """
        # for reproducing results in the report
        # if self.evd_attention_type != AttentionType.ConcatNotEqual: left_tsr = self.map_query_level2(left_tsr)
        new_left_tsr = self._pad_right_tensor(left_tsr, **kargs)
        new_left = new_left_tsr[:, 0, :]  # (B, X)

        padded_avg = self._pad_right_tensor(right_tsr, **kargs)
        mask = (torch.sum(full_padded_document, dim=-1) >= 1)  # (B, n)
        source_prior = None
        if self.use_article_source:
            article_embs = self._get_article_source_embeddings(**kargs)
            source_prior = self._compute_source_credibility_prior(article_embs, mask)
            padded_avg = self._use_article_embeddings(padded_avg, **kargs)
        padded_avg = self._refine_evidence_relations(padded_avg, mask)

        if routing_weights is not None:
            routing_weights = routing_weights.to(device=padded_avg.device, dtype=padded_avg.dtype)
            routing_weights = routing_weights * mask.to(routing_weights.dtype)

        attended_avg, att_weight = self._routing_aware_evidence_attention(
            new_left,
            padded_avg,
            mask,
            routing_weights=routing_weights,
            source_prior=source_prior,
        )
        avg = torch.flatten(attended_avg, start_dim=1)  # (B, num_heads * 2D)
        return avg, att_weight

    def _get_word_attention_func(self, dim: int):
        """
        get the function to compute attention weights on word.
        Parameters
        ----------
        dim: `int` the last dimension of an input of attention func
        """
        input_dim = 2 * dim
        self.self_att_word = ConcatNotEqualSelfAtt(inp_dim=input_dim, out_dim=dim,
                                                   num_heads=self.num_att_heads_for_words)
        # else:
        #     raise NotImplemented("Unknown attention type for words")

    def _get_evd_attention_func(self, dim: int):
        """
        get the function to compute attention weights on evidence.
        Parameters
        ----------
        dim: `int` the last dimension of an input of attention func
        """
        # the first is for claim, the second is for word att on evds
        input_dim = dim + self.num_att_heads_for_words * dim
        if self.use_claim_source: input_dim += self.claim_emb_size
        if self.use_article_source: input_dim += self.article_emb_size
        self.self_att_evd = ConcatNotEqualSelfAtt(inp_dim=input_dim, out_dim=dim, num_heads=self.num_att_heads_for_evds)
        # else:
        #     raise NotImplemented("Unknown attention type for evidences")

    def _get_final_repr(self, left_tsr: torch.Tensor, right_tsr: torch.Tensor, **kargs):
        """
        get final representaion of
        Parameters
        ----------
        left_tsr: `torch.Tensor` of shape (n1 + n2 + ... + nx, X) (query features and query's claims features)
        right_tsr: `torch.Tensor` of shape (B, Y) (document features (text + src))
        kargs

        Returns
        -------

        """
        new_left_tsr = self._pad_right_tensor(left_tsr, **kargs)
        new_left = new_left_tsr[:, 0, :]  # (B, X)
        tmp = torch.cat([new_left, right_tsr], dim=-1)
        return tmp

    def predict(self, query: torch.Tensor, doc: torch.Tensor, verbose: bool = False, **kargs) -> np.ndarray:
        """ query.shape = (B, L), doc.shape = (B, R) """
        self.train(False)  # very important, to disable dropout
        assert query.size(0) == doc.size(0)
        probs = self(query, doc, **kargs)  # (1, )  it is not softmax yet, how to check?
        return probs
    
    def proj(self, repr: torch.Tensor):
        return F.normalize(repr, dim=-2)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)                # (batch, dim)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def contrastive_loss(self, repr_1: torch.Tensor, repr_2: torch.Tensor):
        h1 = self.semi_linear(repr_1)           # need reduced dimension
        h2 = self.semi_linear(repr_2)

        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        
        return ret

    def _generate_adversarial_instance(self, pred, repr, label, mask=None, noise_norm=1.5):
        ce_loss = F.cross_entropy(pred, label)
        ce_loss.detach()                # 从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad

        repr.retain_grad()              # enable .grad attribute for non-leaf Tensors
        ce_loss.backward(retain_graph=True)
        unnormalized_noise = repr.grad.detach_()     # detach a variable from graph, set it as leaf Tensor   

        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        norm = unnormalized_noise.norm(p=2, dim=-1)
        normalized_noise = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-10)
        if mask is not None:
            normalized_noise = normalized_noise * mask.unsqueeze(-1)
        
        noise_repr = repr + noise_norm * normalized_noise
        return noise_repr

    def _generate_noise_mask(self, evd_att_weight, **kargs):
        """
        evd_att_weight: (B, L, Head)
        """
        
        evd_count_per_query = kargs[KeyWordSettings.EvidenceCountPerQuery]
        evd_num = int(my_utils.cpu(torch.sum(evd_count_per_query)))
        mask = my_utils.gpu(torch.zeros(evd_num), self._use_cuda)
        evd_att_weight = torch.sum(evd_att_weight, dim=2)           # (B, L)
        value, indices = evd_att_weight.topk(1, dim=1)

        start = 0
        for i, (cnt, index) in enumerate(zip(evd_count_per_query, indices)):
            mask[start + index] = 1
            start += cnt

        return mask
