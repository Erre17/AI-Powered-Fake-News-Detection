import re
import string
import warnings

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from plotly.subplots import make_subplots
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

# ── Example presets ────────────────────────────────────────────────────────────
EXAMPLES = [
    {
        "label": "🟢 Real — Senate Vote",
        "title": "Senate passes bipartisan infrastructure bill after weeks of negotiations",
        "body": (
            "The United States Senate passed a $1.2 trillion infrastructure bill on Tuesday "
            "with support from both parties, marking one of the largest investments in roads, "
            "bridges, and broadband in decades. The legislation cleared the chamber 69–30, "
            "with 19 Republican senators joining the Democratic majority. President Biden "
            "praised the outcome as proof that bipartisan governance is still possible, while "
            "critics on the left argued the bill fell short on climate provisions. The measure "
            "now heads to the House, where progressive lawmakers have signaled reservations."
        ),
    },
    {
        "label": "🟢 Real — Foreign Policy",
        "title": "NATO allies agree to increase defense spending amid rising geopolitical tensions",
        "body": (
            "Leaders of NATO member states reached an agreement at this week's summit to raise "
            "minimum defense spending targets from 2% to 2.5% of GDP, responding to escalating "
            "security concerns on the alliance's eastern flank. The accord, signed by all 32 "
            "members, also includes new commitments on cybersecurity and joint military exercises. "
            "Secretary General Jens Stoltenberg called it a historic step toward collective "
            "readiness. Several European nations had previously struggled to meet the existing "
            "threshold, and the new target is expected to take effect by 2030."
        ),
    },
    {
        "label": "🟢 Real — Domestic Policy",
        "title": "Supreme Court rules 6–3 to uphold federal agency rulemaking authority",
        "body": (
            "The Supreme Court issued a landmark ruling on Thursday preserving the power of "
            "federal agencies to issue regulations within their congressionally delegated mandates, "
            "rejecting a challenge that sought to dramatically curtail the administrative state. "
            "Writing for the majority, Justice Sonia Sotomayor held that agencies possess "
            "reasonable latitude to interpret ambiguous statutes within their domain of expertise. "
            "The three dissenting justices argued the decision undermined the constitutional "
            "separation of powers. Legal scholars say the ruling has broad implications for "
            "environmental, financial, and labor regulations."
        ),
    },
    {
        "label": "🔴 Fake — Deep State Plot",
        "title": "BOMBSHELL: Secret deep state cabal caught rigging congressional elections, insider reveals",
        "body": (
            "A courageous whistleblower from inside the Washington establishment has come forward "
            "with devastating proof that a shadowy network of unelected bureaucrats has been "
            "systematically stealing elections from true patriot candidates for over a decade. "
            "Leaked memos, which mainstream media refuses to cover, show explicit coordination "
            "between intelligence agencies and party operatives to suppress conservative votes. "
            "The traitors pulling the strings are terrified this information is getting out. "
            "Share this everywhere before the deep state censors scrub it from the internet. "
            "The entire political class is complicit and the people deserve the truth NOW."
        ),
    },
    {
        "label": "🔴 Fake — Foreign Takeover",
        "title": "EXCLUSIVE: Foreign government secretly controls top U.S. senators through blackmail network",
        "body": (
            "Stunning revelations from an anonymous intelligence source confirm that a hostile "
            "foreign power has compromised dozens of senior U.S. senators using a sophisticated "
            "blackmail operation run out of overseas embassies. The source claims encrypted "
            "financial transfers prove that legislative votes on key national security bills "
            "were sold to foreign handlers. The globalist-controlled press is furiously burying "
            "this explosive report to protect their puppet politicians. Patriots who have seen "
            "the evidence say it proves America's sovereignty is being sold from within. "
            "Your government is not working for you — wake up before it is too late."
        ),
    },
    {
        "label": "🔴 Fake — Martial Law Hoax",
        "title": "Government quietly prepares secret martial law camps to imprison political dissidents",
        "body": (
            "Documents allegedly obtained from a federal contractor reveal that the government "
            "has been constructing a network of detention facilities in remote locations, intended "
            "to hold political opponents when martial law is declared following the next election. "
            "An anonymous military source claims the order has already been signed and that "
            "dissident lists containing millions of names are being compiled by surveillance "
            "programs. The controlled media won't touch this story because they are part of the "
            "plan. Trusted insiders say the timeline is accelerating and that the window to "
            "resist the authoritarian takeover is rapidly closing. Spread the word immediately."
        ),
    },
    {
    "label": "🔴 Fake — Racial Replacement Conspiracy",
    "title": "EXPOSED: Government running secret program to deliberately replace white Americans with imported voters",
    "body": (
        "A leaked internal memo from a federal immigration office allegedly confirms what "
        "patriots have long suspected: a coordinated, decade-long program to flood swing "
        "states with non-white immigrants specifically to dilute the voting power of "
        "native-born Americans. The anonymous whistleblower claims top officials openly "
        "discussed demographic 'reengineering' in closed-door meetings. Globalist billionaires "
        "are said to be bankrolling the operation through NGOs with direct government access. "
        "The mainstream media is suppressing this story to protect the anti-white agenda "
        "being carried out in plain sight. True Americans are being replaced and their "
        "silence is exactly what the elites are counting on. Share before this gets censored."
    ),
},
]


import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionModule(nn.Module):
    """
        Self-Attention Module for sequence/point-cloud data using 1D convolutions.

        This module implements a self-attention mechanism that captures long-range
        dependencies across the input sequence. It projects the input into query,
        key, and value spaces using 1D convolutions, computes scaled dot-product
        attention, and blends the attended output with the original input via a
        learnable residual scale parameter (gamma).
    """
    def __init__(self, in_dim):
        super(SelfAttentionModule, self).__init__()

        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv1d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, 1)

        self.gamma = nn.Parameter(torch.full((1,), 0.1))
        self.scale = (in_dim // 8) ** -0.5
    def forward(self, x, return_attention=False):
        B, C, N = x.size()
        self.attn_drop = nn.Dropout(p=0.1)
        query = self.query_conv(x).view(B, -1, N).permute(0,2,1)
        key = self.key_conv(x).view(B, -1, N)

        energy = torch.bmm(query, key) * self.scale
        attention = self.attn_drop(F.softmax(energy, dim=-1))

        value = self.value_conv(x).view(B, -1, N)

        out = torch.bmm(value, attention.permute(0,2,1))
        out = self.gamma * out + x

        if return_attention:
            return out, attention

        return out


def squash(inputs, axis=-1):
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs


class CapsuleLayer(nn.Module):
    def __init__(
        self, num_capsules, num_route_nodes, in_channels, out_channels, num_iterations=3
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        self.num_route_nodes = num_route_nodes
        self.W = nn.Parameter(
            torch.randn(num_capsules, num_route_nodes, in_channels, out_channels)
        )

    def forward(self, x):
        x_exp = x[:, None, :, :, None]
        W_exp = self.W[None, :, :, :, :]
        u_hat = torch.matmul(W_exp.transpose(-1, -2), x_exp).squeeze(-1)
        b_ij = torch.zeros(x.shape[0], self.num_capsules, x.shape[2], device=x.device)
        for i in range(self.num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            s_j = (c_ij[:, :, :, None] * u_hat).sum(dim=2, keepdim=True)
            v_j = squash(s_j, axis=-1)
            if i < self.num_iterations - 1:
                agreement = (u_hat * v_j).sum(dim=-1)
                b_ij = b_ij + agreement
        return v_j.squeeze(2)


class FakeNewsCapsNet(nn.Module):
    def __init__(
        self,
        num_classes,
        bert_model_name="distilbert-base-uncased",
        max_seq_len=64,
        token_dim=128,
        dropout=0.3,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.bert = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        bert_hidden = self.bert.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(bert_hidden, token_dim),
            nn.LayerNorm(token_dim),
            nn.Dropout(dropout),
        )
        self.attention = SelfAttentionModule(token_dim)
        self.primary_caps = nn.ModuleList(
            [nn.Conv1d(token_dim, 32, kernel_size=1) for _ in range(12)]
        )
        self.routing_layer = CapsuleLayer(
            num_capsules=num_classes,
            num_route_nodes=32 * max_seq_len,
            in_channels=12,
            out_channels=16,
        )

    def forward(self, input_ids, attention_mask, return_attention=False):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state
        B, S, H = x.shape
        if S < self.max_seq_len:
            pad = torch.zeros(B, self.max_seq_len - S, H, device=x.device)
            x = torch.cat([x, pad], dim=1)
        else:
            x = x[:, : self.max_seq_len, :]
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        x, attention = self.attention(x, return_attention=True)
        u = [caps(x) for caps in self.primary_caps]
        u = torch.stack(u, dim=-1)
        u = u.view(B, -1, 12)
        u = squash(u)
        v = self.routing_layer(u)
        probs = torch.norm(v, p=2, dim=-1)
        if return_attention:
            return probs, attention
        return probs


try:
    from nltk.corpus import stopwords

    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set()


def _is_meaningful_token(token: str) -> bool:
    token = token.replace("##", "").strip(string.punctuation)
    if len(token) < 3:
        return False
    if token.lower() in STOPWORDS:
        return False
    if not re.match(r"^[a-zA-Z]+$", token):
        return False
    return True


def _merge_subwords(tokens, indices):
    merged_tokens, merged_indices = [], []
    current_token, current_indices = "", []
    for t, idx in zip(tokens, indices):
        if t.startswith("##"):
            current_token += t[2:]
            current_indices.append(idx)
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_indices.append(current_indices)
            current_token = t
            current_indices = [idx]
    if current_token:
        merged_tokens.append(current_token)
        merged_indices.append(current_indices)
    return merged_tokens, merged_indices


class _Hook:
    def __init__(self):
        self.value = None
        self._handle = None

    def register(self, module: nn.Module) -> None:
        def _hook_fn(m, inp, output):
            self.value = output.detach()

        self._handle = module.register_forward_hook(_hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class InstrumentedCapsuleLayer(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self._orig = original_layer
        self.last_c = None

    def forward(self, x):
        B = x.shape[0]
        n_caps = self._orig.num_capsules
        n_routes = self._orig.num_route_nodes
        x_exp = x[:, None, :, :, None]
        W_exp = self._orig.W[None, :, :, :, :]
        u_hat = torch.matmul(W_exp.transpose(-1, -2), x_exp).squeeze(-1)
        b_ij = torch.zeros(B, n_caps, n_routes, device=x.device)
        for i in range(self._orig.num_iterations):
            c_ij = torch.softmax(b_ij, dim=1)
            s_j = (c_ij[:, :, :, None] * u_hat).sum(dim=2, keepdim=True)
            v_j = self._squash(s_j)
            if i < self._orig.num_iterations - 1:
                agreement = (u_hat * v_j).sum(dim=-1)
                b_ij = b_ij + agreement
        self.last_c = c_ij.detach().cpu().permute(0, 2, 1)
        return v_j.squeeze(2)

    @staticmethod
    def _squash(x):
        n2 = (x**2).sum(dim=-1, keepdim=True)
        return (n2 / (1 + n2)) * (x / (n2.sqrt() + 1e-8))


class CapsNetExplainer:
    """
    Wraps a trained FakeNewsCapsNet and produces an interpretability
    dashboard as a Plotly HTML file.

    Parameters
    ----------
    model       : FakeNewsCapsNet instance (eval mode)
    tokenizer   : HuggingFace tokenizer matching the BERT backbone
    class_names : list of class label strings, e.g. ["Real", "Fake"]
    device      : torch device string
    """

    COLORS = {
        "fake": "#E8593C",
        "real": "#3B8BD4",
        "neutral": "#B4B2A9",
        "purple": "#7F77DD",
        "teal": "#1D9E75",
        "amber": "#EF9F27",
    }

    def __init__(self, model, tokenizer, class_names=("Real", "Fake"), device="cpu"):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.class_names = list(class_names)
        self.device = device

        if not isinstance(model.routing_layer, InstrumentedCapsuleLayer):
            model.routing_layer = InstrumentedCapsuleLayer(model.routing_layer)

        self._linear_hook = _Hook()
        self._linear_hook.register(model.proj[0])

        self._prim_hooks = []
        for cap in model.primary_caps:
            h = _Hook()
            target = cap[0] if isinstance(cap, nn.Sequential) else cap
            h.register(target)
            self._prim_hooks.append(h)

    def explain(
        self, text: str, save_html: str = "capsnet_report.html", show: bool = True
    ) -> dict:
        artifacts = self._run_inference(text)
        fig = self._build_dashboard(text, artifacts)
        pio.write_html(fig, save_html, auto_open=show)
        print(f"[CapsNetExplainer] Dashboard saved → {save_html}")
        return artifacts

    def explain_batch(
        self, texts: list, save_html: str = "capsnet_batch.html", show: bool = True
    ):
        for i, t in enumerate(texts):
            out = save_html.replace(".html", f"_{i}.html")
            self.explain(t, save_html=out, show=show)

    def _run_inference(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.max_seq_len,
            padding="max_length",
        )
        input_ids = enc["input_ids"].to(self.device)
        attn_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            probs, attn = self.model(input_ids, attn_mask, return_attention=True)

        # ── STEP 1: remove special tokens ───────────────────────────
        ids = input_ids[0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        filtered_tokens, filtered_indices = [], []
        for i, t in enumerate(tokens):
            if t in ("[PAD]", "[CLS]", "[SEP]", "<pad>"):
                continue
            filtered_tokens.append(t)
            filtered_indices.append(i)

        # ── STEP 2: merge subwords into full words ──────────────────
        merged_tokens, merged_indices = _merge_subwords(
            filtered_tokens, filtered_indices
        )

        # ── STEP 3: keep only meaningful tokens ─────────────────────
        valid_tokens, valid_idx_groups = [], []
        for tok, idx_group in zip(merged_tokens, merged_indices):
            clean_tok = tok.strip(string.punctuation)
            if not _is_meaningful_token(clean_tok):
                continue
            valid_tokens.append(clean_tok)
            valid_idx_groups.append(idx_group)

        # ── attention matrix ────────────────────────────────────
        attn_np = attn[0].cpu().numpy()  # [seq, seq]
        n_words = len(valid_idx_groups)
        attn_sub = np.zeros((n_words, n_words))
        for i, group_i in enumerate(valid_idx_groups):
            for j, group_j in enumerate(valid_idx_groups):
                vals = [attn_np[ii, jj] for ii in group_i for jj in group_j]
                attn_sub[i, j] = np.mean(vals) if vals else 0.0

        n_valid = attn_sub.shape[1]

        # ── token attention: entropy-filter + subtract uniform baseline ──
        row_entropy = -(attn_sub * np.log(attn_sub + 1e-9)).sum(axis=1)
        sharp_mask = row_entropy <= np.percentile(row_entropy, 50)
        sharp = attn_sub[sharp_mask] if sharp_mask.any() else attn_sub
        raw = sharp.mean(axis=0)
        raw -= raw.min()
        token_attn = raw / (raw.max() + 1e-8)

        # ── attention matrix for display: log-normalise ─────────
        attn_log = np.log1p(attn_sub * n_valid)
        attn_display = attn_log / (attn_log.max() + 1e-8)

        # ── projection importance: pre-LayerNorm Linear activations ──
        lin_acts = self._linear_hook.value  # [B, seq, 128]
        if lin_acts is not None:
            proj_importance = lin_acts[0].abs().mean(dim=-1).numpy()
            proj_sub = np.array(
                [proj_importance[group].mean() for group in valid_idx_groups]
            )
            proj_sub /= proj_sub.max() + 1e-8
        else:
            proj_sub = np.ones(len(valid_idx_groups))

        # ── per-token routing bias ──────────────────────────────
        routing_c = self.model.routing_layer.last_c
        max_seq = self.model.max_seq_len
        n_cls = len(self.class_names)
        if routing_c is not None:
            r = routing_c[0].reshape(-1, max_seq, n_cls)  # [32, seq, n_cls]
            r_per_tok = r.mean(dim=0).numpy()  # [seq, n_cls]
            routing_per_token = np.array(
                [r_per_tok[group].mean(axis=0) for group in valid_idx_groups]
            )
            routing_mean = routing_c[0].mean(dim=0).numpy()
        else:
            routing_per_token = np.full((len(valid_idx_groups), n_cls), 1.0 / n_cls)
            routing_mean = np.ones(n_cls) / n_cls

        probs_np = probs[0].cpu().numpy()
        pred_cls = int(probs_np.argmax())

        return dict(
            text=text,
            tokens=list(valid_tokens),
            token_attn=token_attn,
            attn_matrix=attn_sub,
            attn_display=attn_display,
            proj_importance=proj_sub,
            routing_mean=routing_mean,
            routing_per_token=routing_per_token,
            probs=probs_np,
            pred_cls=pred_cls,
        )

    # ── dashboard builder ──────────────────────────────────────

    def _build_dashboard(self, text: str, art: dict) -> go.Figure:
        tokens = art["tokens"]
        n = len(tokens)
        is_fake = art["pred_cls"] == 1
        verdict = self.class_names[art["pred_cls"]]
        verdict_color = self.COLORS["fake"] if is_fake else self.COLORS["real"]

        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=(
                "Token Self-Attention (above-average attention only)",
                "Attention Matrix — log-normalised",
                "Class Capsule Norms",
                "Per-Token Routing Bias (Fake − Real)", 
                "BERT Projection Importance (pre-LayerNorm, top tokens)",
                "",
                "",
                "",
            ),
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "bar", "colspan": 2},    None],
                [{"type": "bar", "colspan": 2},    None],
                [{"type": "bar", "colspan": 2},    None],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12,
        )

        # ── 1. Token attention ─────────────────────────────────
        colors_tok = [
            (
                self.COLORS["fake"]
                if (is_fake and v > 0.5)
                else (
                    self.COLORS["real"]
                    if (not is_fake and v > 0.5)
                    else self.COLORS["neutral"]
                )
            )
            for v in art["token_attn"]
        ]
        fig.add_trace(
            go.Bar(
                x=tokens,
                y=art["token_attn"].tolist(),
                marker_color=colors_tok,
                hovertemplate="<b>%{x}</b><br>above-avg attention: %{y:.3f}<extra></extra>",
                name="Token attention",
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(range=[0, 1.05], title_text="Normalised weight", row=1, col=1)

        # ── 2. Attention matrix (log-normalised) ───────────────
        cscale = (
            [[0, "#FFF5F5"], [0.5, "#F0997B"], [1, "#E8593C"]]
            if is_fake
            else [[0, "#EBF5FF"], [0.5, "#85B7EB"], [1, "#185FA5"]]
        )
        fig.add_trace(
            go.Heatmap(
                z=art["attn_display"].tolist(),
                x=tokens,
                y=tokens,
                colorscale=cscale,
                showscale=True,
                colorbar=dict(
                    len=0.35,
                    y=0.78,
                    x=1.01,
                    thickness=10,
                    title=dict(text="log-norm", side="right"),
                ),
                hovertemplate="query: <b>%{y}</b><br>key: <b>%{x}</b><br>log-norm: %{z:.3f}<extra></extra>",
                name="Attention",
            ),
            row=1,
            col=2,
        )

        # ── 4. Class capsule norms ─────────────────────────────
        fig.add_trace(
            go.Bar(
                x=self.class_names,
                y=art["probs"].tolist(),
                marker_color=[self.COLORS["real"], self.COLORS["fake"]],
                marker_opacity=0.8,
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
                name="Class norms",
                text=[f"{v:.2f}" for v in art["probs"].tolist()],
                textposition="outside",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(range=[0, 1.15], title_text="Capsule L2 norm", row=2, col=2)

        # ── 5. Per-token routing bias ──────────────────────────
        routing_bias = (
            art["routing_per_token"][:, 1] - art["routing_per_token"][:, 0]
        ).tolist()
        bias_colors = [
            self.COLORS["fake"] if b > 0 else self.COLORS["real"] for b in routing_bias
        ]
        fig.add_trace(
            go.Bar(
                x=tokens,
                y=routing_bias,
                marker_color=bias_colors,
                marker_opacity=0.8,
                hovertemplate="<b>%{x}</b><br>routing bias (Fake−Real): %{y:.4f}<extra></extra>",
                name="Routing bias",
            ),
            row=3,
            col=1,
        )
        fig.update_yaxes(title_text="Routing bias (Fake − Real)", row=3, col=1)
        fig.update_xaxes(tickangle=-40, row=3, col=1)
        fig.add_hline(
            y=0, line_width=1, line_dash="dot", line_color="#999", row=3, col=1
        )

        # ── 6. BERT projection importance ─────────────────────
        order = np.argsort(art["proj_importance"])[::-1]
        top_n = min(n, 15)
        top_idx = order[:top_n]
        top_tok = [tokens[i] for i in top_idx]
        top_imp = art["proj_importance"][top_idx].tolist()
        imp_cols = [
            (
                self.COLORS["fake"]
                if (is_fake and v > 0.5)
                else (
                    self.COLORS["real"]
                    if (not is_fake and v > 0.5)
                    else self.COLORS["neutral"]
                )
            )
            for v in top_imp
        ]
        fig.add_trace(
            go.Bar(
                x=top_tok,
                y=top_imp,
                marker_color=imp_cols,
                marker_opacity=0.75,
                hovertemplate="<b>%{x}</b><br>mean |activation|: %{y:.3f}<extra></extra>",
                name="Projection importance",
            ),
            row=4,
            col=1,
        )
        fig.update_yaxes(
            range=[0, 1.1], title_text="Normalised importance", row=4, col=1
        )

        # ── layout & title ─────────────────────────────────────
        pred_pct = art["probs"][art["pred_cls"]] * 100

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>FakeNewsCapsNet — Interpretability Report</b><br>"
                    f"<span style='font-size:13px;color:{verdict_color}'>"
                    f"Verdict: <b>{verdict}</b> ({pred_pct:.1f}% confidence)</span>"
                ),
                x=0.02,
                xanchor="left",
                font=dict(size=16),
            ),
            height=1400,
            width=1200,
            showlegend=False,
            template="plotly_white",
            paper_bgcolor="#FAFAF8",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Arial, sans-serif", size=11, color="#3d3d3a"),
            margin=dict(t=140, b=40, l=60, r=80),
        )

        for row in range(1, 5):
            for col in range(1, 3):
                fig.update_xaxes(showgrid=False, row=row, col=col)
                fig.update_yaxes(
                    showgrid=True, gridcolor="#EBEBEB", gridwidth=0.5, row=row, col=col
                )

        fig.add_annotation(
            x=0.5,
            y=1.07,
            xref="paper",
            yref="paper",
            text=f"  ● PREDICTED: <b>{verdict}</b> ({pred_pct:.1f}% confidence)  ",
            showarrow=False,
            align="center",
            font=dict(size=13, color="white"),
            bgcolor=verdict_color,
            bordercolor=verdict_color,
            borderwidth=1,
            borderpad=6,
            opacity=0.9,
        )

        return fig


MODEL_PATH = "best_epoch_model-v2-2.pt"
BERT_NAME = "distilbert-base-uncased"
MAX_SEQ = 64
DEVICE = "cpu"


@st.cache_resource(show_spinner="Loading model…")
def load_explainer():
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    model = FakeNewsCapsNet(
        num_classes=2, bert_model_name=BERT_NAME, max_seq_len=MAX_SEQ
    )
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return CapsNetExplainer(
        model, tokenizer, class_names=["Real", "Fake"], device=DEVICE
    )


def main():
    st.set_page_config(
        page_title="Fake News Detector — CapsNet",
        page_icon="📰",
        layout="wide",
    )

    st.title("📰 Fake News Detector")
    st.caption("Powered by **FakeNewsCapsNet** · DistilBERT backbone + Capsule Network")
    st.markdown("---")

    # ── Example selector ──────────────────────────────────────────────────────
    st.subheader("🗂 Try an example")
    example_labels = ["— select an example —"] + [e["label"] for e in EXAMPLES]
    selected_label = st.selectbox(
        "Choose a pre-loaded example to populate the form below:",
        options=example_labels,
        index=0,
    )

    # Determine which example (if any) is selected
    selected_example = next(
        (e for e in EXAMPLES if e["label"] == selected_label), None
    )

    # Pre-fill values from example or leave blank
    default_title = selected_example["title"] if selected_example else ""
    default_body = selected_example["body"] if selected_example else ""

    if selected_example:
        is_fake_example = "🔴" in selected_example["label"]
        badge_color = "#E8593C" if is_fake_example else "#3B8BD4"
        badge_text = "FAKE" if is_fake_example else "REAL"
        st.markdown(
            f"<span style='background:{badge_color};color:white;padding:3px 10px;"
            f"border-radius:4px;font-size:0.85em;font-weight:bold'>{badge_text} example</span> "
            f"<span style='color:#666;font-size:0.85em'>— form pre-filled below. Click <b>Classify & Explain</b> to run.</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("news_form"):
        title = st.text_input(
            "News Title",
            value=default_title,
            placeholder="Enter the headline…",
        )
        body = st.text_area(
            "News Body",
            value=default_body,
            placeholder="Paste the full article text here…",
            height=200,
        )
        submitted = st.form_submit_button(
            "🔍 Classify & Explain", use_container_width=True
        )

    if not submitted:
        return

    if not title.strip() and not body.strip():
        st.warning("Please enter a title or body before submitting.")
        return

    full_text = f"{title.strip()} . {body.strip()}" if title.strip() else body.strip()

    with st.spinner("Running inference…"):
        explainer = load_explainer()
        artifacts = explainer._run_inference(full_text)

    pred_cls = artifacts["pred_cls"]
    probs = artifacts["probs"]
    verdict = explainer.class_names[pred_cls]
    confidence = float(probs[pred_cls]) * 100
    is_fake = pred_cls == 1

    # ── Verdict banner ────────────────────────────────────────────────────────
    col_real, col_fake = st.columns(2)
    real_pct = float(probs[0]) * 100
    fake_pct = float(probs[1]) * 100

    with col_real:
        st.metric(
            label="✅ Real",
            value=f"{real_pct:.1f}%",
            delta="predicted" if not is_fake else None,
        )
    with col_fake:
        st.metric(
            label="🚨 Fake",
            value=f"{100 - real_pct:.1f}%",
            delta="predicted" if is_fake else None,
        )


    st.markdown("---")
    st.subheader("Interpretability Dashboard")

    art = explainer.explain(full_text)
    fig = explainer._build_dashboard(full_text, artifacts)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()