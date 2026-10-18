"""Psi: an open-source workflow language model.

Psi reads a memory of past workflow decisions (embedding, label) and answers a new call by weighing its nearest
memories with a learned, multi-head network (a neural kNN hybrid) instead of a fixed softmax(beta * cosine). Each retrieved memory is a pattern; its relative
distance to the query is the token's positional encoding, and pairwise distances between memories (plus whether two
memories share a label) are per-head attention biases. The model is label-agnostic, so one pretrained Psi works on a
new workflow without training: give it the workflow's memory and it votes.

    p(label | query) = sum over retrieved memories i with that label of softmax(Psi(query, memories))_i

Train (Modal, one H100):  modal run psi.py
Use:                      from psi import Psi, predict
"""
from __future__ import annotations

import json, math, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 64          # memories retrieved per call
DIM = 384       # BAAI/bge-small-en-v1.5
NFREQ = 32      # sinusoidal frequencies for the distance positional encoding


class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h, self.n1, self.n2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, bias):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.n2(x))


class Psi(nn.Module):
    def __init__(self, d: int = 1024, layers: int = 16, heads: int = 16):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads)
        self.inp = nn.Linear(4 * DIM, d)                    # [x, q*x, x-q, q]
        self.feat = nn.Linear(3 + 2 * NFREQ, d)             # scalars + distance positional encoding
        self.cls = nn.Linear(DIM, d)                        # the query token
        self.pair = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, heads))  # (relative distance, same label) -> head bias
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm, self.out = nn.LayerNorm(d), nn.Linear(d, 1)
        self.register_buffer("freq", 2.0 ** torch.linspace(0, 9, NFREQ), persistent=False)

    def pe(self, dist):  # relative distance -> sinusoidal positional encoding
        a = dist[..., None] * self.freq
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, q, X, y_nb, n_c, m):
        """q (B,DIM) query, X (B,K,DIM) retrieved memories, y_nb (B,K) their labels, n_c (B,K) rows with that label in the memory,
        m (B,) memory size -> (B,K) logits over the retrieved memories."""
        s = (X * q[:, None]).sum(-1)
        dist = 1 - s
        tok = self.inp(torch.cat([X, X * q[:, None], X - q[:, None], q[:, None].expand_as(X)], -1))
        sc = torch.stack([s, torch.log1p(n_c.float()) / 8, torch.log1p(m.float())[:, None].expand_as(s) / 8], -1)
        tok = tok + self.feat(torch.cat([sc, self.pe(dist)], -1))
        x = torch.cat([self.cls(q)[:, None] + self.feat(torch.cat([torch.zeros_like(sc[:, :1]), self.pe(torch.zeros_like(dist[:, :1]))], -1)), tok], 1)
        # pairwise relative distances (query = token 0) and label agreement -> per-head attention bias
        A = torch.cat([q[:, None], X], 1)
        D = 1 - A @ A.transpose(1, 2)
        lab = torch.cat([torch.full_like(y_nb[:, :1], -1), y_nb], 1)
        same = (lab[:, :, None] == lab[:, None, :]).float()
        bias = self.pair(torch.stack([D, same], -1)).permute(0, 3, 1, 2).to(x.dtype)
        for b in self.blocks:
            x = b(x, bias)
        return self.out(self.norm(x[:, 1:])).squeeze(-1)


def retrieve(Q, M, ym, k=K):
    """Q (B,DIM), memory M (N,DIM) unit rows with labels ym (N,) -> top-k memories, their labels, class counts."""
    S = Q @ M.T
    k = min(k, M.shape[0])
    idx = S.topk(k, dim=1).indices
    counts = torch.bincount(ym, minlength=int(ym.max()) + 1)
    return M[idx], ym[idx], counts[ym[idx]]


@torch.no_grad()
def predict(model, Q, M, ym, n_labels, bs=512):
    """Label distribution (B, n_labels) for queries Q against memory (M, ym)."""
    out = []
    for i in range(0, len(Q), bs):
        q = Q[i:i + bs]
        X, yn, nc = retrieve(q, M, ym)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=q.is_cuda):
            w = model(q, X, yn, nc, torch.full((len(q),), len(M), device=q.device)).float().softmax(-1)
        out.append(torch.zeros(len(q), n_labels, device=q.device).scatter_add_(1, yn, w))
    return torch.cat(out)


# ---- the current Psi read (dreaming/predictors/memory.py), for parity: softmax(beta*cos) mass per label / its row count
BETAS = (5, 10, 20, 40, 80, 160, 320, 640)


def hopfield_read(S, Y, beta):
    z = beta * S
    p = (torch.exp(z - z.max(1, keepdim=True).values) @ Y) / Y.sum(0).clamp(min=1)
    return p / p.sum(1, keepdim=True)


def hopfield_beta(M, ym, n_labels):
    Y = F.one_hot(ym, n_labels).double()
    q = torch.arange(0, len(ym), -(-len(ym) // 2000), device=M.device)
    S = (M[q] @ M.T).double()
    ar = torch.arange(len(q), device=M.device)
    S[ar, q] = -float("inf")
    nll = [-(hopfield_read(S, Y, b)[ar, ym[q]] + 1e-9).log().mean().item() for b in BETAS]
    return BETAS[int(np.argmin(nll))]


def hopfield_predict(Q, M, ym, n_labels):
    beta, Y = hopfield_beta(M, ym, n_labels), F.one_hot(ym, n_labels).double()
    return torch.cat([hopfield_read((Q[i:i + 1024] @ M.T).double(), Y, beta) for i in range(0, len(Q), 1024)])


# ---------------------------------------------------------------- training on Modal
try:
    import modal
except ImportError:  # using the model needs only torch
    modal = None

if modal is not None:
    app = modal.App("psi-train")
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch==2.5.1", "numpy<2.2", "sentence-transformers==3.3.1", "datasets==3.2.0", "safetensors==0.4.5")
             .add_local_python_source("psi"))
    out_vol = modal.Volume.from_name("psi-data", create_if_missing=True)
    dreaming = modal.Volume.from_name("dreaming-weights")

    HF = [  # (dataset, config, split) diverse domains: news, intents, emotions, questions, finance, reviews, spam, topics
        ("fancyzhx/ag_news", None, "train"), ("mteb/banking77", None, "train"), ("dair-ai/emotion", None, "train"),
        ("clinc_oos", "plus", "train"), ("mteb/amazon_massive_intent", "en", "train"), ("mteb/amazon_massive_scenario", "en", "train"),
        ("SetFit/20_newsgroups", None, "train"), ("fancyzhx/dbpedia_14", None, "train"), ("SetFit/sst5", None, "train"),
        ("zeroshot/twitter-financial-news-topic", None, "train"),
        ("mteb/tweet_sentiment_extraction", None, "train"), ("SetFit/enron_spam", None, "train"),
        ("SetFit/TREC-QC", None, "train"), ("SetFit/subj", None, "train"),
        ("SetFit/bbc-news", None, "train"), ("SetFit/hate_speech_offensive", None, "train"), ("SetFit/ade_corpus_v2_classification", None, "train"),
        ("SetFit/toxic_conversations_50k", None, "train"), ("Yelp/yelp_review_full", None, "train"), ("SetFit/student-question-categories", None, "train"),
    ]
    HELDOUT = {"mteb/banking77", "dair-ai/emotion"}  # never trained on: zero-shot generalization
    SITES = ["bookkeeper.categorize", "bookkeeper.match_invoice", "bookkeeper.anything_odd", "quickbooks.bill_account"]
    CAP = 6000

    def load_hf(enc):
        from datasets import load_dataset
        out = {}
        for name, cfg, split in HF:
            try:
                ds = load_dataset(name, cfg, split=split)
                cols = ds.column_names
                tc = next(c for c in ("text", "sentence", "content", "question_title", "label_text_x") if c in cols)
                lc = next(c for c in ("label", "intent", "label_coarse", "topic", "coarse_label") if c in cols)
                ds = ds.shuffle(seed=0).select(range(min(CAP, len(ds))))
                texts, labs = [str(t)[:600] for t in ds[tc]], ds[lc]
                vocab = {v: i for i, v in enumerate(sorted(set(labs), key=str))}
                y = np.array([vocab[v] for v in labs])
                E = enc.encode(texts, batch_size=512, normalize_embeddings=True, convert_to_numpy=True)
                out[name] = (E.astype(np.float32), y)
                print(f"data {name}: {len(y)} rows, {len(vocab)} labels", flush=True)
            except Exception as e:
                print(f"data {name}: skipped ({type(e).__name__}: {str(e)[:80]})", flush=True)
        return out

    def load_sites():
        import hashlib
        out = {}
        for site in SITES:
            d = f"/weights/campaign/{site}"
            if not os.path.exists(f"{d}/embeddings.npy"):
                print(f"site {site}: not prepared", flush=True); continue
            meta = json.load(open(f"{d}/meta.json"))
            rows = [json.loads(l) for l in open(f"{d}/rows.jsonl")]
            seeds = [json.loads(l) for l in open(f"{d}/seeds.jsonl")] if os.path.exists(f"{d}/seeds.jsonl") else []
            E = np.load(f"{d}/embeddings.npy").astype(np.float32)
            E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
            lab = {l: i for i, l in enumerate(meta["labels"])}
            keep = [i for i, r in enumerate(rows) if r["label"] in lab]
            order = sorted(keep, key=lambda i: (rows[i]["ts"], rows[i]["rendered"]))
            cut = len(order) * 4 // 5
            fwd, train, test = order[cut:], [], []
            for i in order[:cut]:
                b = int(hashlib.sha256(str(rows[i].get("group", "")).encode()).hexdigest(), 16) % 10
                (test if 2 <= b < 4 else train if b >= 4 else []).append(i)
            sidx = [len(rows) + j for j, s in enumerate(seeds) if s["label"] in lab]
            y = np.array([lab.get(r["label"], -1) for r in rows] + [lab.get(s["label"], -1) for s in seeds])
            out[site] = dict(E=E, y=y, mem=np.array(train + sidx), unseen=np.array(test), repeat=np.array(fwd), n=len(meta["labels"]))
            print(f"site {site}: memory {len(train) + len(sidx)}, unseen {len(test)}, repeat {len(fwd)}, labels {len(meta['labels'])}", flush=True)
        return out

    def acc(P, y):
        return float((P.argmax(1) == y).float().mean())

    def evaluate(model, sites, hf, dev):
        res = {}
        for site, s in sites.items():
            E, y = torch.tensor(s["E"], device=dev), torch.tensor(s["y"], device=dev)
            M, ym = E[s["mem"]], y[s["mem"]]
            for split in ("unseen", "repeat"):
                idx = torch.tensor(s[split], device=dev)
                if len(idx) == 0 or len(M) < 2:
                    continue
                Q, yq = E[idx], y[idx]
                res[f"{site}/{split}"] = dict(n=len(idx), psi=acc(predict(model, Q, M, ym, s["n"]), yq),
                                              hopfield=acc(hopfield_predict(Q, M, ym, s["n"]), yq))
        for name in HELDOUT & hf.keys():
            E, y = hf[name]
            g = torch.Generator().manual_seed(1)
            p = torch.randperm(len(y), generator=g)
            E, y = torch.tensor(E, device=dev)[p], torch.tensor(y, device=dev)[p]
            M, ym, Q, yq = E[:2000], y[:2000], E[2000:4000], y[2000:4000]
            n = int(y.max()) + 1
            res[f"{name}/zero-shot"] = dict(n=len(yq), psi=acc(predict(model, Q, M, ym, n), yq), hopfield=acc(hopfield_predict(Q, M, ym, n), yq))
        return res

    @app.function(image=image, gpu="H100", timeout=30 * 60, volumes={"/out": out_vol, "/weights": dreaming})
    def train(budget_s: int = 300, d: int = 1024, layers: int = 16, heads: int = 16, tag: str = "psi"):
        from safetensors.torch import save_file
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        dev = "cuda"
        torch.manual_seed(0)
        enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)
        enc.max_seq_length = 128
        hf, sites = load_hf(enc), load_sites()
        import signal; signal.alarm(0)  # a custom-code prompt in datasets leaves an alarm pending; it would fire mid-training
        print(f"data ready in {time.time() - t0:.0f}s", flush=True)
        pools = [(torch.tensor(E, device=dev), torch.tensor(y, device=dev)) for n, (E, y) in hf.items() if n not in HELDOUT]
        pools += [(torch.tensor(s["E"][s["mem"]], device=dev), torch.tensor(s["y"][s["mem"]], device=dev)) for s in sites.values() if len(s["mem"]) > 300]
        wts = torch.tensor([math.sqrt(len(p[1])) for p in pools])
        model = Psi(d, layers, heads).to(dev)
        npar = sum(p.numel() for p in model.parameters())
        print(f"Psi {npar / 1e6:.0f}M parameters, {len(pools)} training pools", flush=True)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05, betas=(0.9, 0.95))
        t1, step, ema = time.time(), 0, None
        while (el := time.time() - t1) < budget_s:
            frac = el / budget_s
            for g in opt.param_groups:
                g["lr"] = 2e-4 * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
            E, y = pools[int(torch.multinomial(wts, 1))]
            n = len(y)
            perm = torch.randperm(n, device=dev)
            m = int(math.exp(np.random.uniform(math.log(100), math.log(max(150, min(5000, int(n * 0.7)))))))
            Mi, Qi = perm[:m], perm[m:m + 256]
            M, ym, Q, yq = E[Mi], y[Mi], E[Qi], y[Qi]
            if np.random.rand() < 0.35:  # repeat traffic: near-copies of some queries already sit in the memory
                r = Q[:128] + 0.02 * torch.randn_like(Q[:128])
                M, ym = torch.cat([M, r / r.norm(dim=1, keepdim=True)]), torch.cat([ym, yq[:128]])
            X, yn, nc = retrieve(Q, M, ym)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                w = model(Q, X, yn, nc, torch.full((len(Q),), len(M), device=dev)).float()
            hit = yn == yq[:, None]
            ok = hit.any(1)
            p_true = (w.softmax(-1) * hit).sum(-1)
            loss = -(p_true[ok] + 1e-6).log().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % 100 == 0:
                print(f"step {step} t {el:.0f}s loss {ema:.4f}", flush=True)
        model.eval()
        res = evaluate(model, sites, hf, dev)
        for k, v in res.items():
            print(f"eval {k}: psi {v['psi']:.3f} hopfield {v['hopfield']:.3f} (n {v['n']})", flush=True)
        os.makedirs(f"/out/{tag}", exist_ok=True)
        save_file({k: v.detach().to(torch.float16).contiguous() for k, v in model.state_dict().items()}, f"/out/{tag}/model.safetensors")
        info = dict(params=npar, config=model.cfg, k=K, encoder="BAAI/bge-small-en-v1.5", steps=step, train_seconds=budget_s,
                    pools=len(pools), datasets=sorted(hf), heldout=sorted(HELDOUT), sites=sorted(sites), eval=res)
        json.dump(info, open(f"/out/{tag}/metrics.json", "w"), indent=1)
        out_vol.commit()
        return info

    @app.local_entrypoint()
    def main(budget_s: int = 300, tag: str = "psi"):
        info = train.remote(budget_s, tag=tag)
        print(json.dumps(info, indent=1))
