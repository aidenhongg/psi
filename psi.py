"""Psi: an open-source workflow language model.

Psi answers a workflow's calls from a memory of its past decisions. For each call it embeds the input, fetches the eight most similar
past decisions, lays them out as worked examples after the workflow's own instructions, and lets a 4B language model that was trained
once to learn from examples (Qwen3-4B-Instruct-2507 plus the rank-32 LoRA adapter in ./adapter) write the answer.

    from psi import Psi
    psi = Psi("Categorize the transaction. Answer with exactly one of: Travel, Hosting, Meals.")
    psi.add("UBER *TRIP 8.12", "Travel")
    answer, rows = psi("LYFT RIDE 14.20")

Tested with transformers 5.19, peft 0.21 and sentence-transformers 6.1. A GPU is recommended: the bf16 weights are about 8 GB.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

BASE = "Qwen/Qwen3-4B-Instruct-2507"
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
K = 8            # past decisions shown per call: the adapter was trained with eight
MAX_SEQ = 2048   # caps the embedder's 32k window so one pathological input cannot stall a batch


class Psi:
    def __init__(self, instructions: str, adapter: str | Path = Path(__file__).parent / "adapter", *, k: int = K,
                 max_new_tokens: int = 768, device: str | None = None) -> None:
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.instructions, self.k, self.max_new_tokens = instructions, k, max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(EMBEDDER, device=self.device, model_kwargs={"dtype": torch.float32},
                                            processor_kwargs={"padding_side": "left"})
        self.embedder.max_seq_length = min(self.embedder.max_seq_length or MAX_SEQ, MAX_SEQ)
        self.tok = AutoTokenizer.from_pretrained(BASE)
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(self.device)
        self.model = PeftModel.from_pretrained(model, str(adapter)).eval()
        self.rows: list[tuple[str, str]] = []   # the memory: (input, answer)
        self.vecs = np.zeros((0, self.embedder.get_sentence_embedding_dimension()), dtype=np.float32)

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False).astype(np.float32)

    def add(self, text: str, answer: str) -> None:
        """Remember a decision. It counts on the very next call: no training step."""
        self.rows.append((text, answer))
        self.vecs = np.vstack([self.vecs, self._embed([text])])

    def remove(self, i: int) -> None:
        """Forget row i. Its influence is gone: the adapter never trained on your memory."""
        del self.rows[i]
        self.vecs = np.delete(self.vecs, i, axis=0)

    def nearest(self, text: str) -> list[int]:
        """Indices of the k most similar rows by cosine, the closest LAST (as in training)."""
        if not self.rows:
            return []
        sims = self.vecs @ self._embed([text])[0]
        return [int(j) for j in np.argsort(-sims, kind="stable")[: self.k][::-1]]

    @torch.no_grad()
    def __call__(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        """-> (the answer, the past decisions it read). With an empty memory the adapter runs zero-shot, which it was not trained for."""
        rows = [self.rows[j] for j in self.nearest(text)]
        chat = [{"role": "system", "content": self.instructions}]
        for x, y in rows:
            chat += [{"role": "user", "content": x}, {"role": "assistant", "content": y}]
        chat.append({"role": "user", "content": text})
        prompt = self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip(), rows
