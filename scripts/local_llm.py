"""Yes/No scoring with a local open-weight LLM (Apple MLX), shared by the label_*_local scripts.

A `YesNoScorer` holds one model and one system prompt. `score(user)` returns
(log P(Yes), log P(No)) for the first answer token. The system prompt is identical for every
item, so its KV cache is computed once and a copy is restored per item (this also works for
Gemma's rotating / sliding-window caches, which cannot be trimmed).
"""

from __future__ import annotations


def render(tok, system: str, user: str) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kw = {"add_generation_prompt": True, "tokenize": False}
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kw)  # Qwen3: no <think>
    except TypeError:
        return tok.apply_chat_template(msgs, **kw)


def answer_ids(tok) -> tuple[list[int], list[int]]:
    """First-token ids for Yes / No, with and without a leading space."""
    ids = lambda w: {tok.encode(v, add_special_tokens=False)[0] for v in (w, " " + w)}  # noqa: E731
    yes, no = ids("Yes"), ids("No")
    assert not yes & no
    return sorted(yes), sorted(no)


class YesNoScorer:
    def __init__(self, model_name: str, system: str):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self.mx, self.make_cache = mx, make_prompt_cache
        self.model, self.tok = load(model_name)
        self.system = system
        self.yes, self.no = answer_ids(self.tok)

        a = self.tok.encode(render(self.tok, system, "AAAA"), add_special_tokens=False)
        b = self.tok.encode(render(self.tok, system, "BBBB"), add_special_tokens=False)
        self.n_prefix = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y) - 2
        self.prefix = a[: self.n_prefix]
        cache = make_prompt_cache(self.model)
        self.model(mx.array(self.prefix)[None], cache=cache)
        self.snapshot = [self._copy(c.state) for c in cache]
        mx.eval([x for st in self.snapshot for x in st if isinstance(x, mx.array)])

    def _copy(self, st):
        return tuple(self.mx.array(x) if isinstance(x, self.mx.array) else x for x in st)

    def _fresh_cache(self):
        new = self.make_cache(self.model)
        for c, st in zip(new, self.snapshot):
            c.state = self._copy(st)
        return new

    def _logits(self, user: str, use_cache: bool = True):
        mx = self.mx
        toks = self.tok.encode(render(self.tok, self.system, user), add_special_tokens=False)
        if use_cache and toks[: self.n_prefix] == self.prefix:
            return self.model(mx.array(toks[self.n_prefix:])[None], cache=self._fresh_cache())[0, -1]
        return self.model(mx.array(toks)[None], cache=self.make_cache(self.model))[0, -1]

    def score(self, user: str) -> tuple[float, float]:
        mx = self.mx
        lp = self._logits(user).astype(mx.float32)
        lp = lp - mx.logsumexp(lp)
        return float(mx.logsumexp(lp[mx.array(self.yes)])), float(mx.logsumexp(lp[mx.array(self.no)]))

    def check(self, user: str) -> str:
        """Compare the cached-prefix logits with a full forward pass."""
        mx = self.mx
        full, fast = self._logits(user, use_cache=False), self._logits(user)
        pu = lambda lg: float(mx.softmax(mx.stack([mx.logsumexp(lg[mx.array(self.yes)]),  # noqa: E731
                                                    mx.logsumexp(lg[mx.array(self.no)])]).astype(mx.float32))[0])
        return (f"check: max |logit diff| {float(mx.abs(full - fast).max()):.3f}; "
                f"p_yes full {pu(full):.4f} vs cached {pu(fast):.4f}")
