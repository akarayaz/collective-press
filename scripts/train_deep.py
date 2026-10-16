"""Deep models on the raw scene: does a model that sees how players relate do better?

    python scripts/train_deep.py --selftest          # run this first, about a minute
    python scripts/train_deep.py                     # the real run
    python scripts/train_deep.py --models mlp_context,mlp_metrics   # the tabular networks

Needs ``cache/scenes.npz`` (``scripts/export_scenes.py``) and
``cache/oof_<label>.csv`` (``scripts/run_models.py``).

Two models read the same 23 tokens per engagement (22 players and the ball,
see ``export_scenes.py``) plus SkillCorner's own context and geometry, the
inputs of the ``skillcorner_geometry`` tabular model:

``deepsets``     every token is encoded on its own, then pooled. Each player
                 is seen relative to the carrier, so it can count defenders
                 near the ball, but it cannot see how two defenders stand
                 relative to each other.
``transformer``  tokens attend to each other before pooling, so it can see
                 pairs: a presser and the teammate covering him, a defender
                 and the lane he stands in.

Two more networks see no scene, so the scene models have a fair baseline:

``mlp_context``  SkillCorner's context and geometry only
``mlp_metrics``  the same plus the 32 hand-made pressing metrics, the inputs
                 of the ``tracking_geometry`` tabular model

The comparisons are the experiment. Transformer against DeepSets: does how
players relate to each other carry information beyond where each stands
relative to the ball? Each against ``mlp_context``: does the raw scene add to
the context? ``mlp_metrics`` against ``mlp_context``: do the hand-made metrics
add to the context, with the same kind of model on both sides?

Predictions from earlier runs are kept in ``cache/oof_deep_<label>.csv``, so
models can be run one at a time and compared at the end.

Validation is the same leave-one-match-out split as the tabular models; inside
each training fold two matches are held back to decide when to stop. The
placebos shuffle the scenes (or the metrics) among the rows of each match,
keeping the context, so whatever the real model gains over its placebo comes
from the scenes (or the metrics).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pressing import config, modelling  # noqa: E402

CONTEXT = modelling.CONTEXT_FEATURES + modelling.SKILLCORNER_GEOMETRY
METRICS = modelling.TRACKING_GEOMETRY

#: model -> (uses the scene tokens, tabular columns it reads)
KINDS = {
    "deepsets": (True, CONTEXT),
    "transformer": (True, CONTEXT),
    "mlp_context": (False, CONTEXT),
    "mlp_metrics": (False, CONTEXT + METRICS),
}

# Feature positions in a token, as written by export_scenes.py.
Y_FEATURES = (1, 3, 5)  # y, dy, vy: flipped together to mirror the pitch


# --- inputs ----------------------------------------------------------------------

class ContextEncoder:
    """Tabular columns as a plain numeric matrix, fitted on the training fold.

    Numeric columns are standardised, with a missing indicator where needed;
    categorical columns become one column per level, with "missing" a level.
    """

    def __init__(self, columns=CONTEXT):
        self.columns = list(columns)

    def fit(self, df: pd.DataFrame) -> "ContextEncoder":
        self.numeric = [c for c in self.columns if c not in modelling.CATEGORICAL]
        self.categorical = [c for c in self.columns if c in modelling.CATEGORICAL]
        num = df[self.numeric].astype(float)
        self.median = num.median().fillna(0.0)
        filled = num.fillna(self.median)
        self.mean, self.std = filled.mean(), filled.std().replace(0, 1).fillna(1)
        self.with_missing = [c for c in self.numeric if num[c].isna().any()]
        self.levels = {c: sorted(set(self._as_level(df[c]))) for c in self.categorical}
        return self

    @staticmethod
    def _as_level(s: pd.Series) -> np.ndarray:
        return s.astype(object).where(s.notna(), "missing").astype(str).to_numpy()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = df[self.numeric].astype(float)
        parts = [((num.fillna(self.median) - self.mean) / self.std).to_numpy()]
        if self.with_missing:
            parts.append(num[self.with_missing].isna().to_numpy(dtype=float))
        for c, levels in self.levels.items():
            v = self._as_level(df[c])
            parts.append(np.column_stack([v == lv for lv in levels]).astype(float))
        return np.hstack(parts).astype(np.float32)


def load(label: str):
    table = modelling.load_table()
    table["meta_event_id"] = table["meta_event_id"].astype(str)

    path = config.REPO_ROOT / "cache" / "scenes.npz"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run scripts/export_scenes.py first")
    sc = np.load(path, allow_pickle=False)
    keys = pd.DataFrame({"meta_match_id": sc["match_id"],
                         "meta_event_id": sc["event_id"].astype(str),
                         "_scene": np.arange(len(sc["X"]))})
    keys["meta_match_id"] = keys["meta_match_id"].astype(table["meta_match_id"].dtype)
    data = table.merge(keys, on=["meta_match_id", "meta_event_id"], how="inner")
    if len(data) < 0.99 * len(table):
        raise SystemExit(f"only {len(data)} of {len(table)} rows have a scene; "
                         "rerun scripts/export_scenes.py")
    X = sc["X"][data["_scene"].to_numpy()]
    mask = sc["mask"][data["_scene"].to_numpy()]
    return data.reset_index(drop=True), X, mask


# --- the models ------------------------------------------------------------------

def build_model(kind: str, n_feat: int, n_ctx: int, d: int, layers: int, heads: int,
                dropout: float, base_rate: float):
    import torch
    from torch import nn

    uses_scene = KINDS[kind][0]

    class SetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            if not uses_scene:
                pass
            elif kind == "transformer":
                self.embed = nn.Linear(n_feat, d)
                layer = nn.TransformerEncoderLayer(
                    d, heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True
                )
                self.mixer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            else:  # deepsets: a small network applied to each token on its own
                self.embed = nn.Sequential(nn.Linear(n_feat, d), nn.ReLU(),
                                           nn.Linear(d, d), nn.ReLU())
            # The pooled scene is normalised before it meets the context, so a
            # tiny scene signal at the start of training is not drowned out by
            # the context path, which is shallower and learns faster.
            if uses_scene:
                self.scene_norm = nn.LayerNorm(3 * d)
            self.context = nn.Sequential(nn.Linear(n_ctx, d), nn.ReLU())
            n_in = 4 * d if uses_scene else d
            self.head = nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(d, 1))
            with torch.no_grad():  # start from the base rate rather than 50%
                self.head[-1].bias.fill_(float(np.log(base_rate / (1 - base_rate))))

        def forward(self, X, mask, C):
            if not uses_scene:  # a plain two-layer network on the tabular columns
                return self.head(self.context(C)).squeeze(-1)
            h = self.embed(X)
            if self.kind == "transformer":
                h = self.mixer(h, src_key_padding_mask=~mask)
            m = mask.unsqueeze(-1).to(h.dtype)
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            peak = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
            carrier = h[:, 0]  # token 0 is always the carrier
            scene = self.scene_norm(torch.cat([mean, peak, carrier], dim=-1))
            z = torch.cat([scene, self.context(C)], dim=-1)
            return self.head(z).squeeze(-1)

    return SetModel()


def fit_predict(X, M, C, y, groups, train, test, kind, args, seed):
    """Train on ``train`` rows (two of its matches held back to stop early), predict ``test``."""
    import torch
    from torch import nn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)

    train_matches = np.unique(groups[train])
    held = rng.choice(train_matches, size=min(2, len(train_matches) - 1), replace=False)
    val = train & np.isin(groups, held)
    fit = train & ~val

    t = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
    Xf, Mf, Cf, yf = t(X[fit]), t(M[fit], torch.bool), t(C[fit]), t(y[fit])
    Xv, Mv, Cv, yv = t(X[val]), t(M[val], torch.bool), t(C[val]), t(y[val])

    model = build_model(kind, X.shape[2], C.shape[1], args.width, args.layers, args.heads,
                        args.dropout, base_rate=float(y[fit].mean())).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    flip = torch.ones(X.shape[2], device=device)
    flip[list(Y_FEATURES)] = -1.0

    with torch.no_grad():  # what predicting the base rate for everyone would score
        p0 = torch.full_like(yv, float(y[fit].mean()))
        base_loss = nn.functional.binary_cross_entropy(p0, yv).item()

    best, best_state, best_epoch, stale = np.inf, None, 0, 0
    n = len(yf)
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(n).to(device)
        for start in range(0, n, args.batch):
            idx = order[start:start + args.batch]
            xb = Xf[idx]
            if args.mirror:  # flip half the batch top to bottom
                mirror = torch.rand(len(idx), device=device) < 0.5
                xb = torch.where(mirror[:, None, None], xb * flip, xb)
            opt.zero_grad()
            loss = loss_fn(model(xb, Mf[idx], Cf[idx]), yf[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xv, Mv, Cv), yv).item()
        if vloss < best - 1e-4:
            best, best_epoch, stale = vloss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience and epoch + 1 >= args.min_epochs:
                break

    if best_state is None:
        raise SystemExit("validation loss was never finite; try a lower --lr and paste the output")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(t(X[test]), t(M[test], torch.bool), t(C[test]))
    info = (f"best epoch {best_epoch:>2}/{epoch + 1:<2}  "
            f"val loss {best:.4f} (base rate {base_loss:.4f})")
    return torch.sigmoid(logits).cpu().numpy(), info


def cross_validate(data, X, M, y, kind, args, placebo: bool = False):
    groups = data["meta_match_id"].to_numpy()
    uses_scene, columns = KINDS[kind]
    X, M = X.copy(), M.copy()
    if placebo and uses_scene:  # each match's scenes dealt out to the wrong rows
        rng = np.random.default_rng(args.seed + 1)
        for g in np.unique(groups):
            rows = np.where(groups == g)[0]
            perm = rng.permutation(rows)
            X[rows], M[rows] = X[perm], M[perm]
    elif placebo:  # each match's metrics dealt out to the wrong rows, context kept
        data = modelling.shuffle_within_match(data, METRICS, seed=args.seed + 1)

    oof = np.full(len(y), np.nan)
    matches = np.unique(groups)[: args.folds] if args.folds else np.unique(groups)
    for i, match in enumerate(matches):
        t0 = time.time()
        test = groups == match
        train = ~test
        enc = ContextEncoder(columns).fit(data[train])
        C = enc.transform(data)
        oof[test], info = fit_predict(X, M, C, y, groups, train, test, kind, args,
                                      seed=args.seed + i)
        print(f"    fold {i + 1:>2}/{len(matches)}  match {match}  {info}  "
              f"{time.time() - t0:.0f}s", flush=True)
    return oof


# --- comparison ------------------------------------------------------------------

def as_result(name, p, data, label):
    return modelling.CVResult(
        name, pd.Series(p, index=data.index), data[label].astype(int),
        data["meta_match_id"],
    )


def report(results: dict, data, label, n_boot, seed):
    print("\nOut-of-fold performance")
    rows = [{"model": k, **v.metrics()} for k, v in results.items()]
    print(pd.DataFrame(rows).set_index("model").to_string(float_format=lambda v: f"{v:.4f}"))

    pairs = [
        ("skillcorner_geometry", "deepsets", "the raw scene, each player on its own"),
        ("skillcorner_geometry", "transformer", "the raw scene, players seen together"),
        ("deepsets", "transformer", "what seeing players together adds"),
        ("transformer_placebo", "transformer", "real scenes against shuffled ones"),
        ("tracking_geometry", "transformer", "transformer against the hand-made metrics"),
        ("tracking_geometry", "ensemble", "averaging the transformer with the tabular model"),
        ("mlp_context", "deepsets", "the raw scene on a neural context model, players alone"),
        ("mlp_context", "transformer", "the raw scene on a neural context model, players together"),
        ("mlp_context", "mlp_metrics", "the hand-made metrics on a neural context model"),
        ("mlp_metrics_placebo", "mlp_metrics", "real metrics against shuffled ones"),
        ("transformer", "mlp_metrics", "the hand-made metrics against the raw scene"),
    ]
    print("\nComparisons (positive = the second model is better)")
    for a, b, what in pairs:
        if a not in results or b not in results:
            continue
        delta = modelling.bootstrap_delta(results[a], results[b], n_samples=n_boot, seed=seed)
        won, total = modelling.matches_improved(results[a], results[b])
        print(f"\n  {a}  ->  {b}   ({what})")
        for metric in ("roc_auc", "log_loss"):
            point, lo, hi = delta[metric]
            print(f"    {metric:<9} {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"    matches improved: {won}/{total}")


# --- self test -------------------------------------------------------------------

def selftest(args) -> int:
    """Synthetic scenes where the answer is known; both models must learn something.

    The outcome depends on (a) how many defenders stand within 10 m of the
    carrier, which each player relative to the carrier reveals, and (b) how
    close the presser's nearest teammate is, which needs two defenders seen
    together. A model that learns nothing scores an AUC near 0.5.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n_matches, per = 6, 1000
    n = n_matches * per
    X = np.zeros((n, 23, 12), dtype=np.float32)
    M = np.ones((n, 23), dtype=bool)
    pos = rng.uniform(-30, 30, size=(n, 22, 2))
    pos[:, 0] = rng.uniform(-10, 10, size=(n, 2))                  # carrier
    pos[:, 1] = pos[:, 0] + rng.normal(0, 4, size=(n, 2))          # presser near the carrier
    near = np.hypot(*(pos[:, 2:12] - pos[:, :1]).transpose(2, 0, 1)) <= 10
    support = np.min(np.hypot(*(pos[:, 2:12] - pos[:, 1:2]).transpose(2, 0, 1)), axis=1)
    logit = -1.0 + 1.2 * near.sum(1) - 0.25 * support
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)

    side = np.array([0] + [1] * 11 + [0] * 10)  # token 0 carrier, 1..11 pressing, 12..21 attacking
    X[:, :22, 0] = pos[..., 0] / 52.5
    X[:, :22, 1] = pos[..., 1] / 34
    X[:, :22, 2] = (pos[..., 0] - pos[:, :1, 0]) / 20
    X[:, :22, 3] = (pos[..., 1] - pos[:, :1, 1]) / 20
    X[:, :22, 6] = side
    X[:, :22, 7] = 1 - side
    X[:, 1, 9] = 1.0
    X[:, 0, 10] = 1.0
    X[:, :22, 11] = 1.0
    X[:, 22, 8] = 1.0
    X[:, 22, 11] = 1.0

    data = pd.DataFrame({"meta_match_id": np.repeat(np.arange(n_matches), per)})
    for c in CONTEXT:
        data[c] = (pd.Series(rng.choice(["a", "b"], n), dtype="category")
                   if c in modelling.CATEGORICAL else rng.normal(size=n))

    args.folds = 3
    ok = True
    for kind in ("deepsets", "transformer"):
        print(f"  {kind}")
        oof = cross_validate(data, X, M, y, kind, args)
        done = ~np.isnan(oof)
        auc = roc_auc_score(y[done], oof[done])
        passed = auc > 0.60
        ok &= passed
        print(f"  {kind}: AUC {auc:.3f}  {'ok' if passed else 'TOO LOW, something is wrong'}\n")
    print("self test passed" if ok else "self test FAILED: paste this output")
    return 0 if ok else 1


# --- main ------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=modelling.DEFAULT_LABEL)
    ap.add_argument("--models", default="deepsets,transformer",
                    help=f"comma separated, from: {', '.join(KINDS)}")
    ap.add_argument("--skip-placebo", action="store_true")
    ap.add_argument("--folds", type=int, default=0, help="only the first N matches (quick look)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--min-epochs", type=int, default=10,
                    help="never stop before this many epochs")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mirror", action="store_true",
                    help="augment by flipping scenes top to bottom (off by default: "
                         "SkillCorner's channel in the context is not flipped with them)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("PyTorch is not installed: pip install torch")

    if args.selftest:
        print("Self test on synthetic scenes\n")
        return selftest(args)

    data, X, M = load(args.label)
    y = data[args.label].astype(float).to_numpy()
    print(f"{len(data)} engagements, {X.shape[1]} tokens x {X.shape[2]} features, "
          f"label {args.label} ({y.mean():.1%} positive), device {args.device}\n")

    kinds = [k for k in args.models.split(",") if k]
    unknown = set(kinds) - set(KINDS)
    if unknown:
        raise SystemExit(f"unknown model(s) {sorted(unknown)}; choose from {list(KINDS)}")

    preds = {}
    for kind in kinds:
        print(f"{kind}")
        t0 = time.time()
        preds[kind] = cross_validate(data, X, M, y, kind, args)
        print(f"  done in {(time.time() - t0) / 60:.1f} min\n")
    for kind in ("transformer", "mlp_metrics"):
        if not args.skip_placebo and kind in preds:
            what = "scenes" if KINDS[kind][0] else "metrics"
            print(f"{kind} placebo ({what} shuffled within each match)")
            preds[f"{kind}_placebo"] = cross_validate(data, X, M, y, kind, args, placebo=True)
            print()

    # Keep what earlier runs predicted, for models not rerun this time.
    path = config.REPO_ROOT / "cache" / f"oof_deep_{args.label}.csv"
    keys = data[["meta_match_id", "meta_event_id"]].rename(
        columns={"meta_match_id": "match_id", "meta_event_id": "event_id"})
    if path.is_file() and not args.folds:
        old = pd.read_csv(path)
        old["event_id"] = old["event_id"].astype(str)
        old = keys.merge(old, on=["match_id", "event_id"], how="left")
        for col in old.columns:
            name = col[2:]
            if col.startswith("p_") and name not in preds and old[col].notna().all():
                preds[name] = old[col].to_numpy()
                print(f"kept from the earlier run: {name}")

    out = keys.copy()
    out["y"] = y.astype(int)
    for k, p in preds.items():
        out[f"p_{k}"] = p
    if not args.folds:  # a quick run must not overwrite the full predictions
        out.to_csv(path, index=False)
        print(f"\nout-of-fold predictions: {path}")

    done = ~np.isnan(np.column_stack(list(preds.values()))).any(axis=1)
    if not done.all():
        print(f"(quick run: comparing on the {done.sum()} rows that were predicted)")
    sub = data[done]

    results = {}
    gbm_path = config.REPO_ROOT / "cache" / f"oof_{args.label}.csv"
    if gbm_path.is_file():
        gbm = pd.read_csv(gbm_path).rename(
            columns={"match_id": "meta_match_id", "event_id": "meta_event_id"})
        gbm["meta_event_id"] = gbm["meta_event_id"].astype(str)
        g = sub[["meta_match_id", "meta_event_id"]].merge(
            gbm, on=["meta_match_id", "meta_event_id"], how="left")
        for name in ("context", "skillcorner_geometry", "tracking_geometry"):
            if f"p_{name}" in g and g[f"p_{name}"].notna().all():
                results[name] = as_result(name, g[f"p_{name}"].to_numpy(), sub, args.label)
    for k, p in preds.items():
        results[k] = as_result(k, p[done], sub, args.label)
    if "tracking_geometry" in results and "transformer" in results:
        ens = (results["tracking_geometry"].oof.to_numpy() + preds["transformer"][done]) / 2
        results["ensemble"] = as_result("ensemble", ens, sub, args.label)

    report(results, sub, args.label, args.bootstrap, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
