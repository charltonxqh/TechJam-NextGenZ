"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""

import argparse
import collections
import time

import numpy as np

from data import load, encode, FIELDS
from evaluate import evaluate


def sigmoid(x):
    return 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                x,
                -30,
                30,
            )
        )
    )


# ---------------- item popularity（官方 baseline） ----------------
def run_pop(
    splits,
    split="valid",
    prior=20.0,
):
    pos = collections.Counter()
    imp = collections.Counter()

    for x in splits["train"]:
        imp[x[2]] += 1
        pos[x[2]] += x[6]

    gmean = (
        sum(pos.values())
        / sum(imp.values())
    )

    score = lambda v: (
        (
            pos[v]
            + prior * gmean
        )
        / (
            imp[v]
            + prior
        )
        if imp[v]
        else gmean
    )

    rows = splits[split]

    return {
        split: evaluate(
            [
                x[1]
                for x in rows
            ],
            [
                x[6]
                for x in rows
            ],
            [
                score(x[2])
                for x in rows
            ],
        )
    }


def run_random(
    splits,
    split="valid",
    seed=0,
):
    rng = np.random.default_rng(
        seed
    )

    rows = splits[split]

    return {
        split: evaluate(
            [
                x[1]
                for x in rows
            ],
            [
                x[6]
                for x in rows
            ],
            rng.random(
                len(rows)
            ),
        )
    }


# ---------------- Factorization Machine ----------------
class FM:

    def __init__(
        self,
        dim,
        k=16,
        lr=0.001,
        l2=1e-6,
        seed=0,
    ):
        rng = np.random.default_rng(
            seed
        )

        self.V = rng.normal(
            0,
            0.01,
            (
                dim,
                k,
            ),
        ).astype(
            np.float32
        )

        self.W = np.zeros(
            dim,
            dtype=np.float32,
        )

        self.b = np.float32(
            0.0
        )

        self.lr = lr
        self.l2 = l2

        self.mV = np.zeros_like(
            self.V
        )

        self.vV = np.zeros_like(
            self.V
        )

        self.mW = np.zeros_like(
            self.W
        )

        self.vW = np.zeros_like(
            self.W
        )

        self.t = 0

    def logits(
        self,
        X,
    ):
        E = self.V[X]

        S = E.sum(
            1
        )

        inter = 0.5 * (
            (
                S ** 2
            ).sum(
                1
            )
            - (
                E ** 2
            ).sum(
                (
                    1,
                    2,
                )
            )
        )

        return (
            self.b
            + self.W[X].sum(
                1
            )
            + inter,
            E,
            S,
        )

    def step(
        self,
        X,
        y,
    ):
        B = len(
            y
        )

        z, E, S = self.logits(
            X
        )

        g = (
            (
                sigmoid(z)
                - y
            )
            / B
        ).astype(
            np.float32
        )

        gV = np.zeros_like(
            self.V
        )

        gW = np.zeros_like(
            self.W
        )

        np.add.at(
            gW,
            X,
            g[:, None],
        )

        np.add.at(
            gV,
            X,
            (
                g[
                    :,
                    None,
                    None,
                ]
                * (
                    S[
                        :,
                        None,
                        :,
                    ]
                    - E
                )
            ),
        )

        gV += (
            self.l2
            * self.V
        )

        gW += (
            self.l2
            * self.W
        )

        self.t += 1

        b1 = 0.9
        b2 = 0.999
        eps = 1e-8

        for (
            P,
            G,
            M,
            Vv,
        ) in (
            (
                self.V,
                gV,
                self.mV,
                self.vV,
            ),
            (
                self.W,
                gW,
                self.mW,
                self.vW,
            ),
        ):

            M *= b1

            M += (
                1
                - b1
            ) * G

            Vv *= b2

            Vv += (
                1
                - b2
            ) * (
                G
                * G
            )

            P -= (
                self.lr
                * (
                    M
                    / (
                        1
                        - b1
                        ** self.t
                    )
                )
                / (
                    np.sqrt(
                        Vv
                        / (
                            1
                            - b2
                            ** self.t
                        )
                    )
                    + eps
                )
            )

        self.b -= (
            self.lr
            * g.sum()
        )

        return float(
            -np.mean(
                y
                * np.log(
                    sigmoid(z)
                    + 1e-9
                )
                + (
                    1
                    - y
                )
                * np.log(
                    1
                    - sigmoid(z)
                    + 1e-9
                )
            )
        )

    def predict(
        self,
        X,
        bs=200_000,
    ):
        return np.concatenate(
            [
                self.logits(
                    X[
                        i:
                        i + bs
                    ]
                )[0]
                for i in range(
                    0,
                    len(X),
                    bs,
                )
            ]
        )


def run_fm(
    splits,
    split="valid",
    k=16,
    lr=0.001,
    epochs=40,
    bs=8192,
    patience=4,
    seed=0,
    verbose=True,
):
    # During autonomous research, validation mode does not
    # encode or evaluate the test split.
    encode_splits = {
        "train": splits["train"],
        "valid": splits["valid"],
    }

    if split == "test":
        encode_splits[
            "test"
        ] = splits[
            "test"
        ]

    enc, dim = encode(
        encode_splits
    )

    Xtr, ytr, _ = (
        enc["train"]
    )

    Xva, yva, uva = (
        enc["valid"]
    )

    m = FM(
        dim,
        k=k,
        lr=lr,
        seed=seed,
    )

    rng = np.random.default_rng(
        seed
    )

    best = -1
    best_state = None
    bad = 0

    for ep in range(
        1,
        epochs + 1,
    ):
        idx = rng.permutation(
            len(ytr)
        )

        t0 = time.time()

        losses = [
            m.step(
                Xtr[
                    idx[
                        i:
                        i + bs
                    ]
                ],
                ytr[
                    idx[
                        i:
                        i + bs
                    ]
                ],
            )
            for i in range(
                0,
                len(idx),
                bs,
            )
        ]

        va = evaluate(
            uva,
            yva,
            m.predict(
                Xva
            ),
        )

        if verbose:
            print(
                f"  epoch {ep:2d} | "
                f"loss {np.mean(losses):.4f} | "
                f"valid GAUC {va['GAUC']:.4f} "
                f"nDCG@5 {va['nDCG@5']:.4f} "
                f"primary {va['primary']:.4f} | "
                f"{time.time() - t0:.1f}s"
            )

        if (
            va["primary"]
            > best
            + 1e-5
        ):
            best = (
                va["primary"]
            )

            bad = 0

            best_state = (
                m.V.copy(),
                m.W.copy(),
                np.float32(
                    m.b
                ),
            )

        else:
            bad += 1

            if (
                bad
                >= patience
            ):
                if verbose:
                    print(
                        f"  early stop "
                        f"at epoch {ep}"
                    )

                break

    m.V, m.W, m.b = (
        best_state
    )

    if split == "valid":

        result = evaluate(
            uva,
            yva,
            m.predict(
                Xva
            ),
        )

    else:

        Xte, yte, ute = (
            enc["test"]
        )

        result = evaluate(
            ute,
            yte,
            m.predict(
                Xte
            ),
        )

    return {
        split: result
    }


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data_dir",
        default="./KuaiRand-Pure/data",
        help=(
            "KuaiRand-Pure "
            "解压后的 data 目录"
        ),
    )

    ap.add_argument(
        "--model",
        default="fm",
        choices=[
            "pop",
            "fm",
            "random",
        ],
    )

    ap.add_argument(
        "--split",
        default="valid",
        choices=[
            "valid",
            "test",
        ],
    )

    ap.add_argument(
        "--k",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    a = ap.parse_args()

    print(
        f"loading "
        f"{a.data_dir} ..."
    )

    splits = load(
        a.data_dir
    )

    print(
        {
            k_: len(v)
            for k_, v
            in splits.items()
        },
        f"fields={FIELDS}",
    )

    res = {
        "pop": lambda s: run_pop(
            s,
            split=a.split,
        ),

        "random": lambda s: run_random(
            s,
            split=a.split,
            seed=a.seed,
        ),

        "fm": lambda s: run_fm(
            s,
            split=a.split,
            k=a.k,
            lr=a.lr,
            epochs=a.epochs,
            seed=a.seed,
        ),
    }[
        a.model
    ](
        splits
    )

    r = res[
        a.split
    ]

    print(
        f"\n=== "
        f"{a.model} "
        f"(seed={a.seed}) ==="
    )

    print(
        f"  {a.split:5s}  "
        f"GAUC {r['GAUC']:.4f} | "
        f"nDCG@5 {r['nDCG@5']:.4f} | "
        f"primary {r['primary']:.4f}"
    )