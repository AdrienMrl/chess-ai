"""Trains the NNUE-style eval net (768 -> 256 ReLU -> 1 linear) on
ai/data/positions.jsonl and saves weights to ai/data/weights.npz.

Standalone script, run with the venv interpreter (numpy required):

    cd /Users/adri/dev/chess-ai
    ai/.venv/bin/python3 ai/nnue/train.py

Hand-rolled forward/backward pass (no autograd), plain-momentum SGD,
full-batch gradient descent, capped by both an epoch limit and a wall-clock
time limit so this can never run away.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chess_game import Board, WHITE  # noqa: E402

from ai.nnue.nnue_eval import NUM_FEATURES, encode_board  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "positions.jsonl")
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "weights.npz")

HIDDEN = 256
LABEL_CLIP = 1000.0
LEARNING_RATE = 0.001
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8
MAX_EPOCHS = 15000
TIME_LIMIT_SECONDS = 5 * 60
PRINT_EVERY = 200
SEED = 0

# Overfitting guards: held-out validation split, decoupled weight decay
# (AdamW-style, applied to the weight matrices only, not biases), and early
# stopping on validation loss -- the saved weights are the best-val-loss
# snapshot, not just whatever the last epoch happened to produce.
VAL_FRACTION = 0.15
WEIGHT_DECAY = 1e-4
PATIENCE_EPOCHS = 800


def load_dataset():
    """Reads ai/data/positions.jsonl and returns (X, y) as float32 arrays.
    X is (N, 768) one-hot piece-placement features. y is (N,) centipawn
    labels converted to WHITE's perspective and clipped to [-1000, 1000].

    The file's cp label is from the side-to-move's perspective in that FEN;
    we flip its sign whenever it's Black to move, so every label is always
    White-relative -- matching the turn-agnostic feature encoding and the
    evaluate() contract (White's perspective).
    """
    board = Board()
    xs = []
    ys = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Dataset file is being appended to concurrently by another
                # process; a trailing partially-written line is expected and
                # safely skipped.
                continue
            fen = obj.get("fen")
            cp = obj.get("cp")
            if fen is None or cp is None:
                continue
            try:
                board.load_fen(fen)
            except Exception:
                continue
            label = float(cp)
            if board.turn != WHITE:
                label = -label
            label = max(-LABEL_CLIP, min(LABEL_CLIP, label))
            xs.append(encode_board(board))
            ys.append(label)
    if not xs:
        raise RuntimeError(f"no usable positions found in {DATA_PATH}")
    X = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.float32)
    return X, y


def init_params(rng):
    # He-ish init scaled for a 768-wide sparse one-hot input layer.
    W1 = (rng.standard_normal((NUM_FEATURES, HIDDEN)) * np.sqrt(2.0 / NUM_FEATURES)).astype(np.float32)
    b1 = np.zeros(HIDDEN, dtype=np.float32)
    W2 = (rng.standard_normal((HIDDEN, 1)) * np.sqrt(2.0 / HIDDEN)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    return W1, b1, W2, b2


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1          # (N, HIDDEN)
    h1 = np.maximum(z1, 0.0)  # ReLU
    out = h1 @ W2 + b2        # (N, 1)
    return z1, h1, out


def train():
    print(f"Loading dataset from {DATA_PATH} ...")
    X, y = load_dataset()
    n_total = X.shape[0]
    print(f"Loaded {n_total} positions.")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_val = max(1, int(n_total * VAL_FRACTION))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    n = X_train.shape[0]
    print(f"Split: {n} train / {n_val} val.")

    W1, b1, W2, b2 = init_params(rng)

    # Adam moment buffers (per-parameter adaptive step size -- this keeps
    # updates well-scaled regardless of raw gradient magnitude, which matters
    # here because the 768-wide sparse one-hot input makes plain SGD/momentum
    # gradients blow up and drive the ReLU units permanently negative/"dead").
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    y_train_col = y_train.reshape(-1, 1)
    y_val_col = y_val.reshape(-1, 1)

    best_val_loss = float("inf")
    best_params = None
    epochs_since_improve = 0

    start = time.time()
    epoch = 0
    train_loss = None
    val_loss = None
    while epoch < MAX_EPOCHS:
        elapsed = time.time() - start
        if elapsed > TIME_LIMIT_SECONDS:
            print(f"Time limit ({TIME_LIMIT_SECONDS}s) reached, stopping.")
            break

        z1, h1, out = forward(X_train, params["W1"], params["b1"], params["W2"], params["b2"])
        diff = out - y_train_col  # (N, 1)
        train_loss = float(np.mean(diff ** 2))

        # Backprop: MSE -> linear layer 2 -> ReLU -> linear layer 1.
        d_out = (2.0 / n) * diff              # (N, 1)
        gW2 = h1.T @ d_out                    # (HIDDEN, 1)
        gb2 = d_out.sum(axis=0)               # (1,)

        d_h1 = d_out @ params["W2"].T         # (N, HIDDEN)
        d_z1 = d_h1 * (z1 > 0)                # ReLU grad
        gW1 = X_train.T @ d_z1                # (768, HIDDEN)
        gb1 = d_z1.sum(axis=0)                # (HIDDEN,)

        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
        t = epoch + 1
        for k in params:
            m[k] = ADAM_BETA1 * m[k] + (1 - ADAM_BETA1) * grads[k]
            v[k] = ADAM_BETA2 * v[k] + (1 - ADAM_BETA2) * (grads[k] ** 2)
            m_hat = m[k] / (1 - ADAM_BETA1 ** t)
            v_hat = v[k] / (1 - ADAM_BETA2 ** t)
            params[k] = params[k] - LEARNING_RATE * m_hat / (np.sqrt(v_hat) + ADAM_EPS)
            if k in ("W1", "W2"):  # decoupled weight decay, weights only
                params[k] = params[k] - LEARNING_RATE * WEIGHT_DECAY * params[k]

        _, _, val_out = forward(X_val, params["W1"], params["b1"], params["W2"], params["b2"])
        val_loss = float(np.mean((val_out - y_val_col) ** 2))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {k: v_.copy() for k, v_ in params.items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % PRINT_EVERY == 0 or epoch == MAX_EPOCHS - 1:
            print(f"epoch {epoch:4d}  train_loss={train_loss:.2f} (rmse={np.sqrt(train_loss):.2f})  "
                  f"val_loss={val_loss:.2f} (rmse={np.sqrt(val_loss):.2f})  elapsed={elapsed:.1f}s")

        if epochs_since_improve >= PATIENCE_EPOCHS:
            print(f"No val improvement for {PATIENCE_EPOCHS} epochs, early stopping at epoch {epoch}.")
            break

        epoch += 1

    total_time = time.time() - start
    final_params = best_params if best_params is not None else params
    print(f"Training finished after {epoch} epochs, {total_time:.1f}s. "
          f"Best val_loss={best_val_loss:.2f} (rmse={np.sqrt(best_val_loss):.2f} cp) "
          f"-- saving best-val-loss snapshot, not the final epoch's weights.")

    np.savez(WEIGHTS_PATH, **final_params)
    print(f"Saved weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    train()
