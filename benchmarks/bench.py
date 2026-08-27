"""Timing and correctness harness for bitermplus.

Run before and after a change and compare:

    python benchmarks/bench.py --label before
    python benchmarks/bench.py --label after
    python benchmarks/bench.py --compare before after
"""

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np

import bitermplus as btm

HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "dataset" / "SearchSnippets.txt"

N_TOPICS = 20
WINDOW = 15
SEED = 42
ITERATIONS = 100


def load_texts(limit=None):
    with open(DATASET, encoding="utf-8") as fh:
        texts = [line.strip() for line in fh if line.strip()]
    return texts[:limit] if limit else texts


def timed(repeats, fn):
    """Return (median seconds, last result)."""
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), result


def peak_mb(fn):
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6, result


def run(label, limit, repeats, out_dir):
    texts = load_texts(limit)
    timings = {}
    memory = {}

    def vectorize():
        X, vocab, _ = btm.get_words_freqs(texts)
        docs_vec = btm.get_vectorized_docs(texts, vocab)
        return X, vocab, docs_vec

    timings["vectorize"], (X, vocab, docs_vec) = timed(repeats, vectorize)

    timings["get_biterms"], biterms = timed(
        repeats, lambda: btm.get_biterms(docs_vec, win=WINDOW)
    )
    memory["get_biterms"], _ = peak_mb(lambda: btm.get_biterms(docs_vec, win=WINDOW))

    # The array form is what BTMClassifier uses; its "before" equivalent is
    # the list form above, since as_array did not exist previously.
    timings["get_biterms_array"], biterms_arr = timed(
        repeats, lambda: btm.get_biterms(docs_vec, win=WINDOW, as_array=True)
    )
    memory["get_biterms_array"], _ = peak_mb(
        lambda: btm.get_biterms(docs_vec, win=WINDOW, as_array=True)
    )

    def make_model():
        return btm.BTM(X, vocab, T=N_TOPICS, seed=SEED, win=WINDOW)

    timings["biterms_to_array_fast"], _ = timed(
        repeats, lambda: make_model().fit(biterms_arr, iterations=0, verbose=False)
    )

    # First fit iteration is dominated by _biterms_to_array; measure it at
    # iterations=0 so the Gibbs loop does not mask it.
    timings["biterms_to_array"], _ = timed(
        repeats, lambda: make_model().fit(biterms, iterations=0, verbose=False)
    )
    memory["biterms_to_array"], _ = peak_mb(
        lambda: make_model().fit(biterms, iterations=0, verbose=False)
    )

    model = make_model()
    timings["fit"], _ = timed(
        repeats,
        lambda: model.fit(biterms, iterations=ITERATIONS, verbose=False),
    )

    for infer in ("sum_b", "sum_w", "mix"):
        timings[f"transform_{infer}"], _ = timed(
            repeats,
            lambda infer=infer: model.transform(docs_vec, infer_type=infer, verbose=False),
        )

    timings["coherence"], coherence = timed(repeats, lambda: model.coherence_)

    # transform is pure since 1.0, so p_zd is passed to perplexity explicitly
    p_zd = model.transform(docs_vec, verbose=False)
    timings["perplexity"], perplexity = timed(
        repeats, lambda: btm.perplexity(model.matrix_topics_words_, p_zd, X, N_TOPICS)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "label": label,
        "n_docs": len(texts),
        "vocab_size": len(vocab),
        "n_biterms": int(model.biterms_.shape[0]),
        "n_topics": N_TOPICS,
        "iterations": ITERATIONS,
        "repeats": repeats,
        "timings_s": timings,
        "peak_mb": memory,
    }
    (out_dir / f"results-{label}.json").write_text(json.dumps(results, indent=2))

    np.savez(
        out_dir / f"fingerprint-{label}.npz",
        p_wz=model.matrix_topics_words_,
        theta=model.theta_,
        coherence=np.asarray(coherence),
        perplexity=np.asarray([perplexity]),
        p_zd=p_zd,
    )

    print(f"\n== {label}: {len(texts)} docs, {len(vocab)} words, "
          f"{results['n_biterms']} biterms ==")
    for name, seconds in timings.items():
        mem = memory.get(name)
        suffix = f"   peak {mem:8.1f} MB" if mem else ""
        print(f"  {name:22s} {seconds:9.3f} s{suffix}")
    return results


def compare(before, after, out_dir):
    b = json.loads((out_dir / f"results-{before}.json").read_text())
    a = json.loads((out_dir / f"results-{after}.json").read_text())

    print(f"\n{'section':22s} {before:>10s} {after:>10s} {'speedup':>9s}")
    print("-" * 54)
    for name, bt in b["timings_s"].items():
        at = a["timings_s"].get(name)
        if at is None:
            continue
        print(f"{name:22s} {bt:9.3f}s {at:9.3f}s {bt / at:8.2f}x")

    if b["peak_mb"]:
        print(f"\n{'peak memory':22s} {before:>10s} {after:>10s} {'ratio':>9s}")
        print("-" * 54)
        for name, bm in b["peak_mb"].items():
            am = a["peak_mb"].get(name)
            if am is None:
                continue
            print(f"{name:22s} {bm:8.1f}MB {am:8.1f}MB {bm / am:8.2f}x")

    fb = np.load(out_dir / f"fingerprint-{before}.npz")
    fa = np.load(out_dir / f"fingerprint-{after}.npz")
    print("\nfingerprint (results must be unchanged):")
    ok = True
    for key in fb.files:
        same = np.allclose(fb[key], fa[key], rtol=1e-9, atol=1e-12)
        ok &= same
        same_shape = fb[key].shape == fa[key].shape
        delta = float(np.max(np.abs(fb[key] - fa[key]))) if same_shape else float("nan")
        print(f"  {key:12s} {'MATCH' if same else 'DIFFERS'}  max|delta| = {delta:.3e}")
    print("  => identical\n" if ok else "  => CHANGED\n")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", help="run the benchmark and save under this label")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N documents (default: all)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare, out_dir=args.out_dir)
    elif args.label:
        run(args.label, args.limit, args.repeats, args.out_dir)
    else:
        parser.error("pass --label or --compare")


if __name__ == "__main__":
    main()
