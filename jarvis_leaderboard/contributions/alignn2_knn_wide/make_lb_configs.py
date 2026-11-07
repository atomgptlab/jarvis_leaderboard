"""Derive exact-split leaderboard configs from a tuned finals config.

Copies a base config, forces the fixed-order split (keep_data_order +
explicit n_train/n_val/n_test), and writes one config per training seed.
Only the standard library is needed.
"""
import argparse
import json
import os


def main():
    """Write one exact-split config per torch_seed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", required=True,
                    help="base config json shipped in this folder")
    ap.add_argument("--n_train", type=int, default=44569)
    ap.add_argument("--n_val", type=int, default=5572)
    ap.add_argument("--n_test", type=int, default=5572)
    ap.add_argument("--seeds", default="123,231,312")
    ap.add_argument("--out_prefix", default="config_lb_knn")
    args = ap.parse_args()

    base = json.load(open(args.base_config))
    base["keep_data_order"] = True
    base["n_train"] = args.n_train
    base["n_val"] = args.n_val
    base["n_test"] = args.n_test
    for k in ("train_ratio", "val_ratio", "test_ratio"):
        base.pop(k, None)

    for seed in args.seeds.split(","):
        cfg = dict(base)
        cfg["torch_seed"] = int(seed)
        out = "%s_s%s.json" % (args.out_prefix, seed)
        json.dump(cfg, open(out, "w"), indent=2)
        print("wrote", out, "| torch_seed", seed,
              "| keep_data_order", cfg["keep_data_order"],
              "| n", cfg["n_train"], cfg["n_val"], cfg["n_test"])


if __name__ == "__main__":
    main()
