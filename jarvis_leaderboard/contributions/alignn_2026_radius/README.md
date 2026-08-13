# ALIGNN-2026 (Pure-PyTorch)

Property-prediction models trained with the **pure-PyTorch ALIGNN** backend —
i.e. the `alignn_atomwise_pure` model with the `pure_torch` neighbor strategy.
This is the same graph / line-graph construction used by the ALIGNN-FF force
field, but **without any DGL or torch_geometric dependency** — neighbor lists,
edge displacement vectors, bond-angle line graphs, and batched readout are all
implemented in native PyTorch (`alignn/torch_graph_builder.py`,
`alignn/pure_lmdb_dataset.py`, `alignn/models/alignn_atomwise_pure.py`).

## Benchmarks

| Benchmark | Dataset | Property |
|-----------|---------|----------|
| `AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae` | dft_3d | Formation energy per atom (eV/atom) |
| `AI-SinglePropertyPrediction-optb88vdw_bandgap-dft_3d-test-mae` | dft_3d | OptB88vdW band gap (eV) |

Same 44,569 / 5,572 / 5,572 train/val/test split as the reference ALIGNN entry.

## Results (test MAE)

| Benchmark | Pure-torch ALIGNN | Reference DGL ALIGNN |
|-----------|------------------:|---------------------:|
| formation_energy_peratom (eV/atom) | **0.0312** | ~0.033 |
| optb88vdw_bandgap (eV) | **0.1293** | ~0.14 |

Note on the angular cutoff: an earlier run with `three_body_cutoff=4.0`
(truncating the line graph) gave 0.0429 / 0.1521. Using the full 12-NN line
graph (`three_body_cutoff = cutoff = 8.0`) recovers the reference-level
accuracy — the pair graph (12-NN @ 8 Å) was already correct; the angular
graph was the difference.

## Model / training

- Model: `alignn_atomwise_pure`, 4 ALIGNN + 4 GCN layers, hidden = 256,
  embedding = 64, edge = 80, triplet = 40.
- Graph: `pure_torch` neighbor strategy, cutoff = 8.0 Å, max_neighbors = 12,
  three_body_cutoff = 8.0 Å (full 12-NN line graph, i.e. every pair-graph edge
  also forms angular triplets — matches the reference ALIGNN k-nearest angular
  graph), CGCNN atom features.
- Graph-level regression only (`calculate_gradient=False`,
  `gradwise_weight=0`, `stresswise_weight=0`), single intensive target.
- Optimizer AdamW, OneCycle schedule, lr = 1e-3, L1 loss, batch = 64,
  300 epochs. Graphs cached to LMDB (`use_lmdb=True`).

## Reproduce

```bash
# needs an alignn install exposing alignn_atomwise_pure + pure_torch strategy
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

`run.py` populates each benchmark with `jarvis_populate_data.py`, generates the
config via `make_config.py`, trains with `train_alignn.py`, and zips
`prediction_results_test_set.csv` (`id,target,prediction`) into the submission
`*.csv.zip` files.
