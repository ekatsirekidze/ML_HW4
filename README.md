Kaggle competition: **Challenges in Representation Learning: Facial Expression Recognition Challenge**

PyTorch + Weights & Biases experiment tracking for ML coursework.

## Repository structure

```
ml4/
├── configs/
│   └── experiments.yaml      # Hyperparameter grids per architecture
├── src/
│   ├── checks/
│   │   └── model_checks.py   # Forward & backward pass verification
│   ├── data/
│   │   └── dataset.py        # Data loading, augmentation, splits
│   ├── models/
│   │   └── architectures.py  # 5 architectures (iterative design)
│   └── utils/
│       ├── metrics.py        # Accuracy, F1, under/overfit diagnosis
│       ├── seed.py
│       └── wandb_logger.py   # MLflow-style per-run logging
├── notebooks/
│   └── FER_Colab.ipynb       # Google Colab entry point
├── train.py                  # Single experiment → one W&B run
├── run_experiments.py        # Grid search → many W&B runs
├── predict.py                # Kaggle submission generator
├── requirements.txt
└── README.md                 # Experiment decisions & analysis (fill after runs)
```

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 28,709 | `emotion` (0–6) + `pixels` (48×48 grayscale) |
| `test.csv` | 7,178 | `pixels` only |
| `example_submission.csv` | — | Kaggle format: one `emotion` column |

**Emotion labels:** 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

---

## Architecture progression (required by assignment)

We build **5 architectures** from simple → complex. Each step is a deliberate design choice.

### V1 — `mlp_underfit` (underfitting on purpose)

- **Design:** Single hidden layer (128 units), no convolutions, no augmentation.
- **Why:** Baseline with too little inductive bias for images.
- **Expected:** Low train **and** val accuracy (~45–55%). Model cannot capture spatial patterns.
- **Hyperparameters to sweep:** `lr ∈ {0.001, 0.0005}`, `batch_size ∈ {64, 128}`

### V2 — `cnn_small` (first CNN)

- **Design:** 2 conv blocks (16→32 channels), small FC head.
- **Why:** Adds locality + hierarchy; should beat MLP clearly.
- **Expected:** Moderate accuracy; still limited depth.
- **Hyperparameters:** `dropout`, `lr`, `weight_decay`, augmentation on.

### V3 — `cnn_medium` (main comparator)

- **Design:** 3 conv blocks (32→64→128), FC 256.
- **Why:** Sweet spot for 48×48 images without excessive params.
- **Expected:** Good val accuracy; reasonable train–val gap.
- **Hyperparameters:** Full grid in `configs/experiments.yaml`.

### V4 — `cnn_deep_overfit` (overfitting on purpose)

- **Design:** 4 deep blocks (64→128→256→512), **low dropout**, **no augmentation**, **no weight decay**.
- **Why:** Demonstrate overfitting for the report.
- **Expected:** Train acc ≫ val acc (gap > 12%), val may plateau or worsen.
- **Analysis point:** Memorization vs generalization.

### V5 — `cnn_best` (best model)

- **Design:** 4 blocks + **BatchNorm**, strong dropout (0.4–0.5), augmentation, label smoothing.
- **Why:** Production-quality regularization stack.
- **Expected:** Best val accuracy + smallest harmful gap.

---

## Experiment log template (copy into README after each run)

For **every** run, document:

```markdown
### Run: {run_name}
- **Architecture:** 
- **Hyperparameters:** lr=, batch=, dropout=, weight_decay=, epochs=
- **Train acc / Val acc / Gap:** 
- **Forward check:** OK / FAIL — notes
- **Backward check:** grad norm=, vanishing? exploding?
- **Diagnosis:** underfit | mild_overfit | overfit | good_fit
- **Why:** (2–3 sentences explaining curves)
- **W&B link:** 
```

---

## W&B logging (MLflow-style)

Each experiment = **one `wandb.init()` run** with:

| Category | Logged items |
|----------|----------------|
| **Config** | All hyperparameters, `num_parameters`, train/val sizes |
| **Metrics (per epoch)** | `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `train_val_gap/*`, `learning_rate`, `fit_diagnosis` |
| **Checks** | `checks/forward_pass_ok`, `checks/backward_pass_ok`, `checks/total_grad_norm` |
| **Artifacts** | Best checkpoint `.pt`, confusion matrix PNG, `summary.json` |
| **Summary** | `best_val_accuracy`, `final_fit_diagnosis` |

**Groups:** `group=model_name` so W&B UI clusters runs by architecture (like MLflow experiments).

---

## Quick start (local)

```bash
pip install -r requirements.txt
wandb login

# Single run
python train.py --model mlp_underfit --lr 0.001 --batch_size 64 --epochs 25 --no_augmentation --group mlp_underfit

# Full grid for one architecture
python run_experiments.py --arch_key arch_v3_cnn_medium

# All architectures (run sequentially)
python run_experiments.py --arch_key arch_v1_mlp_underfit
python run_experiments.py --arch_key arch_v2_cnn_small
python run_experiments.py --arch_key arch_v3_cnn_medium
python run_experiments.py --arch_key arch_v4_cnn_deep_overfit
python run_experiments.py --arch_key arch_v5_cnn_best

# Kaggle submission
python predict.py --checkpoint checkpoints/cnn_best_lr0.0005_bs64_do0.5_best.pt --output submission.csv
```

---

## Google Colab setup

1. Upload this repo to GitHub.
2. Open `notebooks/FER_Colab.ipynb` in Colab.
3. Runtime → GPU.
4. Add Kaggle API key (Settings → API → Create New Token).
5. `wandb.login()` with your API key.
6. Run all cells.

---

## Forward & backward checks (lecture requirement)

Implemented in `src/checks/model_checks.py`, logged at **start of every run**:

**Forward check**
- Output shape `(batch, 7)`
- No NaN/Inf in logits
- Softmax sums to 1

**Backward check**
- Finite loss after one backward step
- Every trainable param receives a gradient
- Total gradient norm — flag vanishing (<1e-7) or exploding (>1e3)

---

## Underfitting vs overfitting — how we diagnose

Automatic label `fit_diagnosis` each epoch (`src/utils/metrics.py`):

| Signal | Interpretation |
|--------|----------------|
| Train & val both low | **Underfit** — increase capacity or train longer |
| Train high, val much lower, gap > 12% | **Overfit** — more dropout, augmentation, weight decay |
| Train ≈ val, both reasonably high | **Good fit** |

**Always plot** train vs val curves in W&B and explain *why* in README.

---

## W&B Report (bonus points)

After all runs:

1. W&B → Reports → Create Report
2. Add sections:
   - **Overview** — task, dataset, goal
   - **Architecture comparison** — parallel coordinates or bar chart of `best_val_accuracy` by `group`
   - **Learning curves** — overlay train/val for underfit, good, overfit examples
   - **Hyperparameter analysis** — lr vs val acc per architecture
   - **Checks** — table of forward/backward pass results
   - **Conclusion** — which model for Kaggle, what you learned

Embed the report link in README and submit to classroom.

---

## GitHub + Classroom checklist

- [ ] Create GitHub repo, push this code (exclude large secrets)
- [ ] Connect repo to W&B: Project Settings → GitHub
- [ ] Run ≥3 architectures with ≥2 hyperparameter combos each
- [ ] Document **all** runs in README (not only the best)
- [ ] Include at least one clear **underfit** and one **overfit** example with analysis
- [ ] Submit repo URL to classroom
- [ ] (Bonus) Submit W&B report URL

---

## Results table (fill after experiments)

| Architecture | Best val acc | Train acc | Gap | Fit type | Best hyperparams |
|--------------|-------------|-----------|-----|----------|------------------|
| mlp_underfit | | | | underfit | |
| cnn_small | | | | | |
| cnn_medium | | | | | |
| cnn_deep_overfit | | | | overfit | |
| cnn_best | | | | good_fit | |

---

## Author

Your Name — ML Assignment 4 — Facial Expression Recognition
