Useful Links:
https://api.wandb.ai/links/etsir21-free-university-of-tbilisi-/dqv0bmxz
https://wandb.ai/etsir21-free-university-of-tbilisi-/fer-expression-recognition


# Facial Expression Recognition — ML HW4

A PyTorch implementation for the [FER2013 Kaggle Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge), exploring multiple architectures from underfitting to best-fit, with full experiment tracking on Weights & Biases.

**W&B Project:** https://wandb.ai/etsir21-free-university-of-tbilisi-/fer-expression-recognition  
**GitHub:** https://github.com/ekatsirekidze/ML_HW4

---

## Task

Classify facial expressions into 7 categories: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.  
Dataset: FER2013 — 48×48 grayscale images, ~28,000 training samples.

---

## Repository Structure

```
ML_HW4/
├── train.py                  # Main training script
├── predict.py                # Generates Kaggle submission CSV
├── run_experiments.py        # Runs hyperparameter grids per architecture
├── requirements.txt
├── src/
│   ├── data/
│   │   └── dataset.py        # DataLoader, augmentation, CSV parsing
│   ├── models/               # All model architectures
│   ├── checks/
│   │   └── model_checks.py   # Forward and backward sanity checks
│   └── utils/
├── checkpoints/              # Saved best model weights (.pt)
└── FER_Colab.ipynb           # Google Colab notebook to run everything
```

---

## Architectures Explored

We followed an iterative strategy — start small and add complexity, explaining each decision.

### V1 — MLP (Underfitting Baseline)

**Architecture:** Flatten → Linear(2304→128) → ReLU → Linear(128→7)  
**Purpose:** Establish a baseline. MLPs ignore spatial structure in images, so we expected poor performance.  
**Config:** 25 epochs, lr=0.001, batch=64, no augmentation, no dropout  

**Result:** Best val accuracy **44.0%**, final train/val gap **0.32** (overfit despite small model — MLP memorizes noise)  
**Diagnosis:** The model underfits early but quickly memorizes the small training patterns. Val accuracy plateaus at ~44%, showing the architecture itself is the bottleneck — not hyperparameters.  
**Decision:** Move to CNNs to exploit spatial features.

---

### V2 — CNN Small (2 conv layers)

**Architecture:** Conv(1→16) → Conv(16→32) → FC(128) → output  
**Purpose:** Introduce convolutional layers to capture local image features (edges, textures).  
**Config:** 35 epochs, lr=0.001, batch=64, dropout=0.25, weight_decay=1e-4, with augmentation  

**Results across hyperparameter grid:**

| LR | Dropout | Weight Decay | Best Val Acc | Diagnosis |
|---|---|---|---|---|
| 0.001 | 0.25 | 0.0001 | **56.8%** | good_fit |
| 0.001 | 0.25 | 0.001 | 55.4% | underfit |
| 0.001 | 0.50 | 0.0001 | 56.8% | good_fit |
| 0.0003 | 0.25 | 0.0001 | 55.2% | good_fit |
| 0.0003 | 0.25 | 0.001 | 54.3% | good_fit |
| 0.0003 | 0.50 | 0.0001 | 54.3% | underfit |

**Diagnosis:** A big jump from MLP (+12%). Higher weight decay (0.001) causes underfitting — regularization too strong for this model size. Dropout=0.5 doesn't help beyond 0.25, suggesting the model is not large enough to need heavy regularization.  
**Decision:** Model is still too small to capture complex expression features. Add more layers.

---

### V3 — CNN Medium (3 conv layers)

**Architecture:** Conv(1→32) → Conv(32→64) → Conv(64→128) → FC(256) → output  
**Purpose:** Deeper feature hierarchy to capture higher-level facial structures.  
**Config:** 40 epochs, lr=0.001, batch=64, dropout=0.3, weight_decay=1e-4, with augmentation  

**Result:** Best val accuracy **54.3%**, diagnosis: **underfit**  
**Diagnosis:** Interestingly, adding a third conv layer didn't immediately improve results. The model needed more epochs and a better learning rate schedule to converge. This showed that depth alone isn't sufficient — training dynamics matter.  
**Decision:** Move to a deeper, more carefully tuned best architecture.

---

### V4 — CNN Deep Overfit (intentional overfitting)

**Architecture:** Conv(1→64) → Conv(64→128) → Conv(128→256) → Conv(256→512) → FC(512) → output  
**Purpose:** Deliberately overfit to confirm the architecture has enough capacity, and to observe overfitting behavior clearly.  
**Config:** 50 epochs, lr=0.001, batch=32, dropout=0.1, weight_decay=0.0, **no augmentation**  

**Result:** Train acc >> Val acc, large train/val gap — **clear overfitting**  
**Diagnosis:** Without augmentation and regularization, the large model memorizes training data. This is the expected and desired outcome for this experiment — it proves the model has capacity. The solution is to add regularization, not remove layers.

---

### V5 — CNN Best (final model)

**Architecture:** Conv(1→32) → Conv(32→64) → Conv(64→128) → Conv(128→256) → FC(512) → output  
**Purpose:** Balance capacity and regularization for best generalization.  
**Config:** 15 epochs (CPU-limited), lr=0.0005, batch=64, dropout=0.4, weight_decay=5e-4, label_smoothing=0.05, **with augmentation**  

**Result:** Best val accuracy **62.7%**, final diagnosis: **good_fit**  
Train/val gap: **0.013** (extremely tight — excellent generalization)

**Per-class performance:**
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Happy | 0.711 | 0.918 | 0.801 |
| Surprise | 0.734 | 0.729 | 0.732 |
| Neutral | 0.554 | 0.592 | 0.573 |
| Angry | 0.538 | 0.586 | 0.561 |
| Sad | 0.540 | 0.437 | 0.483 |
| Fear | 0.534 | 0.335 | 0.412 |
| Disgust | 0.773 | 0.195 | 0.312 |

**Diagnosis:** Happy and Surprise are easiest to classify (distinct features). Disgust is hardest — only 87 samples, severe class imbalance causes low recall. Fear and Sad are often confused with each other.

**Why this model was chosen:**
- Deeper than V2/V3 → more expressive features
- Dropout=0.4 + weight_decay + label_smoothing together prevent the overfitting seen in V4
- Data augmentation adds effective training variety
- Learning rate 0.0005 (lower than V2) allows more stable convergence

---

## Forward & Backward Checks

Before each training run, sanity checks are performed automatically:

**Forward check:**
- Output shape matches number of classes (7) ✅
- No NaN or Inf in logits ✅
- Probabilities sum to 1.0 ✅
- Logits are in a reasonable range (not collapsed) ✅

**Backward check:**
- All parameters receive gradients ✅
- No vanishing gradients ✅
- No exploding gradients ✅
- Grad norm reported per run (MLP: ~3.02, CNN: ~1.77–2.09, CNN Best: ~26.9 before clipping)

These checks catch bugs early — for example, a device mismatch bug (`cuda:0` vs `cpu`) was caught and fixed by the forward check failing before any training began.

---

## Key Findings

| Architecture | Best Val Acc | Fit Diagnosis | Key Insight |
|---|---|---|---|
| MLP (V1) | 44.0% | overfit | Spatial structure ignored |
| CNN Small (V2) | 56.8% | good_fit | Convolutions help significantly |
| CNN Medium (V3) | 54.3% | underfit | More depth needs better tuning |
| CNN Deep Overfit (V4) | high train, low val | overfit | Capacity confirmed, regularization needed |
| CNN Best (V5) | **62.7%** | good_fit | Best balance of depth + regularization |

**What caused overfitting:** Large model capacity without dropout/weight decay/augmentation (V4). MLP overfits because it memorizes noise patterns instead of learning structure.

**What caused underfitting:** Too much weight decay (V2 with wd=0.001), too few epochs relative to model complexity (V3).

**What worked best:** Moderate dropout (0.4) + weight decay (5e-4) + label smoothing (0.05) + data augmentation together, applied to a well-sized CNN.

---

## Hyperparameter Analysis

- **Learning rate:** 0.001 works for small models; 0.0005 better for larger ones (more stable)
- **Batch size:** 64 consistently outperformed 128 (more gradient updates per epoch)
- **Dropout:** 0.25–0.4 sweet spot; 0.5 hurts smaller models
- **Weight decay:** 1e-4 good baseline; 1e-3 causes underfitting
- **Label smoothing:** 0.05 slightly improves generalization by preventing overconfident predictions
- **Augmentation:** Always helped on CNN models; not used for intentional overfit experiment

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train a specific architecture
python train.py --train_csv train.csv --model cnn_best --epochs 45 \
  --lr 0.0005 --batch_size 64 --dropout 0.4 --weight_decay 0.0005 \
  --channels 32,64,128,256 --fc_dim 512 --use_augmentation --label_smoothing 0.05

# Run full experiment grid
python run_experiments.py --arch_key arch_v5_cnn_best --project fer-expression-recognition

# Generate submission
python predict.py --checkpoint checkpoints/cnn_best_best.pt --output submission.csv
```

Or use the provided `FER_Colab.ipynb` notebook on Google Colab.