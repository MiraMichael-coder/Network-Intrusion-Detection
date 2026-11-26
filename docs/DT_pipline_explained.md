# Decision Tree (C2) — Explained

This document explains why we built a Decision Tree for the UNSW-NB15 network intrusion detection project, **what files were changed and added**, **what every block of code does**, and **what the results mean**.
---
## 1. Why We Used a Decision Tree (And Why It Matters for RQ2)

Our project's Research Question 2 (**RQ2**) asks:
> *"Which traffic features contribute most to detection, and how much performance do we give up for a model we can read?"*

- **Logistic Regression (Mira's model)** gives one weight per feature, which is mathematical rather than a step-by-step decision rule.
- **Random Forest (Maya's model)** and **SVM (Menna's model)** are "black-box" models—they are hard for a human security analyst to trace by hand.
- **Decision Tree (your model)** works like a flowchart of simple yes/no questions (for example: *"Is `sttl <= 61`? If yes, check `sinpkt`..."*). Anyone can read a small Decision Tree from top to bottom without needing special explanation tools.

In **C2**, we build two versions of the Decision Tree to compare:
1. **Default Decision Tree (`dt_baseline`):** No depth limit → shows **overfitting** (memorizing training data) and becomes unreadable.
2. **Shallow Decision Tree (`dt_shallow_d4`, `max_depth=4`):** Limited to 4 levels of questions → shows **how much performance we keep** when we force the model to stay readable by humans.

---

## 2. Files Changed & Added (And the Idea Behind Each)

### A. Files Modified (Safely, Without Breaking Teammates' Code)

#### 1. `src/preprocessing.py` — Added `scale=True` (optional)
- **Before:** `create_preprocessor(numerical_columns)` always applied `StandardScaler()` to all numeric columns.
- **Change:**
  ```python
  def create_preprocessor(numerical_columns, scale=True):
      categorical_transformer = OneHotEncoder(handle_unknown="ignore")
      numerical_transformer = StandardScaler() if scale else "passthrough"
  ```
- **Why this was needed:**
  - `StandardScaler()` converts real values (like `sbytes = 512` bytes or `sttl = 62`) into z-scores (like `-0.18`).
  - Decision Trees only compare one feature at a time against a threshold, so scaling does **not** improve accuracy or F1—it only makes tree diagrams harder for humans to read.
  - Passing `scale=False` uses `"passthrough"`, keeping numbers in their **real network units** (`sttl <= 61.0`, `sbytes <= 201.0`).
- **Why it does not affect Mira or Menna:** Because `scale=True` is the **default**, any existing code that calls `create_preprocessor(num_cols)` (like LR and the dummy baseline) still runs `StandardScaler()` identically.

#### 2. `src/models.py` — Registered `"dt"` with `scale=False`
- **Change:**
  - Imported `DecisionTreeClassifier`.
  - Added `"dt": lambda: DecisionTreeClassifier(random_state=SEED)` to `MODELS`.
  - Updated `make_pipeline(name, X, scale=None)` so when `scale` is not specified, `scale = name not in {"dt", "rf"}`.
- **Why this was needed:**
  - `lambda:` ensures a **brand-new unfitted tree** is created every time `make_pipeline("dt", X)` is called.
  - `random_state=SEED` (`42`) makes tie-breaks repeatable across runs.
  - Automatic `scale=False` for `"dt"` (and later `"rf"`) means `make_pipeline("dt", X)` and `src/oof.py` automatically skip scaling without touching `"lr"`.

---

### B. Files Added & Updated by Your C2 Run

| File | Why it exists | What happened when we ran C2 |
|---|---|---|
| `notebooks/04_dt.ipynb` | Your main Decision Tree notebook | Built and ran all 4 code cells cleanly |
| `results/metrics.csv` | Shared scoreboard of all models | Appended `dt_baseline` and `dt_shallow_d4` (`split="cv"`) rows |
| `results/figures/dt_shallow_d4.png` | Visual flowchart of the `max_depth=4` tree | Saved a high-resolution PNG diagram for your report and slides |

*(Note: `tests/test_evaluate.py` and a pipeline sanity check were also run after editing `src/` and passed: `"lr"` kept `StandardScaler()`, `"dt"` used `passthrough`.)*

---

## 3. Step-by-Step Walkthrough of `notebooks/04_dt.ipynb`

### Code Cell 1 — Load Data & Separate Features (`X`) from Target (`y`)
- **What it does:** Imports `load_train`, `cv_evaluate`, `make_pipeline`, and `EXCLUDED_COLS`, loads `UNSW_NB15_training-set.csv`, drops `["id", "attack_cat", "label"]` to form `X`, and puts `label` (`0` = Normal, `1` = Attack) into `y`.
- **Why:** Dropping `id` (row number) and `attack_cat` (attack name) prevents **data leakage** (cheating).
- **Output:** `X shape: (175341, 42) | y shape: (175341,)` (175,341 traffic flows, 42 input features).

---

### Code Cell 2 — Baseline Decision Tree (`dt_baseline`: Overfitting Check)
- **What the code does:**
  1. `dt_pipe = make_pipeline("dt", X)` creates an unfitted pipeline (`OneHotEncoder` + `"passthrough"` + default `DecisionTreeClassifier(random_state=SEED)`).
  2. `dt_cv = cv_evaluate("dt_baseline", dt_pipe, X, y)` runs **5-fold stratified cross-validation** (`StratifiedKFold(5, shuffle=True, random_state=42)`), fitting on 4/5 of the data and testing on the held-out 1/5 five times, and saves the average scores to `results/metrics.csv`.
  3. `dt_pipe.fit(X, y)` fits the tree on all training rows, and `f1_score(y, dt_pipe.predict(X))` checks how well it scores on the **same rows it just memorized** (`Train F1`).
- **What came out:**
  - **Train F1:** `0.9987` (99.87%)
  - **CV F1:** `0.9618 ± 0.0011`
  - **Overfitting gap (`Train - CV`):** `0.0369`
  - **Tree depth:** `43` levels | **Leaves:** `5,669` final boxes
  - **Other CV metrics:** Accuracy `0.9480`, Precision `0.9629`, Recall `0.9606`, **FPR `0.0788`** (7.9% false alarms), ROC-AUC `0.9425`, PR-AUC `0.9531`, Fit time `8.91 s`.
- **What this means:** With no depth limit, the tree kept splitting until it grew 43 levels deep with 5,669 rules, memorizing training details (`0.9987` Train F1) that don't generalize as well to unseen validation folds (`0.9618` CV F1). It is accurate, but completely unreadable by a human.

---

### Code Cell 3 — Readable Shallow Tree (`dt_shallow_d4`, `max_depth=4`)
- **What the code does:**
  1. Creates the same pipeline with `model__max_depth=4` (limiting the flowchart to at most 4 questions deep, i.e. at most $2^4 = 16$ final decision boxes).
  2. Calls `cv_evaluate("dt_shallow_d4", shallow_pipe, X, y)` to log its honest 5-fold CV performance to `results/metrics.csv` (needed later for **C6**).
  3. Fits `shallow_pipe.fit(X, y)` on training data, extracts the fitted preprocessor (`prep`) and tree (`tree`), gets expanded feature names with `prep.get_feature_names_out()`, and draws the diagram with `plot_tree(..., filled=True)` into `results/figures/dt_shallow_d4.png`.
- **What came out:**
  - **CV F1:** `0.9514 ± 0.0005`
  - **CV Accuracy:** `0.9304`
  - **CV Precision:** `0.9077`
  - **CV Recall:** `0.9995` (catches 99.95% of attacks!)
  - **CV FPR:** `0.2166` (21.7% false alarms)
  - **CV ROC-AUC:** `0.9799` | **PR-AUC:** `0.9857` | **Fit time:** `0.97 s`
- **Note on `UserWarning: FigureCanvasAgg is non-interactive`:** `src/evaluate.py` sets `matplotlib.use("Agg")` so plots save cleanly to `.png` files without popping up desktop windows; that is why the diagram was saved to `results/figures/dt_shallow_d4.png`.

---

### Code Cell 4 — Printing Plain-Text Decision Rules (`export_text`)
- **What the code does:**
  `rules = export_text(tree, feature_names=list(feature_names))` converts the already-trained `max_depth=4` tree into readable `|--- if/else` text lines.
- **How to read each box in `dt_shallow_d4.png`:**
  - **Split rule** (e.g. `num__sttl <= 61.0`): The yes/no question. Left arrow (`True`) means `<= 61.0`; right arrow (`False`) means `> 61.0`.
  - **`gini`**: How mixed the box is (`0.0` = 100% one class, `0.5` = 50/50 mix).
  - **`samples`**: Total number of training flows in that box.
  - **`value = [normal_count, attack_count]`**: Exact count of Normal vs. Attack flows.
  - **`class`**: Majority prediction (`Normal` = orange, `Attack` = blue; darker color = purer).
- **Key insight from the root split:**
  - The very first question is **`num__sttl <= 61.0`** (`sttl` = Source-to-Destination Time To Live).
  - Left side (`sttl <= 61.0`): `42,386` Normal vs. only `303` Attack flows.
  - Right side (`sttl > 61.0`): `119,038` Attack vs. `13,614` Normal flows.
  - Because `scale=False` was used, every threshold (`sttl <= 61.0`, `sbytes <= 201.0`, `smean <= 54.5`, `ct_srv_dst <= 2.5`) is directly readable in real units.

---

## 4. One-Paragraph Summary for Your Team / Presentation

> *"For C2, we added an optional `scale=True` default to `src/preprocessing.py` and registered `'dt'` in `src/models.py` with `scale=False` so tree rules stay in real network units without affecting Logistic Regression. Our default Decision Tree (`dt_baseline`) grew to depth 43 with 5,669 leaves and overfit (`0.9987` Train F1 vs. `0.9618` CV F1). Constraining the tree to `max_depth=4` (`dt_shallow_d4`) produced a completely human-readable flowchart led by `sttl <= 61.0` and `sbytes <= 201.0` while keeping `0.9514` CV F1 and `0.9995` recall—losing only ~1% F1 for full interpretability, at the trade-off of higher false alarms (`21.7%` FPR vs. `7.9%`)."*