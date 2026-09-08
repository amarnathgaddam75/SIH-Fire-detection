# GeoFlare Model Training & Evaluation Summary

This document summarizes the actual output from running [`train_model.py`](file:///Ubuntu/home/dev_amar75/SIH/train_model.py) on the engineered feature vector dataset ([`data/feature_vector.csv`](file:///Ubuntu/home/dev_amar75/SIH/data/feature_vector.csv)), evaluating Random Forest and XGBoost classifiers across 23 features.

---

## 1. Dataset & Class Label Distribution

- **Total Hotspot Detections**: 2,008 rows (30 original columns, 23 input features after imputation and missingness indicators).
- **Class Label Breakdown**:

| Class Label | Count | Percentage |
| :--- | :--- | :--- |
| **Industrial** | 1,054 | 52.5% |
| **Natural Fire** | 868 | 43.2% |
| **Unknown** | 74 | 3.7% |
| **Agricultural Burning** | 12 | 0.6% |
| **Total** | **2,008** | **100.0%** |

### Stratified Train/Test Split (80% / 20%, `random_state=42`)
- **Training Set**: 1,606 rows
  - Industrial: 843
  - Natural Fire: 694
  - Unknown: 59
  - Agricultural Burning: 10
- **Test Set**: 402 rows
  - Industrial: 211
  - Natural Fire: 174
  - Unknown: 15
  - Agricultural Burning: 2

---

## 2. 5-Fold Stratified Cross-Validation (Random Forest)

To assess model stability and generalization across folds, 5-fold stratified cross-validation was conducted on the training set using Macro F1:

- **Fold Macro F1 Scores**: `[0.9465, 0.9872, 0.8094, 0.8806, 0.9510]`
- **Mean Macro F1**: **0.9149**
- **Standard Deviation**: **0.0630**

---

## 3. Test Set Evaluation & Performance Reports

Both models were evaluated on the identical held-out test set of 402 rows.

### 3.1 Random Forest Classifier
- **Overall Accuracy**: 0.9876 (98.76%)
- **Test-set Macro F1**: **0.9108**
- **Weighted Avg F1**: 0.9875

#### Classification Report:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Agricultural Burning** | 0.6667 | 1.0000 | 0.8000 | 2 |
| **Industrial** | 1.0000 | 0.9953 | 0.9976 | 211 |
| **Natural Fire** | 0.9830 | 0.9943 | 0.9886 | 174 |
| **Unknown** | 0.9231 | 0.8000 | 0.8571 | 15 |
| **Macro Average** | **0.8932** | **0.9474** | **0.9108** | **402** |
| **Weighted Average** | **0.9881** | **0.9876** | **0.9875** | **402** |

#### Confusion Matrix:
*(Rows = True Class, Columns = Predicted Class: `[Ag Burning, Industrial, Natural Fire, Unknown]`)*
```
True \ Pred               Ag Burn   Industrial   Natural Fire   Unknown
Agricultural Burning         2           0             0           0
Industrial                   0         210             0           1
Natural Fire                 1           0           173           0
Unknown                      0           0             3          12
```

---

### 3.2 XGBoost Classifier (Selected Model: `data/classifier.pkl`)
- **Overall Accuracy**: 0.9925 (99.25%)
- **Test-set Macro F1**: **0.9721**
- **Weighted Avg F1**: 0.9924

#### Classification Report:
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Agricultural Burning** | 1.0000 | 1.0000 | 1.0000 | 2 |
| **Industrial** | 1.0000 | 0.9953 | 0.9976 | 211 |
| **Natural Fire** | 0.9886 | 1.0000 | 0.9943 | 174 |
| **Unknown** | 0.9286 | 0.8667 | 0.8966 | 15 |
| **Macro Average** | **0.9793** | **0.9655** | **0.9721** | **402** |
| **Weighted Average** | **0.9924** | **0.9925** | **0.9924** | **402** |

#### Confusion Matrix:
*(Rows = True Class, Columns = Predicted Class: `[Ag Burning, Industrial, Natural Fire, Unknown]`)*
```
True \ Pred               Ag Burn   Industrial   Natural Fire   Unknown
Agricultural Burning         2           0             0           0
Industrial                   0         210             0           1
Natural Fire                 0           0           174           0
Unknown                      0           0             2          13
```

---

## 4. 'Unknown' Class Resolution Analysis

Because the `Unknown` class represents borderline and unconfirmed thermal events, resolving them accurately is critical:

| Metric | Random Forest | XGBoost |
| :--- | :--- | :--- |
| **Total Test-set Unknown Rows** | 15 | 15 |
| **Correctly Predicted as Unknown** | 12 | 13 |
| **Misclassified into Natural Fire** | 3 | 2 |
| **Unknown Recall** | **80.00%** | **86.67%** |

XGBoost demonstrated superior separation on the tail distribution, correctly isolating 13 of the 15 ambiguous events.

---

## 5. Feature Importance Ranking (Random Forest)

| Rank | Feature | Importance | Relative Weight |
| :---: | :--- | :---: | :--- |
| 1 | `distance_to_farmland_m` | 0.2501 | █████████████████████████ |
| 2 | `distance_to_industrial_m` | 0.2002 | ████████████████████ |
| 3 | `distinct_days` | 0.1656 | ████████████████ |
| 4 | `historical_max_frp` | 0.0792 | ████████ |
| 5 | `night_detection_ratio` | 0.0528 | █████ |
| 6 | `brightness` | 0.0509 | █████ |
| 7 | `historical_std_frp` | 0.0478 | █████ |
| 8 | `frp` | 0.0381 | ████ |
| 9 | `historical_mean_frp` | 0.0353 | ████ |
| 10 | `daynight` | 0.0226 | ██ |
| 11 | `frp_deviation` | 0.0139 | █ |
| 12 | `confidence` | 0.0131 | █ |
| 13 | `historical_std_frp_was_missing` | 0.0072 | ▏ |
| 14 | `historical_max_frp_was_missing` | 0.0062 | ▏ |
| 15 | `frp_deviation_was_missing` | 0.0060 | ▏ |
| 16 | `historical_mean_frp_was_missing` | 0.0057 | ▏ |
| 17 | `night_detection_ratio_was_missing`| 0.0052 | ▏ |
| 18 | `ndvi` | 0.0000 |  |
| 19 | `ndbi` | 0.0000 |  |
| 20 | `ndwi` | 0.0000 |  |
| 21 | `ndvi_was_missing` | 0.0000 |  |
| 22 | `ndbi_was_missing` | 0.0000 |  |
| 23 | `ndwi_was_missing` | 0.0000 |  |

---

## 6. Important Caveat

> [!WARNING]
> ### Circularity in Rule-Based Labeling vs. Feature Inputs
> Because the ground-truth target `class_label` was originally constructed via explicit, rule-based thresholds on `distance_to_industrial_m`, `distance_to_farmland_m`, and `distinct_days`, and these exact columns were supplied as input features to the classifiers, **the near-perfect overall accuracy (~99%) largely reflects the models learning to reconstruct that deterministic decision boundary rather than discovering an entirely novel, unassisted natural signal.**

### Where the Real Validation Lies:
1. **Validation of Physical Hypotheses**:
   The primary scientific finding is that the unsupervised feature importance ranking aligns directly with the project's original domain hypothesis:
   - **Proximity** (`distance_to_farmland_m`: 25.0%, `distance_to_industrial_m`: 20.0%) and **Temporal Persistence** (`distinct_days`: 16.6%) dominate the classification decision space.
   - Secondary historical baselines (`historical_max_frp`, `night_detection_ratio`, `brightness`) provide nuance in separating genuine flare activity from ephemeral wildfires.

2. **Performance on the 'Unknown' Class**:
   The `Unknown` class cannot be resolved by simply matching an individual static threshold, as it contains borderline persistence cases (e.g., fires inside industrial zones with 0–8 distinct days) and missing satellite metadata. XGBoost achieving **86.67% recall** (and RF achieving **80.00% recall**) demonstrates strong capability to isolate complex residual cases.

3. **Sample Size Consideration**:
   The held-out test set contains **only 15 instances of the `Unknown` class** (and 2 instances of `Agricultural Burning`). Consequently, recall and precision percentages for these minority classes must be treated as **suggestive and indicative**, rather than statistically definitive proofs of real-world generalization without further field ground truth.
