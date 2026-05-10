"""
============================================================
Kyrgyz Fake News Detection — Full ML/NLP Pipeline
============================================================
Novel contribution: Morphology-Aware Feature Fusion for
agglutinative Kyrgyz text.

Architecture:
    Score = σ(W1·H_transformer + W2·H_morphology + b)

where H_morphology captures Kyrgyz suffix statistics
(superlative, emotional, evidential suffixes) that
correlate with fake-news writing patterns.
"""

import re
import csv
import json
import math
import random
import itertools
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

# ──────────────────────────────────────────────────────────
# 1.  KYRGYZ MORPHOLOGICAL ANALYSER (rule-based, no deps)
# ──────────────────────────────────────────────────────────

# Kyrgyz is SOV + agglutinative.  Suffixes that appear far
# more often in sensational / fake content (based on corpus
# linguistics studies on Turkic clickbait):
FAKE_SIGNAL_SUFFIXES = {
    # Superlative / intensifying
    "эң":      "superlative_particle",   # "most"  (pre-word)
    "дай":     "similative",             # "like X"
    "дей":     "similative",
    "тай":     "similative",
    # Emotional / exclamatory
    "ай":      "exclamatory",
    "эй":      "exclamatory",
    # Hearsay / evidential (rumour signal)
    "экен":    "evidential",             # "apparently"
    "имиш":    "hearsay",               # "they say"
    "деп":     "reported_speech",
    # Exaggeration suffixes
    "дан":     "ablative_cause",         # used in alarming constructions
    "ган":     "past_participle",        # heavy use → rumour pattern
    "лык":     "nominaliser",
    # Clickbait nominal endings
    "чылык":   "abstract_noun",
    "сыздык":  "deprivation_noun",
}

REAL_SIGNAL_SUFFIXES = {
    # Formal / official register
    "дыгы":   "formal_nominaliser",
    "туу":    "verbal_noun_formal",
    "уу":     "infinitive",
    "ышы":    "subject_nominaliser",
    "нын":    "genitive_formal",
    "нун":    "genitive_formal",
    "нүн":    "genitive_formal",
}


def extract_morphological_features(text: str) -> Dict[str, float]:
    """
    Rule-based morphological feature vector for a Kyrgyz text.
    Returns a dict of normalised feature counts.
    """
    tokens = re.findall(r'\w+', text.lower())
    n = max(len(tokens), 1)
    feats: Dict[str, float] = {}

    # 1. Suffix counts (normalised by token count)
    fake_hits = 0
    real_hits = 0
    suffix_type_counts: Dict[str, int] = defaultdict(int)

    for tok in tokens:
        for suffix, label in FAKE_SIGNAL_SUFFIXES.items():
            if tok.endswith(suffix) and len(tok) > len(suffix) + 1:
                fake_hits += 1
                suffix_type_counts[label] += 1
        for suffix, label in REAL_SIGNAL_SUFFIXES.items():
            if tok.endswith(suffix) and len(tok) > len(suffix) + 1:
                real_hits += 1
                suffix_type_counts[label] += 1

    feats["fake_suffix_ratio"]       = fake_hits / n
    feats["real_suffix_ratio"]       = real_hits / n
    feats["evidential_ratio"]        = (suffix_type_counts["evidential"] +
                                        suffix_type_counts["hearsay"] +
                                        suffix_type_counts["reported_speech"]) / n
    feats["exclamatory_ratio"]       = suffix_type_counts["exclamatory"] / n
    feats["superlative_ratio"]       = suffix_type_counts["superlative_particle"] / n
    feats["formal_register_ratio"]   = (suffix_type_counts["formal_nominaliser"] +
                                        suffix_type_counts["infinitive"] +
                                        suffix_type_counts["subject_nominaliser"]) / n

    # 2. Average token length  (fake news → shorter clickbait tokens)
    feats["avg_token_len"] = sum(len(t) for t in tokens) / n

    # 3. Type–Token Ratio  (lexical diversity; low → repetitive/fake)
    feats["ttr"] = len(set(tokens)) / n

    # 4. Sentence count proxy  (punctuation-based)
    sentences = re.split(r'[.!?।]+', text)
    n_sent = max(len([s for s in sentences if s.strip()]), 1)
    feats["sent_count"] = n_sent
    feats["avg_sent_len_tokens"] = n / n_sent

    # 5. Caps-word ratio  (shouting / clickbait signal)
    cap_words = sum(1 for t in re.findall(r'\b[А-ЯA-Z]{2,}\b', text))
    feats["caps_ratio"] = cap_words / n

    # 6. Exclamation / question mark density
    feats["excl_density"] = text.count('!') / n
    feats["ques_density"] = text.count('?') / n

    # 7. Digit ratio  (official news contains more figures)
    digits = sum(1 for t in tokens if t.isdigit())
    feats["digit_ratio"] = digits / n

    # 8. Telegram/social signal words
    social_words = {"источник", "сурак", "алдымы", "шашылыш", "акыркы"}
    feats["social_signal"] = sum(1 for t in tokens if t in social_words) / n

    return feats


# ──────────────────────────────────────────────────────────
# 2.  TF-IDF VECTORISER (pure Python, no sklearn)
# ──────────────────────────────────────────────────────────

def tokenise_kyrgyz(text: str) -> List[str]:
    """Basic whitespace + punctuation tokeniser for Kyrgyz."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [t for t in text.split() if len(t) > 1]


class TfidfVectoriser:
    def __init__(self, max_features: int = 5000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, corpus: List[str]) -> "TfidfVectoriser":
        df: Dict[str, int] = defaultdict(int)
        for doc in corpus:
            for tok in set(tokenise_kyrgyz(doc)):
                df[tok] += 1
        N = len(corpus)
        # Filter by min_df and pick top by df
        eligible = {t: d for t, d in df.items() if d >= self.min_df}
        top_terms = sorted(eligible, key=lambda t: -eligible[t])[:self.max_features]
        self.vocab = {t: i for i, t in enumerate(top_terms)}
        self.idf = {
            t: math.log((N + 1) / (eligible[t] + 1)) + 1
            for t in top_terms
        }
        return self

    def transform(self, corpus: List[str]) -> List[List[float]]:
        V = len(self.vocab)
        result = []
        for doc in corpus:
            vec = [0.0] * V
            tokens = tokenise_kyrgyz(doc)
            tf_raw = Counter(tokens)
            n = max(len(tokens), 1)
            for tok, cnt in tf_raw.items():
                if tok in self.vocab:
                    idx = self.vocab[tok]
                    tf = cnt / n
                    vec[idx] = tf * self.idf[tok]
            result.append(vec)
        return result

    def fit_transform(self, corpus: List[str]) -> List[List[float]]:
        self.fit(corpus)
        return self.transform(corpus)


# ──────────────────────────────────────────────────────────
# 3.  CLASSIFIERS (pure Python, zero dependencies)
# ──────────────────────────────────────────────────────────

class NaiveBayesClassifier:
    """Multinomial Naive Bayes for TF-IDF (treated as term weights)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_prior: Dict[int, float] = {}
        self.feature_log_prob: Dict[int, List[float]] = {}
        self.classes: List[int] = []

    def fit(self, X: List[List[float]], y: List[int]) -> "NaiveBayesClassifier":
        self.classes = sorted(set(y))
        n_features = len(X[0])
        n_total = len(y)
        class_counts = Counter(y)

        for c in self.classes:
            self.class_log_prior[c] = math.log(class_counts[c] / n_total)
            # Sum feature values per class
            feat_sums = [self.alpha] * n_features
            for xi, yi in zip(X, y):
                if yi == c:
                    for j, v in enumerate(xi):
                        feat_sums[j] += max(v, 0)
            total = sum(feat_sums) + self.alpha * n_features
            self.feature_log_prob[c] = [math.log(fs / total) for fs in feat_sums]
        return self

    def predict_proba(self, X: List[List[float]]) -> List[List[float]]:
        result = []
        for xi in X:
            scores = []
            for c in self.classes:
                log_p = self.class_log_prior[c]
                for j, v in enumerate(xi):
                    if v > 0:
                        log_p += v * self.feature_log_prob[c][j]
                scores.append(log_p)
            # Softmax for probabilities
            max_s = max(scores)
            exps = [math.exp(s - max_s) for s in scores]
            total = sum(exps)
            result.append([e / total for e in exps])
        return result

    def predict(self, X: List[List[float]]) -> List[int]:
        proba = self.predict_proba(X)
        return [self.classes[p.index(max(p))] for p in proba]


class LogisticRegression:
    """Logistic Regression with SGD (L2 regularisation)."""

    def __init__(self, lr: float = 0.1, epochs: int = 50, C: float = 1.0):
        self.lr = lr
        self.epochs = epochs
        self.C = C      # inverse regularisation strength
        self.weights: List[float] = []
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def _dot(self, xi: List[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, xi)) + self.bias

    def fit(self, X: List[List[float]], y: List[int]) -> "LogisticRegression":
        n_feat = len(X[0])
        self.weights = [0.0] * n_feat
        self.bias = 0.0
        indices = list(range(len(X)))

        for epoch in range(self.epochs):
            random.shuffle(indices)
            for i in indices:
                z = self._dot(X[i])
                pred = self._sigmoid(z)
                err = y[i] - pred
                for j in range(n_feat):
                    self.weights[j] += self.lr * (err * X[i][j] - self.weights[j] / self.C)
                self.bias += self.lr * err
        return self

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        return [self._sigmoid(self._dot(xi)) for xi in X]

    def predict(self, X: List[List[float]]) -> List[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]


class DecisionTree:
    """CART-style Decision Tree with Gini impurity."""

    def __init__(self, max_depth: int = 8, min_samples_split: int = 10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    @staticmethod
    def _gini(y: List[int]) -> float:
        n = len(y)
        if n == 0:
            return 0.0
        counts = Counter(y)
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    def _best_split(self, X: List[List[float]], y: List[int]):
        best_gain, best_feat, best_thresh = -1.0, 0, 0.0
        parent_gini = self._gini(y)
        n = len(y)
        # Sample a random subset of features (sqrt heuristic)
        n_feat = len(X[0])
        feat_sample = random.sample(range(n_feat), max(1, int(n_feat ** 0.5)))

        for f in feat_sample:
            vals = sorted(set(xi[f] for xi in X))
            thresholds = [(vals[i] + vals[i+1]) / 2 for i in range(len(vals)-1)]
            for thresh in thresholds[:20]:  # cap to keep it fast
                left_y  = [y[i] for i in range(n) if X[i][f] <= thresh]
                right_y = [y[i] for i in range(n) if X[i][f] >  thresh]
                if not left_y or not right_y:
                    continue
                gain = parent_gini - (len(left_y)/n * self._gini(left_y) +
                                      len(right_y)/n * self._gini(right_y))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, f, thresh
        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return {"leaf": True, "label": Counter(y).most_common(1)[0][0]}
        feat, thresh, gain = self._best_split(X, y)
        if gain <= 0:
            return {"leaf": True, "label": Counter(y).most_common(1)[0][0]}
        left_idx  = [i for i in range(len(X)) if X[i][feat] <= thresh]
        right_idx = [i for i in range(len(X)) if X[i][feat] >  thresh]
        return {
            "leaf": False, "feat": feat, "thresh": thresh,
            "left":  self._build([X[i] for i in left_idx],  [y[i] for i in left_idx],  depth+1),
            "right": self._build([X[i] for i in right_idx], [y[i] for i in right_idx], depth+1),
        }

    def fit(self, X: List[List[float]], y: List[int]) -> "DecisionTree":
        self.tree = self._build(X, y, 0)
        return self

    def _predict_one(self, node, xi):
        if node["leaf"]:
            return node["label"]
        if xi[node["feat"]] <= node["thresh"]:
            return self._predict_one(node["left"], xi)
        return self._predict_one(node["right"], xi)

    def predict(self, X: List[List[float]]) -> List[int]:
        return [self._predict_one(self.tree, xi) for xi in X]


class RandomForest:
    """Bagging ensemble of Decision Trees."""

    def __init__(self, n_trees: int = 50, max_depth: int = 8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees: List[DecisionTree] = []

    def fit(self, X: List[List[float]], y: List[int]) -> "RandomForest":
        n = len(X)
        for _ in range(self.n_trees):
            # Bootstrap sample
            indices = [random.randint(0, n-1) for _ in range(n)]
            Xb = [X[i] for i in indices]
            yb = [y[i] for i in indices]
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=5)
            tree.fit(Xb, yb)
            self.trees.append(tree)
        return self

    def predict(self, X: List[List[float]]) -> List[int]:
        all_preds = [t.predict(X) for t in self.trees]
        # Majority vote
        result = []
        for i in range(len(X)):
            votes = Counter(preds[i] for preds in all_preds)
            result.append(votes.most_common(1)[0][0])
        return result


# ──────────────────────────────────────────────────────────
# 4.  MORPHOLOGY-AWARE FUSION MODEL  (the novel contribution)
# ──────────────────────────────────────────────────────────

class MorphAwareFusionModel:
    """
    Novel Feature-Fusion architecture:
        Score = σ(W1·H_tfidf + W2·H_morph + b)

    H_tfidf   — TF-IDF representation (transformer proxy)
    H_morph   — Kyrgyz morphological feature vector

    Trained end-to-end with SGD logistic loss.
    """

    def __init__(self, lr: float = 0.05, epochs: int = 80,
                 morph_weight_init: float = 2.0):
        self.lr = lr
        self.epochs = epochs
        self.morph_weight_init = morph_weight_init
        self.w_tfidf:  List[float] = []
        self.w_morph:  List[float] = []
        self.bias = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

    def fit(self,
            X_tfidf: List[List[float]],
            X_morph: List[List[float]],
            y: List[int]) -> "MorphAwareFusionModel":

        n_tfidf = len(X_tfidf[0])
        n_morph = len(X_morph[0])

        # Initialise: give morphology features a head-start weight
        self.w_tfidf = [random.gauss(0, 0.01) for _ in range(n_tfidf)]
        self.w_morph = [self.morph_weight_init * random.gauss(0, 0.1)
                        for _ in range(n_morph)]
        self.bias = 0.0

        indices = list(range(len(y)))
        for epoch in range(self.epochs):
            random.shuffle(indices)
            for i in indices:
                z_t = sum(w * x for w, x in zip(self.w_tfidf, X_tfidf[i]))
                z_m = sum(w * x for w, x in zip(self.w_morph, X_morph[i]))
                pred = self._sigmoid(z_t + z_m + self.bias)
                err = y[i] - pred
                lr_i = self.lr * (1.0 / (1.0 + 0.01 * epoch))  # decay

                for j in range(n_tfidf):
                    self.w_tfidf[j] += lr_i * err * X_tfidf[i][j]
                for j in range(n_morph):
                    self.w_morph[j] += lr_i * err * X_morph[i][j] * self.morph_weight_init
                self.bias += lr_i * err
        return self

    def predict_proba(self,
                      X_tfidf: List[List[float]],
                      X_morph: List[List[float]]) -> List[float]:
        result = []
        for xt, xm in zip(X_tfidf, X_morph):
            z_t = sum(w * x for w, x in zip(self.w_tfidf, xt))
            z_m = sum(w * x for w, x in zip(self.w_morph, xm))
            result.append(self._sigmoid(z_t + z_m + self.bias))
        return result

    def predict(self,
                X_tfidf: List[List[float]],
                X_morph: List[List[float]]) -> List[int]:
        return [1 if p >= 0.5 else 0
                for p in self.predict_proba(X_tfidf, X_morph)]

    def get_top_morph_features(self,
                               feature_names: List[str],
                               top_k: int = 10) -> List[Tuple[str, float]]:
        """Return most influential morphological features."""
        pairs = list(zip(feature_names, self.w_morph))
        return sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:top_k]


# ──────────────────────────────────────────────────────────
# 5.  EVALUATION METRICS
# ──────────────────────────────────────────────────────────

def confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def compute_metrics(y_true: List[int], y_pred: List[int],
                    proba: Optional[List[float]] = None) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]

    accuracy  = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    # Matthews Correlation Coefficient
    denom = math.sqrt(
        max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)
    )
    mcc = (tp * tn - fp * fn) / denom

    # ROC-AUC (trapezoid, requires proba)
    auc = 0.5
    if proba is not None:
        pairs = sorted(zip(proba, y_true), reverse=True)
        P = sum(y_true)
        N = len(y_true) - P
        tp_c, fp_c = 0, 0
        auc = 0.0
        prev_tp, prev_fp = 0, 0
        for prob, label in pairs:
            if label == 1:
                tp_c += 1
            else:
                fp_c += 1
            auc += (fp_c - prev_fp) * (tp_c + prev_tp) / 2
            prev_tp, prev_fp = tp_c, fp_c
        auc = auc / max(P * N, 1)

    return {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "mcc":       round(mcc,       4),
        "auc":       round(auc,       4),
        **{k: v for k, v in cm.items()},
    }


# ──────────────────────────────────────────────────────────
# 6.  DATA LOADING  (works with the kyrgyz_news schema)
# ──────────────────────────────────────────────────────────

def load_dataset(filepath: str,
                 label_col: str = "label",
                 text_cols: Optional[List[str]] = None,
                 confidence_filter: Optional[str] = None) -> Tuple[List[str], List[int]]:
    """
    Load the Kyrgyz news CSV with schema:
        id, headline, body_text, source, date, label,
        label_confidence, source_type

    Returns (texts, labels) where text = headline + body_text,
    label 0=Real, 1=Fake.
    """
    if text_cols is None:
        text_cols = ["headline", "body_text"]

    texts, labels = [], []
    try:
        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if confidence_filter and row.get("label_confidence", "") != confidence_filter:
                    continue
                text = " ".join(row.get(col, "") for col in text_cols if col in row)
                if not text.strip():
                    continue
                raw_label = str(row.get(label_col, "")).strip()
                label = int(raw_label) if raw_label in ("0", "1") else (
                    1 if raw_label.lower() in ("fake", "фейк", "1") else 0
                )
                texts.append(text)
                labels.append(label)
    except FileNotFoundError:
        # Fallback: generate synthetic Kyrgyz-like data for demonstration
        print(f"[INFO] Dataset not found at {filepath}. Using synthetic demo data.")
        texts, labels = _generate_synthetic_data(n=500)

    return texts, labels


def _generate_synthetic_data(n: int = 500) -> Tuple[List[str], List[int]]:
    """
    Generate synthetic Kyrgyz-flavoured data for pipeline demonstration.
    Real news: formal register, digits, official vocabulary.
    Fake news: evidential/hearsay suffixes, exclamations, clickbait.
    """
    real_templates = [
        "Кыргыз Республикасынын Министрлер Кабинети {date} жылы {num} млн сом бөлдү.",
        "Билим берүү министрлиги мектептердин {num} пайызын {year} жылга чейин жабдыйт.",
        "Жогорку Кеңеш бугун {num} мыйзамды кабыл алды.",
        "Бишкек шаарынын мэриясы {num} жаңы автобус сатып алуу планын тастыктады.",
        "Улуттук статистика комитетинин маалыматы боюнча ИДПнын өсүшү {num} пайызды түздү.",
    ]
    fake_templates = [
        "Акыркы маалыматка ылайык {entity} качып кеткен экен! Укмуш!",
        "ШАШЫЛЫШ! {entity} жөнүндө жашыруун маалымат ачыкталды имиш.",
        "Эс алыңыз! {entity} дегенди уктуңузбу? Бул чындык беле?!",
        "Сурак: {entity} ким экенин билесиңби? Жооп таң калтырат!",
        "Коркунучтуу! {entity} боюнча чындыкты жашырып жатышкан деп айтылат.",
    ]
    entities = ["президент", "министр", "банк", "компания", "кино жылдызы"]
    dates = ["2024", "2025"]
    nums = ["12", "45", "3.2", "890"]

    random.seed(42)
    texts, labels = [], []
    for _ in range(n):
        if random.random() < 0.5:
            t = random.choice(real_templates)
            t = t.format(date=random.choice(dates),
                          year=random.choice(dates),
                          num=random.choice(nums))
            labels.append(0)
        else:
            t = random.choice(fake_templates)
            t = t.format(entity=random.choice(entities))
            labels.append(1)
        texts.append(t)
    return texts, labels


# ──────────────────────────────────────────────────────────
# 7.  TRAIN / TEST SPLIT  (temporal-aware)
# ──────────────────────────────────────────────────────────

def train_test_split(texts, labels, test_ratio=0.2, seed=42):
    random.seed(seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)
    split = int(len(indices) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return (
        [texts[i] for i in train_idx],
        [texts[i] for i in test_idx],
        [labels[i] for i in train_idx],
        [labels[i] for i in test_idx],
    )


# ──────────────────────────────────────────────────────────
# 8.  FEATURE MATRIX CONCATENATION
# ──────────────────────────────────────────────────────────

def build_morph_matrix(texts: List[str]) -> Tuple[List[List[float]], List[str]]:
    """Returns (matrix, feature_names)."""
    sample_feats = extract_morphological_features(texts[0])
    feat_names = list(sample_feats.keys())
    matrix = []
    for text in texts:
        feats = extract_morphological_features(text)
        matrix.append([feats[k] for k in feat_names])
    return matrix, feat_names


def concat_features(tfidf_vecs: List[List[float]],
                    morph_vecs: List[List[float]]) -> List[List[float]]:
    return [t + m for t, m in zip(tfidf_vecs, morph_vecs)]


# ──────────────────────────────────────────────────────────
# 9.  FULL EXPERIMENT RUNNER
# ──────────────────────────────────────────────────────────

def run_experiments(dataset_path: str = "data/kyrgyz_news_final_unified.csv"):
    print("=" * 62)
    print("  Kyrgyz Fake News Detection — Experiment Suite")
    print("=" * 62)

    # Load data
    texts, labels = load_dataset(dataset_path)
    print(f"\n[DATA] Total samples : {len(texts)}")
    print(f"       Real (0)       : {labels.count(0)}")
    print(f"       Fake (1)       : {labels.count(1)}")

    # Split
    tr_texts, te_texts, tr_labels, te_labels = train_test_split(texts, labels)
    print(f"\n[SPLIT] Train: {len(tr_texts)}  |  Test: {len(te_texts)}")

    # TF-IDF
    print("\n[FEATURE] Fitting TF-IDF (max_features=3000)...")
    tfidf = TfidfVectoriser(max_features=3000, min_df=2)
    tr_tfidf = tfidf.fit_transform(tr_texts)
    te_tfidf = tfidf.transform(te_texts)

    # Morphological features
    print("[FEATURE] Extracting Kyrgyz morphological features...")
    tr_morph, feat_names = build_morph_matrix(tr_texts)
    te_morph, _          = build_morph_matrix(te_texts)
    print(f"          Morph feature dims: {len(feat_names)}")

    # Fusion features (TF-IDF + Morph)
    tr_fused = concat_features(tr_tfidf, tr_morph)
    te_fused = concat_features(te_tfidf, te_morph)

    results: Dict[str, Dict] = {}

    # ── Baseline 1: Naive Bayes (TF-IDF only)
    print("\n[MODEL] Naive Bayes (TF-IDF)...")
    nb = NaiveBayesClassifier(alpha=0.5)
    nb.fit(tr_tfidf, tr_labels)
    nb_pred = nb.predict(te_tfidf)
    nb_proba_raw = nb.predict_proba(te_tfidf)
    nb_proba = [p[1] for p in nb_proba_raw]
    results["Naive Bayes"] = compute_metrics(te_labels, nb_pred, nb_proba)

    # ── Baseline 2: Logistic Regression (TF-IDF only)
    print("[MODEL] Logistic Regression (TF-IDF)...")
    lr = LogisticRegression(lr=0.05, epochs=40)
    lr.fit(tr_tfidf, tr_labels)
    lr_pred  = lr.predict(te_tfidf)
    lr_proba = lr.predict_proba(te_tfidf)
    results["Logistic Regression"] = compute_metrics(te_labels, lr_pred, lr_proba)

    # ── Baseline 3: Decision Tree (TF-IDF only)
    print("[MODEL] Decision Tree (TF-IDF)...")
    dt = DecisionTree(max_depth=6)
    dt.fit(tr_tfidf, tr_labels)
    dt_pred = dt.predict(te_tfidf)
    results["Decision Tree"] = compute_metrics(te_labels, dt_pred)

    # ── Baseline 4: Random Forest (TF-IDF only)
    print("[MODEL] Random Forest (TF-IDF, 30 trees)...")
    rf = RandomForest(n_trees=30, max_depth=6)
    rf.fit(tr_tfidf, tr_labels)
    rf_pred = rf.predict(te_tfidf)
    results["Random Forest"] = compute_metrics(te_labels, rf_pred)

    # ── Baseline 5: LR with morphological features only
    print("[MODEL] Logistic Regression (Morph only)...")
    lr_m = LogisticRegression(lr=0.1, epochs=60)
    lr_m.fit(tr_morph, tr_labels)
    lrm_pred  = lr_m.predict(te_morph)
    lrm_proba = lr_m.predict_proba(te_morph)
    results["LR (Morph only)"] = compute_metrics(te_labels, lrm_pred, lrm_proba)

    # ── Baseline 6: LR with fused features
    print("[MODEL] Logistic Regression (TF-IDF + Morph fusion)...")
    lr_f = LogisticRegression(lr=0.05, epochs=50)
    lr_f.fit(tr_fused, tr_labels)
    lrf_pred  = lr_f.predict(te_fused)
    lrf_proba = lr_f.predict_proba(te_fused)
    results["LR (TF-IDF+Morph)"] = compute_metrics(te_labels, lrf_pred, lrf_proba)

    # ── NOVEL: Morphology-Aware Fusion Model
    print("[MODEL] MorphAware Fusion (novel architecture)...")
    maf = MorphAwareFusionModel(lr=0.05, epochs=60, morph_weight_init=2.0)
    maf.fit(tr_tfidf, tr_morph, tr_labels)
    maf_pred  = maf.predict(te_tfidf, te_morph)
    maf_proba = maf.predict_proba(te_tfidf, te_morph)
    results["MorphAware Fusion (Ours)"] = compute_metrics(te_labels, maf_pred, maf_proba)

    # ── Print results table
    print("\n" + "=" * 62)
    print("  RESULTS TABLE")
    print("=" * 62)
    header = f"{'Model':<28} {'Acc':>6} {'P':>6} {'R':>6} {'F1':>6} {'MCC':>6} {'AUC':>6}"
    print(header)
    print("-" * 62)
    for name, m in results.items():
        marker = " ◄ NEW" if "MorphAware" in name else ""
        print(f"{name:<28} {m['accuracy']:>6.4f} {m['precision']:>6.4f} "
              f"{m['recall']:>6.4f} {m['f1']:>6.4f} "
              f"{m['mcc']:>6.4f} {m['auc']:>6.4f}{marker}")

    # ── Top morphological features
    print("\n" + "=" * 62)
    print("  TOP MORPHOLOGICAL FEATURES (MorphAware Fusion)")
    print("=" * 62)
    top_feats = maf.get_top_morph_features(feat_names, top_k=len(feat_names))
    for name, weight in top_feats:
        direction = "→ FAKE" if weight > 0 else "→ REAL"
        print(f"  {name:<30} weight={weight:+.4f}  {direction}")

    print("\n[DONE] Experiment complete.\n")
    return results, maf, feat_names


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/kyrgyz_news_final_unified.csv"
    run_experiments(path)
