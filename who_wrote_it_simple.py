# Who Wrote It? - figure out which LLM wrote a response
# Steps: 1) load data  2) split  3) CNN + LSTM  4) run RQ1-RQ4  5) save results
# Install first: pip install torch pandas scikit-learn matplotlib

import os, re, json, copy, random
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import urllib.request
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- settings (change these if you want) ----------------
DATASET_URL = "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/resolve/main/train.csv"
RAW_FILE = "arena_raw.csv"   # the raw download is saved here
DATA_FILE = None          # put a csv path here to use your own data instead
CACHE_FILE = "data.csv"   # processed data is saved here so we only download once
OUT = "results"
EPOCHS = 20              # max epochs (training stops early if it stops improving)
PATIENCE = 4             # stop after this many epochs with no improvement
BATCH = 32
LR = 0.001
MAX_IN = 100              # max words kept from the prompt
MAX_OUT = 500             # max words kept from the response
MIN_PER_CLASS = 2000      # ignore LLM families with fewer examples than this (fewer, bigger classes = easier)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)

# ---------------- 1. load the data ----------------
# model name starts with -> LLM family
FAMILIES = {"gpt-": "gpt", "chatgpt": "gpt", "claude": "claude", "gemini": "gemini",
            "palm": "gemini", "llama-": "llama", "mistral-": "mistral", "mixtral-": "mistral",
            "qwen": "qwen", "deepseek": "deepseek", "yi-": "yi"}


def get_family(model_name):
    for prefix in FAMILIES:
        if model_name.lower().startswith(prefix):
            return FAMILIES[prefix]
    return None


def first_turn(text):
    # prompts/responses are stored as JSON lists (one item per turn), we keep the first turn
    try:
        items = json.loads(text)
    except Exception:
        return None
    if isinstance(items, list) and len(items) > 0 and isinstance(items[0], str):
        return items[0]
    return None


def mostly_english(text):
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / max(len(text), 1) > 0.95


def guess_task(prompt, output):
    # simple keyword rules to label each example (needed for RQ3)
    p = prompt.lower()
    if "```" in output or re.search(r"\b(python|javascript|java|sql|function|code|script|html|css|bug)\b", p):
        return "coding"
    if re.search(r"\b(solve|calculate|equation|integral|probability|how many|sum of|prove)\b", p):
        return "math"
    if re.search(r"\b(write|story|poem|essay|song|letter|email|rewrite|translate|roleplay)\b", p):
        return "writing"
    return "qa"


def load_data():
    if DATA_FILE:
        return pd.read_csv(DATA_FILE)
    if os.path.exists(CACHE_FILE):
        return pd.read_csv(CACHE_FILE)
    if not os.path.exists(RAW_FILE):
        print("downloading dataset (about 184 MB)...")
        urllib.request.urlretrieve(DATASET_URL, RAW_FILE)
    raw = pd.read_csv(RAW_FILE)
    rows = []
    for _, r in raw.iterrows():
        prompt = first_turn(r["prompt"])
        if prompt is None:
            continue
        for side in ["a", "b"]:  # each row has two models: a and b
            family = get_family(r["model_" + side])
            output = first_turn(r["response_" + side])
            if family and output:
                rows.append([family, prompt, output])
    df = pd.DataFrame(rows, columns=["llm_name", "llm_input", "llm_output"])
    df = df[df.llm_output.str.len().between(50, 6000)]
    df = df[df.llm_input.apply(mostly_english) & df.llm_output.apply(mostly_english)]
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df["task"] = [guess_task(p, o) for p, o in zip(df.llm_input, df.llm_output)]
    df.to_csv(CACHE_FILE, index=False)
    return df


def balance(df):
    # same number of examples for every LLM
    smallest = df.llm_name.value_counts().min()
    parts = [g.sample(smallest, random_state=SEED) for _, g in df.groupby("llm_name")]
    return pd.concat(parts).reset_index(drop=True)


def split_by_prompt(df, test_size):
    # same prompt never ends up in both train and test (avoids cheating)
    groups = df.llm_input.factorize()[0]
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED).split(df, groups=groups))
    return df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)


# ---------------- 2. turn text into numbers ----------------
def tokenize(text):
    return re.findall(r"\n|\w+|[^\w\s]", text)  # words, punctuation and line breaks


def get_tokens(df, mode, clean=None):
    # mode: "input", "output" or "both"
    all_tokens = []
    for inp, out in zip(df.llm_input, df.llm_output):
        if clean:
            inp, out = clean(inp), clean(out)
        inp_tokens = tokenize(inp)[:MAX_IN]
        full_out = tokenize(out)
        # response length is a strong clue, but cutting the text hides it,
        # so we add a token that says how long the full response was
        length_token = "<len" + str(min(len(full_out) // 50, 30)) + ">"
        out_tokens = [length_token] + full_out[:MAX_OUT]
        if mode == "input":
            all_tokens.append(inp_tokens)
        elif mode == "output":
            all_tokens.append(out_tokens)
        else:
            all_tokens.append(inp_tokens + ["<sep>"] + out_tokens)
    return all_tokens


def build_vocab(token_lists):
    counts = Counter(t for tokens in token_lists for t in tokens)
    vocab = {"<pad>": 0, "<unk>": 1, "<sep>": 2}
    for word, c in counts.most_common(30000):
        if c >= 2:
            vocab[word] = len(vocab)
    return vocab


def to_tensor(token_lists, vocab):
    length = max(5, max(len(t) for t in token_lists))
    X = torch.zeros(len(token_lists), length, dtype=torch.long)
    lengths = torch.ones(len(token_lists), dtype=torch.long)
    for i, tokens in enumerate(token_lists):
        ids = [vocab.get(t, 1) for t in tokens] or [1]
        X[i, :len(ids)] = torch.tensor(ids)
        lengths[i] = len(ids)
    return X, lengths


# ---------------- 3. the two models ----------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.embed_dropout = nn.Dropout(0.2)
        # four filter sizes look at 2, 3, 4 and 5 words at a time
        self.convs = nn.ModuleList([nn.Conv1d(128, 256, k) for k in [2, 3, 4, 5]])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256 * 4, n_classes)

    def forward(self, x, lengths):
        e = self.embed_dropout(self.embed(x)).transpose(1, 2)
        pooled = [torch.relu(conv(e)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x, lengths):
        # "pack" so the LSTM ignores the padding
        packed = nn.utils.rnn.pack_padded_sequence(self.embed(x), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # use the output at every word (not only the last one), ignoring padding
        lengths = lengths.to(out.device)
        mask = (torch.arange(out.size(1), device=out.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(2)
        average = (out * mask).sum(dim=1) / lengths.unsqueeze(1)
        biggest = out.masked_fill(~mask, -1e9).max(dim=1).values
        return self.fc(self.dropout(torch.cat([average, biggest], dim=1)))


def predict(model, X, lengths):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out = model(X[i:i + 256].to(device), lengths[i:i + 256])
            preds.append(out.argmax(1).cpu())
    return torch.cat(preds).numpy()


def train_and_test(kind, train, val, test, classes, mode, clean=None):
    # kind: "cnn" or "lstm"
    train_tokens = get_tokens(train, mode, clean)
    vocab = build_vocab(train_tokens)  # vocab only from training data
    Xtr, Ltr = to_tensor(train_tokens, vocab)
    Xva, Lva = to_tensor(get_tokens(val, mode, clean), vocab)
    Xte, Lte = to_tensor(get_tokens(test, mode, clean), vocab)
    ytr = torch.tensor([classes.index(c) for c in train.llm_name])
    yva = np.array([classes.index(c) for c in val.llm_name])
    yte = np.array([classes.index(c) for c in test.llm_name])

    if kind == "cnn":
        model = TextCNN(len(vocab), len(classes)).to(device)
    else:
        model = BiLSTM(len(vocab), len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1
    best_weights = None
    bad_epochs = 0
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(Xtr))
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            out = model(Xtr[idx].to(device), Ltr[idx])
            loss = loss_fn(out, ytr[idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stops training from blowing up
            optimizer.step()
        val_acc = accuracy_score(yva, predict(model, Xva, Lva))
        print(f"   {kind} epoch {epoch + 1}/{EPOCHS} val acc: {val_acc:.3f}")
        if val_acc > best_acc:  # keep the best version of the model
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("   no improvement, stopping early")
                break

    model.load_state_dict(best_weights)
    preds = predict(model, Xte, Lte)
    return accuracy_score(yte, preds), f1_score(yte, preds, average="macro"), yte, preds


def save_confusion(y, preds, classes, name):
    cm = confusion_matrix(y, preds, labels=range(len(classes)))
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{OUT}/cm_{name}.png")
    plt.close()


def get_text(df, mode):
    if mode == "input":
        return df.llm_input
    if mode == "output":
        return df.llm_output
    return df.llm_input + " [SEP] " + df.llm_output


# ---------------- style features for RQ4 ----------------
def style_features(t):
    words = re.findall(r"\w+", t)
    n_words = max(len(words), 1)
    n_chars = max(len(t), 1)
    lines = t.split("\n")
    sentences = [s for s in re.split(r"[.!?]+\s", t) if s.strip()]
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    bullets = sum(1 for l in lines if re.match(r"\s*([-*•]|\d+[.)])\s", l))
    return {
        "len_chars": len(t), "len_words": len(words),
        "avg_sent_len": n_words / max(len(sentences), 1),
        "avg_word_len": avg_word_len,
        "ttr": len(set(w.lower() for w in words)) / n_words,  # vocabulary variety
        "n_lines": len(lines),
        "n_paragraphs": len([p for p in t.split("\n\n") if p.strip()]),
        "n_bullets": bullets,
        "n_headers": sum(1 for l in lines if l.lstrip().startswith("#")),
        "n_bold": t.count("**"),
        "n_codeblocks": t.count("```") // 2,
        "comma": t.count(",") / n_chars, "period": t.count(".") / n_chars,
        "question": t.count("?") / n_chars, "exclaim": t.count("!") / n_chars,
        "colon": t.count(":") / n_chars, "semicolon": t.count(";") / n_chars,
        "dash": (t.count("—") + t.count("–")) / n_chars, "paren": t.count("(") / n_chars,
        "quote": t.count('"') / n_chars,
        "upper_ratio": sum(1 for c in t if c.isupper()) / n_chars,
        "digit_ratio": sum(1 for c in t if c.isdigit()) / n_chars,
        "nonascii": sum(1 for c in t if ord(c) > 127) / n_chars,
    }


FEATURE_GROUPS = {
    "length": ["len_chars", "len_words", "avg_sent_len"],
    "vocabulary": ["avg_word_len", "ttr"],
    "formatting": ["n_lines", "n_paragraphs", "n_bullets", "n_headers", "n_bold", "n_codeblocks"],
    "punctuation": ["comma", "period", "question", "exclaim", "colon", "semicolon", "dash", "paren", "quote"],
    "characters": ["upper_ratio", "digit_ratio", "nonascii"],
}


def remove_formatting(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"[*#`>_~\-•]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def remove_formatting_and_lowercase(t):
    return remove_formatting(t).lower()


# ---------------- run everything ----------------
df = load_data()
print("examples per LLM before filtering:")
print(df.llm_name.value_counts())
counts = df.llm_name.value_counts()
df = df[df.llm_name.isin(counts[counts >= MIN_PER_CLASS].index)]  # drop tiny classes
df = balance(df)
classes = sorted(df.llm_name.unique())
print("classes:", classes)
print("chance accuracy (random guessing):", round(1 / len(classes), 3))

rest_split = split_by_prompt(df, 0.30)
train = rest_split[0]
val, test = split_by_prompt(rest_split[1], 0.50)
print("train/val/test:", len(train), len(val), len(test))

results = []

# ---- RQ1 + RQ2: output only, input only, input + output ----
for mode in ["output", "input", "both"]:
    print("\n== RQ1/RQ2, mode:", mode)
    # simple baseline (not a neural net) for comparison
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")
    clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform(get_text(train, mode)), train.llm_name)
    p = clf.predict(vec.transform(get_text(test, mode)))
    results.append({"exp": "RQ1/2", "model": "tfidf-baseline", "setting": mode,
                    "acc": accuracy_score(test.llm_name, p), "f1": f1_score(test.llm_name, p, average="macro")})
    for kind in ["cnn", "lstm"]:
        acc, f1, y, preds = train_and_test(kind, train, val, test, classes, mode)
        results.append({"exp": "RQ1/2", "model": kind, "setting": mode, "acc": acc, "f1": f1})
        save_confusion(y, preds, classes, kind + "_" + mode)

# ---- RQ3: train on some tasks, test on a task the model never saw ----
if "task" in df.columns and df.task.nunique() > 1:
    for task in sorted(df.task.unique()):
        print("\n== RQ3, held-out task:", task)
        test_t = df[df.task == task].reset_index(drop=True)
        others = df[df.task != task].reset_index(drop=True)
        train_t, val_t = split_by_prompt(others, 0.15)
        for kind in ["cnn", "lstm"]:
            acc, f1, y, preds = train_and_test(kind, train_t, val_t, test_t, classes, "output")
            results.append({"exp": "RQ3", "model": kind, "setting": "heldout_" + task, "acc": acc, "f1": f1})
else:
    print("\nRQ3 skipped: no 'task' column")

# ---- RQ4: what makes the LLMs different? ----
print("\n== RQ4")
Ftrain = pd.DataFrame([style_features(t) for t in train.llm_output])
Ftest = pd.DataFrame([style_features(t) for t in test.llm_output])
all_cols = list(Ftrain.columns)

# average of each feature for each LLM
Ftrain.assign(llm=train.llm_name.values).groupby("llm").mean().T.to_csv(f"{OUT}/rq4_feature_means.csv")

# random forest on style features only
rf = RandomForestClassifier(300, random_state=SEED, n_jobs=-1).fit(Ftrain, train.llm_name)
full_acc = accuracy_score(test.llm_name, rf.predict(Ftest))
results.append({"exp": "RQ4", "model": "random-forest", "setting": "all_features", "acc": full_acc})
importance = pd.Series(rf.feature_importances_, index=all_cols).sort_values(ascending=False)
importance.to_csv(f"{OUT}/rq4_importance.csv")
print("top features:\n", importance.head(10))

# logistic regression (coefficients show which features push toward which LLM)
scaler = StandardScaler().fit(Ftrain)
lr = LogisticRegression(max_iter=3000).fit(scaler.transform(Ftrain), train.llm_name)
pd.DataFrame(lr.coef_, index=lr.classes_, columns=all_cols).T.to_csv(f"{OUT}/rq4_lr_coefficients.csv")
results.append({"exp": "RQ4", "model": "logreg", "setting": "all_features",
                "acc": accuracy_score(test.llm_name, lr.predict(scaler.transform(Ftest)))})

# remove one group of features at a time, or use only that group
for group in FEATURE_GROUPS:
    keep = [c for c in all_cols if c not in FEATURE_GROUPS[group]]
    only = FEATURE_GROUPS[group]
    acc_drop = accuracy_score(test.llm_name, RandomForestClassifier(300, random_state=SEED, n_jobs=-1)
                              .fit(Ftrain[keep], train.llm_name).predict(Ftest[keep]))
    acc_only = accuracy_score(test.llm_name, RandomForestClassifier(300, random_state=SEED, n_jobs=-1)
                              .fit(Ftrain[only], train.llm_name).predict(Ftest[only]))
    results.append({"exp": "RQ4", "model": "random-forest", "setting": "drop_" + group, "acc": acc_drop})
    results.append({"exp": "RQ4", "model": "random-forest", "setting": "only_" + group, "acc": acc_only})
    print(f"{group}: without it {acc_drop:.3f} | only it {acc_only:.3f} | all {full_acc:.3f}")

# remove formatting from the text and retrain the CNN
for name, clean in [("no_formatting", remove_formatting), ("no_formatting_lowercase", remove_formatting_and_lowercase)]:
    acc, f1, y, preds = train_and_test("cnn", train, val, test, classes, "output", clean)
    results.append({"exp": "RQ4", "model": "cnn", "setting": name, "acc": acc, "f1": f1})

# ---------------- save + print results ----------------
res = pd.DataFrame(results)
res.to_csv(f"{OUT}/results.csv", index=False)
print("\n=== ALL RESULTS ===")
print(res.round(3).to_string(index=False))
