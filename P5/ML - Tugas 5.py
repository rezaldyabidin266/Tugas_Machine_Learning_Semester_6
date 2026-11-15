# -*- coding: utf-8 -*-
# Versi Python dari notebook dengan nama yang sama.
from IPython.display import display


# %% [markdown]
# <a href="https://colab.research.google.com/github/pradityaputra888-lab/latihan-machine-learning/blob/main/pertemuan5/latihan5MachineLearning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% (sel 2)
import pandas as pd, numpy as np, matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# %% (sel 3)
#import dataset
from pathlib import Path

# Mendukung script Python dan notebook yang dijalankan dari folder proyek/pertemuan.
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_DIR = next(
    (p for p in (BASE_DIR, BASE_DIR.parent) if (p / "datasets").is_dir()),
    BASE_DIR.parent,
)

dataset_path = PROJECT_DIR / "datasets" / "breast_cancer.csv"
if dataset_path.is_file():
    df = pd.read_csv(dataset_path)
else:
    # Wisconsin Diagnostic Breast Cancer tersedia langsung di scikit-learn.
    from sklearn.datasets import load_breast_cancer
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.data.copy()
    df["diagnosis"] = dataset.target.map({0: "M", 1: "B"})
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
X = df.drop(columns="diagnosis"); y = (df["diagnosis"] == "M").astype(int)  # M=1

# %% (sel 4)
df.info()

# %% (sel 5)
#split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# %% (sel 6)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []
for d in [2, 4, 6, None]:
    for m in [1, 5, 10]:
        mk = lambda: DecisionTreeClassifier(max_depth=d, min_samples_leaf=m, random_state=42)
        r = cross_validate(mk(), Xtr, ytr, cv=cv, scoring="accuracy")
        t = mk().fit(Xtr, ytr)
        rows.append(dict(max_depth=str(d), min_samples_leaf=m,
            train_acc=accuracy_score(ytr, t.predict(Xtr)),
            cv_mean=r["test_score"].mean(), cv_std=r["test_score"].std(),
            leaf=t.get_n_leaves(), depth_real=t.get_depth(),
            test_acc=accuracy_score(yte, t.predict(Xte)), _d=d))
res = pd.DataFrame(rows)

# %% (sel 7)
best = res.loc[res.cv_mean.idxmax()]
se = best.cv_std / np.sqrt(5)
cand = res[res.cv_mean >= best.cv_mean - se]
pick = cand.sort_values(["leaf", "cv_mean"], ascending=[True, False]).iloc[0]
res["gap_train_cv"] = res.train_acc - res.cv_mean
res["gap_train_test"] = res.train_acc - res.test_acc

out = res.drop(columns="_d").round(4)
out.to_csv("tabel_hasil.csv", index=False)
print(out.to_string())
print("\nCV terbaik :", best.max_depth, best.min_samples_leaf, round(best.cv_mean,4), "SE", round(se,4))
print("DIPILIH    :", pick.max_depth, pick.min_samples_leaf)

final = DecisionTreeClassifier(max_depth=(None if pick.max_depth=="None" else int(pick.max_depth)), min_samples_leaf=int(pick.min_samples_leaf), random_state=42).fit(Xtr, ytr)
tr, te = accuracy_score(ytr, final.predict(Xtr)), accuracy_score(yte, final.predict(Xte))
print(f"\nFINAL train={tr:.4f} CV={pick.cv_mean:.4f} test={te:.4f} gap={tr-te:.4f} leaf={final.get_n_leaves()} depth={final.get_depth()}")

# %% (sel 8)
#Visualisasi tabel CV & pohon

fig, ax = plt.subplots(figsize=(7,4.5))
for m in [1,5,10]:
    s = res[res.min_samples_leaf==m]; ax.plot(s.max_depth, s.train_acc,"o--",label=f"train msl={m}")
    ax.plot(s.max_depth, s.cv_mean,"s-",label=f"CV msl={m}")
ax.set_xlabel("max_depth"); ax.set_ylabel("Akurasi"); ax.legend(fontsize=7,ncol=2); ax.grid(alpha=.3)
ax.set_title("Train vs CV per kombinasi hyperparameter"); plt.tight_layout(); plt.savefig("grafik_train_vs_cv.png", dpi=150); plt.show()

fig = plt.figure(figsize=(16,8))
plot_tree(final, feature_names=list(X.columns), class_names=["Jinak (B)","Ganas (M)"], filled=True, rounded=True, fontsize=9)
plt.title(f"Pohon terpilih: max_depth={pick.max_depth}, min_samples_leaf={int(pick.min_samples_leaf)}")
plt.tight_layout(); plt.savefig("pohon_terpilih.png", dpi=150); plt.show()
