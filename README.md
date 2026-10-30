# Machine Learning Lab Experiments 6–15

Simple Python implementations for the ten ML experiments. The code prioritises easy-to-follow lab-exam structure.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Run one experiment from the project folder, for example:

```bash
python experiments/06_logistic_pima.py
```

Some scripts download datasets from public sources/OpenML, so an internet connection may be needed on the first run. Dataset downloads can take time, especially Fashion MNIST and MNIST. The scripts use small subsets for the SVM/KNN experiments to keep runtime manageable.

## Experiments

- `06_logistic_pima.py` — logistic regression with/without scaling
- `07_naive_bayes_newsgroups.py` — Multinomial vs Bernoulli Naïve Bayes
- `08_knn_fashion_mnist.py` — KNN with different K values
- `09_id3_online_retail.py` — entropy/information-gain decision tree for retail customer groups
- `10_income_models.py` — Logistic Regression vs Decision Tree on Adult Income
- `11_linear_svm_iris.py` — Linear SVM, decision boundary and margins (Setosa vs non-Setosa)
- `12_svm_kernels_fashion.py` — linear, polynomial and RBF SVM
- `13_mlp_wine_quality.py` — MLP with different hidden-layer architectures
- `14_activation_mnist.py` — MLP with Sigmoid, ReLU and Tanh
- `15_tune_fashion_mnist.py` — small hyperparameter experiment for MLP

## Dataset notes

- Pima: loaded from a public CSV URL.
- 20 Newsgroups: downloaded through scikit-learn.
- Fashion MNIST and MNIST: fetched through OpenML.
- Online Retail: provide `Online Retail.xlsx` in the project folder, or the script will explain where to get it.
- Adult Income: fetched through OpenML.
- Wine Quality: loaded from the UCI public CSV.

## Git: clone, commit and push

### Clone an existing repository

Use the actual repository URL (the profile URL by itself is not a clone URL):

```bash
git clone https://github.com/OWNER/REPOSITORY.git
cd REPOSITORY
```

### Push this folder to a new repository you own

Create an empty repository on GitHub first, then run these commands inside this folder:

```bash
git init
git add .
git commit -m "Add ML lab experiments 6-15"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

### Push to an existing repository you have write access to

Clone it first, copy the `experiments` folder, `README.md`, and `requirements.txt` into the cloned folder, then:

```bash
git add .
git commit -m "Add ML lab experiments 6-15"
git push
```

If the remote already has files, do not blindly replace them. Check `git status` and `git pull` first. You need write access to push to someone else's repository.
