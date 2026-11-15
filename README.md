# RFF-HSIC Lasso

Code and data for *Accelerated HSIC Lasso Based on Random Fourier Features*.

The main comparison uses eleven public datasets, six selectors, top-5/top-10 budgets, and RBF-SVM. Available configurations share 50 frozen folds from stratified 10-fold cross-validation repeated five times. The original RFF-only budget study also uses up to 20 features.

## Contents

- **benchmark_data.zip**: eleven input arrays, provenance, 550 ordinary splits, and 150 duplicate-grouped splits.
- **experiment_records.jsonl.zip** or **experiment_records.jsonl.gz**: archived selections, predictions, and observations. A ZIP containing the GZIP file is supported directly.
- **reported_results.json**: the eleven manuscript tables, numerical figure data, dimension-rule realizations, and nested-dimension results.
- **hsic.py**: random mapping, normalized factors, blocked construction, exact/RFF objectives, FISTA, and ordinary proximal gradient.
- **reproduce.py**: one selector and classifier fit on a frozen split.
- **experiments.py**: batch main/grouped comparisons, ablations, budgets, parameter sweeps, rule diagnostics, nested selection, solver comparisons, and isolated resource experiments.
- **plot_results.py**: Figures 1-11 from archived numerical records.
- **summarize.py**: prediction checks, summaries, corrected paired intervals, Holm adjustment, Jaccard overlap, and CSV table export.
- **io_utils.py**: compressed-record loading and the dimension rule.
- **windows_resources.py**: Windows committed-memory limits and process memory measurements.
- **requirements.txt**: dependencies.

Upload the files in this directory to the repository root. The archives can remain compressed; extracted duplicate folders are unnecessary.

## Installation and one frozen case

Use Python 3.12 and the dependencies in requirements.txt.

~~~text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python reproduce.py --dataset Wine --split 0 --method RFF-HSIC --budget 10 --check-reference
.venv\Scripts\python reproduce.py --dataset Wine --split 0 --method Block-HSIC --budget 5 --check-reference
.venv\Scripts\python reproduce.py --dataset Ionosphere --protocol grouped --split 0 --method RFF-HSIC --budget 10 --check-reference
~~~

On macOS/Linux use .venv/bin/python. Commands below use python from the activated environment. The split index is 0-49. Reference checking compares selected columns, predictions, and accuracy with the archived main/grouped configuration. The relevance-only ablation is named RFF-HSIC-screen.

## Plan without fitting

~~~text
python experiments.py main --dry-run
python experiments.py parameters --dry-run
python experiments.py nested --dry-run
python plot_results.py --dry-run
~~~

These commands list the work without fitting models or creating result files.

## Main and grouped comparisons

~~~text
python experiments.py main --datasets Wine --splits 0 --output runs/wine.jsonl
python experiments.py main --output runs/main.jsonl
python experiments.py grouped --output runs/grouped.jsonl
~~~

The full commands use all 50 splits. Jobs run in fresh, serial processes. Exact/RFF and conventional selectors share a fitted ranking across budgets; Block HSIC Lasso fits its requested path separately. Explicit-design allocation failures are recorded without accuracy values.

## Ablation and feature budgets

~~~text
python experiments.py ablation --output runs/ablation.jsonl
python experiments.py budget --output runs/budgets.jsonl
python experiments.py solver --output runs/solvers.jsonl
~~~

The relevance-only study uses the original ten datasets and top-10. Budgets are 5/10/min(20,d). The solver comparison uses Breast Cancer, Digits, Libras Movement, and ORL on splits 0/10/20/30/40: identical Q and h, zero initialization, identical step size and KKT tolerance, and 100000 allowed updates for both FISTA and PG. Construction and classifier fitting are excluded from solve times.

## Parameters and dimension rule

~~~text
python experiments.py parameters --axis lambda --output runs/lambda.jsonl
python experiments.py parameters --axis D --output runs/dimension.jsonl
python experiments.py parameters --axis gamma --output runs/gamma.jsonl
python experiments.py rules --output runs/dimension_rule.jsonl
python experiments.py nested --output runs/nested_dimension.jsonl
~~~

Lambda uses 0.001/0.01/0.1 on Breast Cancer, Libras Movement, and Semeion over 50 folds. D uses 5/10/the rule value, and gamma uses 0.125/0.5/2 on Breast Cancer, Digits, and Libras Movement over splits 0/10/20/30/40.

The rule diagnostic varies the lower bound 5/10/20, upper bound 40/80/160, and product constant 7500/15000/30000 one at a time. It records realized dimensions without fitting models.

Nested selection uses Digits and Multiple Features, ten frozen outer folds, three inner folds, and candidate dimensions 10/20/40/80. Inner split seeds are 1042+outer_split; mapping seeds are 100000+100*outer_split+inner_fold. Candidates must converge in every inner fold; score ties choose smaller D. The chosen D and original rule are evaluated on the same outer test fold.

The --datasets, --splits, and --methods options select smaller runs. For example, --datasets "Digits,Multiple Features" --splits 0,10 selects two datasets and two splits. reproduce.py also accepts --D, --gamma, and --lam for individual fits.

## Runtime and memory

These commands implement the published Windows process-memory protocol.

~~~text
python experiments.py resources --output runs/resources.jsonl
python experiments.py construction --output runs/construction.jsonl
~~~

Resources use identical stratified Digits subsets of 100/200/400/600 samples, seeds 42/43/44, all 64 columns, three selectors, two numerical-library threads, and a 4096 MiB process committed-memory limit. Each case uses a fresh process. Peak working set is measured after selection, before classifier fitting.

Construction compares blocked and unblocked RFF construction on Multiple Features split 0 under 3072 MiB. It measures construction only. Memory failures are retained. Historical resource observations remain in the archive; a new machine produces new measurements.

## Statistics, tables, and figures

~~~text
python summarize.py --output summaries.json --export-tables tables
python summarize.py --records runs/main.jsonl --output rerun_summary.json
python summarize.py --records runs/wine.jsonl --allow-partial
python plot_results.py --output figures --formats png,pdf
~~~

CSV export contains the eleven archived manuscript tables. New main/grouped runs can be summarized with --records. Plotting reads archived numerical data and fits no models. Figures 1-2 show paired differences, 3-4 feature frequencies, 5-6 Jaccard overlap, 7-8 selection times, and 9-11 lambda sensitivity. Output paths must be new; existing results and figures are not replaced.

## Protocol and sources

The split seed is 42. Outer mapping seeds are 42+split. Training-only scaling and selection precede RBF-SVM with C=1 and gamma=scale; its scaler is fitted on selected training columns.

Exact/RFF use gamma=0.5, lambda=0.01, zero initialization, KKT tolerance 1e-6, and 10000 main-study updates. The rule is D=min(80,max(10,floor(min(sqrt(n_train),15000/d)))). Feature blocks contain 20 original columns. Exact coefficient ties use increasing original index.

Block HSIC Lasso uses the original authors' pyHSICLasso 1.4.2 with sample-block size 20, three permutations, default kernels, n_jobs=1, and a separate NN-LARS fit for each budget.

- [UCI](https://archive.ics.uci.edu/)
- [scikit-learn datasets](https://scikit-learn.org/stable/datasets.html)
- [ASU/scikit-feature ORL](https://jundongl.github.io/scikit-feature/datasets.html)
- [Original pyHSICLasso](https://github.com/riken-aip/pyHSICLasso)

ORL contains 400 images, 1024 columns, and 40 classes. Input and split checksums are checked during loading. Original data and third-party software remain subject to their providers' terms.
