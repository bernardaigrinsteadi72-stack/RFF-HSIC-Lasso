# RFF-HSIC Lasso

Code and data for *Accelerated HSIC Lasso Based on Random Fourier Features*.

The main comparison uses eleven public datasets, six selectors, top-5/top-10 budgets, and RBF-SVM. Available configurations share 50 frozen folds from stratified 10-fold cross-validation repeated five times. The original RFF-only budget study also uses up to 20 features.

## Contents

- **benchmark_data.zip**: eleven input arrays, provenance, 550 ordinary splits, and 150 duplicate-grouped splits.
- **experiment_records.jsonl.zip** or **experiment_records.jsonl.gz**: archived selections, predictions, and observations. A ZIP containing the GZIP file is supported directly.
- **reported_results.json**: the fourteen manuscript tables, numerical figure data, dimension-rule realizations, nested-dimension results, and compact implementation-control records.
- **hsic.py**: random mapping, normalized factors, blocked construction, exact/RFF objectives, FISTA, and ordinary proximal gradient.
- **reproduce.py**: one selector and classifier fit on a frozen split.
- **experiments.py**: batch main/grouped comparisons, ablations, budgets, parameter sweeps, rule diagnostics, nested selection, solver comparisons, and isolated resource experiments.
- **controls.py**: matched Block NN-LARS/FISTA controls, label-kernel checks, storage-layout controls, and archived-record verification.
- **plot_results.py**: Figures 1-11 from archived numerical records.
- **summarize.py**: prediction checks, summaries, corrected paired intervals, Holm adjustment, Jaccard overlap, and CSV table export.
- **io_utils.py**: compressed-record loading and the dimension rule.
- **windows_resources.py**: Windows committed-memory limits and process memory measurements.
- **requirements.txt**: dependencies.

The archives can remain compressed. The package reuses the same input arrays and splits for all controls; full kernel matrices, coefficient vectors, logs, and manuscript files are not distributed.

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
python controls.py block-solvers --dry-run
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

## Matched implementation controls

~~~text
python controls.py verify
python controls.py block-solvers --variant dense --output runs/block_dense.jsonl
python controls.py block-solvers --variant matrix-free --output runs/block_matrix_free.jsonl
python controls.py label-kernels --output runs/label_kernels.jsonl
python controls.py storage-layout --output runs/storage_layout.jsonl
~~~

`verify` checks the existing records and predictions without fitting models. Other commands rerun the indicated controls in fresh serial processes; `--datasets` and `--splits` select smaller runs. All controls use the five frozen splits 0/10/20/30/40, two numerical-library threads, and RBF-SVM when classification is evaluated.

The Block solver control uses Digits, Libras Movement, and ORL at top-5/top-10. Both solvers receive the same upstream Block design and the penalty at the requested NN-LARS path endpoint. Dense FISTA uses the manuscript solver with twice the half-loss penalty. Its timer includes fresh redundancy-matrix construction; NN-LARS includes native neighbor calculations. Common Block construction, penalty discovery, validation, and classifier fitting are excluded. Each fold reports the median of three alternating-order runs, and Table 10 averages these five medians. Both FISTA controls allow 100000 updates; the matrix-free control additionally has a 120 s limit per fit. These conditional controls are separate from the full-pipeline timings in Tables 4-5.

Label-kernel checks use Breast Cancer, Digits, Libras Movement, and ORL at top-10, comparing the one-hot factor, dense Delta kernel, and class-weighted Delta kernel. Storage-layout controls use full Digits training folds, comparing the original explicit layout, Fortran-order explicit layout, streamed exact construction with feature blocks of 20, and RFF. Storage measurements require Windows and a 4096 MiB committed-memory cap; peak working set is a separate whole-process measurement.

The result file contains 60 Block solver cases, 20 label-kernel cases, and 20 storage-layout cases. Only the records needed for these reported comparisons are added; full coefficients and large intermediate arrays are omitted. Timing and memory values depend on the machine and numerical libraries.

## Parameters and dimension rule

~~~text
python experiments.py parameters --axis lambda --output runs/lambda.jsonl
python experiments.py parameters --axis D --output runs/dimension.jsonl
python experiments.py parameters --axis gamma --output runs/gamma.jsonl
python experiments.py rules --output runs/dimension_rule.jsonl
python experiments.py nested --output runs/nested_dimension.jsonl
~~~

All three parameter slices cover Breast Cancer, Digits, Libras Movement, and Semeion. Lambda uses 0.001/0.01/0.1 over 50 folds, with a fresh full fit for each value and rotated value order. D uses 5/10/the rule value, and gamma uses 0.125/0.5/2. All three parameter slices use the same 50 frozen outer folds. The Semeion rule dimension is 37. The records archive includes the complete D/gamma comparisons on all 50 folds, retaining the original five folds and adding the remaining 45 folds. The completion runs use the original Python 3.12.1 environment.

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
python summarize.py --export-controls control_tables --output control_summaries.json
python summarize.py --records runs/main.jsonl --output rerun_summary.json
python summarize.py --records runs/wine.jsonl --allow-partial
python plot_results.py --output figures --formats png,pdf
~~~

CSV export contains the fourteen tables in the 1116 manuscript. `--export-controls` exports four additional numerical tables: Block solver controls (including matrix-free FISTA), label-kernel balanced accuracies, paired accuracy tests, and main-solver convergence. New main/grouped runs can be summarized with `--records`. Plotting reads archived numerical data and fits no models. Figures 1-2 show paired differences, 3-4 feature frequencies, 5-6 Jaccard overlap, 7-8 selection times, and 9-11 lambda sensitivity. Output paths must be new; existing results and figures are not replaced.

| Manuscript tables | Contents | Reproduction entry |
|---|---|---|
| 1-3 | Dataset descriptions and top-10/top-5 accuracy | `experiments.py main` |
| 4-6 | Selection times and paired time-ratio intervals | Archived timing records; `summarize.py` table export |
| 7 | Digits subset runtime, memory, and intervals | `experiments.py resources` |
| 8 | Full objective versus relevance-only ranking | `experiments.py ablation` |
| 9 | FISTA versus ordinary proximal gradient | `experiments.py solver` |
| 10 | NN-LARS versus dense-system FISTA on Block designs | `controls.py block-solvers --variant dense` |
| 11 | Feature-budget comparison | `experiments.py budget` |
| 12 | Lambda, dimension, and kernel-width sensitivity | `experiments.py parameters` |
| 13 | Dimension rule versus training-only validation | `experiments.py nested` |
| 14 | Duplicate-grouped evaluation | `experiments.py grouped` |

## Protocol and sources

The split seed is 42. Outer mapping seeds are 42+split. Training-only scaling and selection precede RBF-SVM with C=1 and gamma=scale; its scaler is fitted on selected training columns.

Exact/RFF use gamma=0.5, lambda=0.01, zero initialization, KKT tolerance 1e-6, and 10000 main-study updates. The rule is D=min(80,max(10,floor(min(sqrt(n_train),15000/d)))). Feature blocks contain 20 original columns. Exact coefficient ties use increasing original index.

Block HSIC Lasso uses the original authors' pyHSICLasso 1.4.2 with sample-block size 20, three permutations, default kernels, n_jobs=1, and a separate NN-LARS fit for each budget.

- [UCI](https://archive.ics.uci.edu/)
- [scikit-learn datasets](https://scikit-learn.org/stable/datasets.html)
- [ASU/scikit-feature ORL](https://jundongl.github.io/scikit-feature/datasets.html)
- [Original pyHSICLasso](https://github.com/riken-aip/pyHSICLasso)

ORL contains 400 images, 1024 columns, and 40 classes. Input and split checksums are checked during loading. Original data and third-party software remain subject to their providers' terms.
