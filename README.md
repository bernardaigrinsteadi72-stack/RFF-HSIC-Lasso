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

The relevance-only study uses all eleven datasets at top-10 on the same 50 frozen splits. The feature-budget study also covers all eleven datasets, with budgets 5/10/min(20,d) using the same training-fold ranking across budgets. The solver comparison uses Breast Cancer, Digits, Libras Movement, and ORL on all 50 frozen splits: identical Q and h, zero initialization, identical step size and KKT tolerance, and 100000 allowed updates for both FISTA and PG. Construction and classifier fitting are excluded from solve times. The original five folds are retained; the remaining folds counterbalance solver order so each method runs first on 25 folds per dataset.

## Matched implementation controls

~~~text
python controls.py verify
python controls.py block-solvers --variant dense --output runs/block_dense.jsonl
python controls.py block-solvers --variant matrix-free --output runs/block_matrix_free.jsonl
python controls.py label-kernels --output runs/label_kernels.jsonl
python controls.py storage-layout --output runs/storage_layout.jsonl
~~~

`verify` checks the existing records and predictions without fitting models. Other commands rerun the indicated controls in fresh serial processes; `--datasets` and `--splits` select smaller runs. The dense Block solver control uses all 50 frozen splits. The matrix-free, label-kernel, and storage-layout controls retain splits 0/10/20/30/40. All use two numerical-library threads and RBF-SVM when classification is evaluated.

The Block solver control uses Digits, Libras Movement, and ORL at top-5/top-10. Both solvers receive the same upstream Block design and the penalty at the requested NN-LARS path endpoint. Dense FISTA uses the manuscript solver with twice the half-loss penalty. Its timer includes fresh redundancy-matrix construction; NN-LARS includes native neighbor calculations. Common Block construction, penalty discovery, validation, and classifier fitting are excluded. Each fold reports the median of three alternating-order runs, and Table 10 averages these 50 medians. Both FISTA controls allow 100000 updates; the matrix-free control additionally has a 120 s limit per fit. These conditional controls are separate from the full-pipeline timings in Tables 4-5.

Label-kernel checks use Breast Cancer, Digits, Libras Movement, and ORL at top-10, comparing the one-hot factor, dense Delta kernel, and class-weighted Delta kernel. Storage-layout controls use full Digits training folds, comparing the original explicit layout, Fortran-order explicit layout, streamed exact construction with feature blocks of 20, and RFF. Storage measurements require Windows and a 4096 MiB committed-memory cap; peak working set is a separate whole-process measurement.

The public records contain 300 dense Block solver cases, 30 matrix-free Block solver cases, 20 label-kernel cases, and 20 storage-layout cases. The original control cases remain in reported_results.json; the additional dense Block cases and all 400 paired FISTA/PG fits are stored in the compressed records archive. Only the records needed for these reported comparisons are added; full coefficients and large intermediate arrays are omitted. Timing and memory values depend on the machine and numerical libraries.

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

CSV export contains the fourteen tables in the manuscript. `--export-controls` exports four additional numerical tables: Block solver controls (including matrix-free FISTA), label-kernel balanced accuracies, paired accuracy tests, and main-solver convergence. New main/grouped runs can be summarized with `--records`. Plotting reads archived numerical data and fits no models. Figures 1-2 show paired differences, 3-4 feature frequencies, 5-6 Jaccard overlap, 7-8 selection times, and 9-11 lambda sensitivity. Output paths must be new; existing results and figures are not replaced.

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

## Readable result tables

These tables display the existing archived results without running code. The numerical entries are copied from the reported results; no additional fits or raw data are included.

- [Paired accuracy differences and Holm-adjusted tests](#paired-accuracy-tests)
- [Label-kernel checks](#label-kernel-checks)
- [Main-solver convergence](#main-solver-convergence)
- [Block solver controls](#block-solver-controls)
- [Storage-layout controls](#storage-layout-controls)

### Paired accuracy tests

Differences are RFF-HSIC Lasso minus the named comparator, in percentage points (pp). HSIC denotes HSIC Lasso and Block denotes Block HSIC Lasso. Each comparison uses the same 50 outer folds. Intervals are pointwise 95% intervals with repeated-cross-validation variance correction and 49 degrees of freedom. Holm adjustment applies to the full family of 40 hypothesis tests, not to interval endpoints.

| Dataset | k | Method | Difference / pp | 95% CI / pp | Raw P | Holm P |
| --- | --- | --- | --- | --- | --- | --- |
| Breast Cancer | 10 | HSIC | 0.071 | [-0.895, 1.036] | 0.8834 | 1.0000 |
| Breast Cancer | 10 | Block | 0.175 | [-0.691, 1.042] | 0.6859 | 1.0000 |
| Wine | 10 | HSIC | 0.000 | [0.000, 0.000] | 1.0000 | 1.0000 |
| Wine | 10 | Block | 0.000 | [0.000, 0.000] | 1.0000 | 1.0000 |
| Digits | 10 | HSIC | -0.145 | [-0.885, 0.595] | 0.6963 | 1.0000 |
| Digits | 10 | Block | 2.226 | [0.538, 3.914] | 0.0108 | 0.3881 |
| Ionosphere | 10 | HSIC | 0.459 | [-1.290, 2.207] | 0.6004 | 1.0000 |
| Ionosphere | 10 | Block | 0.746 | [-1.115, 2.607] | 0.4244 | 1.0000 |
| Sonar | 10 | HSIC | 0.171 | [-3.505, 3.848] | 0.9257 | 1.0000 |
| Sonar | 10 | Block | 1.710 | [-3.513, 6.932] | 0.5137 | 1.0000 |
| Libras Movement | 10 | HSIC | -0.278 | [-3.725, 3.169] | 0.8720 | 1.0000 |
| Libras Movement | 10 | Block | 9.056 | [3.057, 15.054] | 0.0039 | 0.1467 |
| Semeion | 10 | Block | -1.093 | [-4.013, 1.826] | 0.4553 | 1.0000 |
| Vehicle | 10 | HSIC | -0.047 | [-1.253, 1.159] | 0.9382 | 1.0000 |
| Vehicle | 10 | Block | -0.025 | [-2.125, 2.076] | 0.9813 | 1.0000 |
| Image Segmentation | 10 | HSIC | -0.346 | [-1.091, 0.399] | 0.3547 | 1.0000 |
| Image Segmentation | 10 | Block | -0.554 | [-1.141, 0.033] | 0.0637 | 1.0000 |
| Multiple Features | 10 | Block | 0.660 | [-0.208, 1.528] | 0.1329 | 1.0000 |
| ORL | 10 | HSIC | 0.250 | [-5.576, 6.076] | 0.9316 | 1.0000 |
| ORL | 10 | Block | 1.600 | [-4.868, 8.068] | 0.6214 | 1.0000 |
| Breast Cancer | 5 | HSIC | -0.074 | [-2.620, 2.472] | 0.9537 | 1.0000 |
| Breast Cancer | 5 | Block | 0.769 | [-1.544, 3.082] | 0.5073 | 1.0000 |
| Wine | 5 | HSIC | -0.333 | [-1.303, 0.636] | 0.4930 | 1.0000 |
| Wine | 5 | Block | -0.333 | [-1.303, 0.636] | 0.4930 | 1.0000 |
| Digits | 5 | HSIC | 0.244 | [-2.223, 2.710] | 0.8435 | 1.0000 |
| Digits | 5 | Block | 3.851 | [0.932, 6.770] | 0.0108 | 0.3881 |
| Ionosphere | 5 | HSIC | -0.057 | [-0.720, 0.606] | 0.8632 | 1.0000 |
| Ionosphere | 5 | Block | 0.057 | [-0.728, 0.842] | 0.8842 | 1.0000 |
| Sonar | 5 | HSIC | -0.767 | [-4.431, 2.898] | 0.6760 | 1.0000 |
| Sonar | 5 | Block | 1.176 | [-3.591, 5.943] | 0.6222 | 1.0000 |
| Libras Movement | 5 | HSIC | 1.167 | [-5.317, 7.650] | 0.7192 | 1.0000 |
| Libras Movement | 5 | Block | 13.889 | [4.251, 23.526] | 0.0056 | 0.2084 |
| Semeion | 5 | Block | -5.586 | [-8.623, -2.550] | 0.0006 | 0.0215 |
| Vehicle | 5 | HSIC | 0.286 | [-1.986, 2.558] | 0.8013 | 1.0000 |
| Vehicle | 5 | Block | -0.309 | [-2.486, 1.869] | 0.7767 | 1.0000 |
| Image Segmentation | 5 | HSIC | -0.052 | [-0.757, 0.653] | 0.8829 | 1.0000 |
| Image Segmentation | 5 | Block | -1.299 | [-1.899, -0.698] | 0.0001 | 0.0028 |
| Multiple Features | 5 | Block | 1.390 | [-1.023, 3.803] | 0.2525 | 1.0000 |
| ORL | 5 | HSIC | -1.150 | [-7.696, 5.396] | 0.7256 | 1.0000 |
| ORL | 5 | Block | -0.800 | [-9.013, 7.413] | 0.8456 | 1.0000 |

### Label-kernel checks

Mean balanced accuracies on five matched folds at top-10. The one-hot factor and dense Delta kernel represent the same label kernel. The class-weighted Delta kernel changes the objective and is evaluated separately.

| Dataset | One-hot factor | Dense Delta kernel | Class-weighted Delta kernel |
| --- | --- | --- | --- |
| Breast Cancer | 0.9716 | 0.9716 | 0.9716 |
| Digits | 0.9385 | 0.9385 | 0.9385 |
| Libras Movement | 0.7756 | 0.7756 | 0.7689 |
| ORL | 0.6900 | 0.6900 | 0.6900 |

### Main-solver convergence

Each successful full-ranking fit is counted once when it serves both feature budgets. Iteration counts and residuals cover the 1000 distinct successful main timing fits. Allocation failures occur before solving and are not included. At limit / fits records how many fits reached the iteration limit.

| Dataset | Method | Iterations min–max | Median iterations | Maximum KKT residual | At limit / fits |
| --- | --- | --- | --- | --- | --- |
| Breast Cancer | HSIC | 530–1440 | 940 | 9.979e-07 | 0/50 |
| Breast Cancer | RFF-HSIC | 330–1330 | 805 | 9.993e-07 | 0/50 |
| Wine | HSIC | 90–160 | 130 | 9.824e-07 | 0/50 |
| Wine | RFF-HSIC | 70–170 | 110 | 9.957e-07 | 0/50 |
| Digits | HSIC | 150–220 | 150 | 9.911e-07 | 0/50 |
| Digits | RFF-HSIC | 100–260 | 170 | 9.998e-07 | 0/50 |
| Ionosphere | HSIC | 140–200 | 180 | 9.716e-07 | 0/50 |
| Ionosphere | RFF-HSIC | 130–180 | 160 | 9.720e-07 | 0/50 |
| Sonar | HSIC | 140–240 | 190 | 9.915e-07 | 0/50 |
| Sonar | RFF-HSIC | 80–250 | 175 | 9.879e-07 | 0/50 |
| Libras Movement | HSIC | 960–2450 | 1730 | 9.946e-07 | 0/50 |
| Libras Movement | RFF-HSIC | 850–2230 | 1490 | 9.966e-07 | 0/50 |
| Semeion | RFF-HSIC | 230–270 | 250 | 9.959e-07 | 0/50 |
| Vehicle | HSIC | 160–450 | 305 | 9.877e-07 | 0/50 |
| Vehicle | RFF-HSIC | 150–420 | 240 | 9.974e-07 | 0/50 |
| Image Segmentation | HSIC | 280–380 | 330 | 9.928e-07 | 0/50 |
| Image Segmentation | RFF-HSIC | 190–940 | 390 | 9.961e-07 | 0/50 |
| Multiple Features | RFF-HSIC | 1460–2800 | 1910 | 9.924e-07 | 0/50 |
| ORL | HSIC | 2660–3900 | 3245 | 9.993e-07 | 0/50 |
| ORL | RFF-HSIC | 1900–4090 | 3040 | 9.970e-07 | 0/50 |

### Block solver controls

NN-LARS and FISTA use identical Block designs and penalties matched at the NN-LARS training-path endpoint. Dense-system rows use 50 frozen folds, matching manuscript Table 10; matrix-free rows use five matched folds. Times are conditional solver costs, not complete selection-pipeline times. Dense FISTA includes fresh redundancy-matrix construction and NN-LARS includes native neighbor calculations. Common Block construction, penalty discovery, validation, and classifier fitting are excluded. Identical sets / folds counts agreement of the selected feature sets.

| Dataset | Budget k | FISTA implementation | NN-LARS / s | FISTA / s | Identical sets / folds |
| --- | --- | --- | --- | --- | --- |
| Digits | 5 | Dense system | 0.0418 | 0.0114 | 50 / 50 |
| Digits | 5 | Matrix-free | 0.0365 | 0.9537 | 5 / 5 |
| Digits | 10 | Dense system | 0.0963 | 0.0109 | 50 / 50 |
| Digits | 10 | Matrix-free | 0.0844 | 1.0474 | 5 / 5 |
| Libras Movement | 5 | Dense system | 0.0104 | 0.0084 | 50 / 50 |
| Libras Movement | 5 | Matrix-free | 0.0077 | 0.3941 | 5 / 5 |
| Libras Movement | 10 | Dense system | 0.0259 | 0.0108 | 48 / 50 |
| Libras Movement | 10 | Matrix-free | 0.0203 | 0.6434 | 5 / 5 |
| ORL | 5 | Dense system | 0.1221 | 0.4672 | 50 / 50 |
| ORL | 5 | Matrix-free | 0.1136 | 22.9855 | 5 / 5 |
| ORL | 10 | Dense system | 0.2288 | 0.5068 | 50 / 50 |
| ORL | 10 | Matrix-free | 0.2317 | 36.3082 | 5 / 5 |

### Storage-layout controls

Whole-process peak working set on five matched full Digits training folds, under a 4096 MiB committed-memory cap. Exact-HSIC is the original explicit layout; Exact-A-F uses Fortran-order explicit construction; Exact-streamed regenerates exact kernels in feature blocks of 20; RFF-HSIC is the factor-space method. Peak working set and committed-memory limit are separate measurements.

| Construction | Folds | Mean peak working set / MiB |
| --- | --- | --- |
| Exact-HSIC | 5 | 1534.0906 |
| Exact-A-F | 5 | 1513.9273 |
| Exact-streamed | 5 | 1035.5664 |
| RFF-HSIC | 5 | 241.2266 |
