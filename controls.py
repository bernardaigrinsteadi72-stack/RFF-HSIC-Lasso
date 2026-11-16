"""Reproduce the implementation controls and verify their compact archived records."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'

ROOT = Path(__file__).resolve().parent
FOLDS = [0, 10, 20, 30, 40]
MAX_ITER, TOL, MAX_SECONDS = 100000, 1e-6, 120


def matrix_free_fista(design, h, lam):
    """Original half-loss control, including its initialization and casts."""
    import numpy as np
    began = time.perf_counter()
    lip = max(float(np.sum(np.square(design, dtype=np.float64), dtype=np.float64)), 1e-12)
    step, penalty = np.float32(1/lip), np.float32(lam)
    x = np.zeros(design.shape[1], dtype=np.float32)
    z, momentum, residual = x.copy(), 1.0, float('inf')
    timed_out = False
    for iteration in range(1, MAX_ITER+1):
        gradient = design.T@(design@z)-h+penalty
        new = np.maximum(z-step*gradient, np.float32(0))
        nxt = (1+(1+4*momentum*momentum)**0.5)/2
        z = new+np.float32((momentum-1)/nxt)*(new-x)
        x, momentum = new, nxt
        if iteration % 10 == 0:
            gradient = design.T@(design@x)-h+penalty
            residual = float(np.max(np.abs(x-np.maximum(x-gradient, np.float32(0)))))
            if residual <= TOL:
                break
            if time.perf_counter()-began > MAX_SECONDS:
                timed_out = True
                break
    return x, dict(solve_seconds=time.perf_counter()-began, iterations=iteration, kkt_residual=residual,
                   converged=residual <= TOL, timed_out=timed_out)


def projected_kkt(design, h, alpha, lam):
    import numpy as np
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    g = design.T@(design@a)-np.asarray(h, dtype=np.float64).reshape(-1)+lam
    return float(np.max(np.abs(a-np.maximum(a-g, 0))))


def evaluation(alpha, X, y, tr, te, k):
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    from reproduce import predict_selected
    idx = np.argsort(-alpha, kind='stable')[:min(k, X.shape[1])]
    result = predict_selected(X[tr], X[te], y[tr], y[te], idx)
    result['balanced_accuracy'] = float(balanced_accuracy_score(y[te], result['predictions']))
    return dict(selected_features=idx.tolist(), **result)


def block_solvers(job, X, y, tr, te):
    import numpy as np
    import pyHSICLasso.api as api
    from pyHSICLasso import HSICLasso
    from sklearn.preprocessing import StandardScaler
    from hsic import solve
    seed, k, variant = 42+job['split'], job['k'], job['variant']
    original = api.nlars
    captured = {}

    def observe(design, relevance, budget, neighbors):
        captured.update(design=design, h=relevance,
                        common_build_seconds=time.perf_counter()-began)
        return original(design, relevance, budget, neighbors)

    warm = np.ones((64, 64)); warm@warm
    np.random.seed(seed)
    api.nlars = observe
    began = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model = HSICLasso()
            model.input(StandardScaler().fit_transform(X[tr]), y[tr])
            model.classification(k, B=20, M=3, n_jobs=1, discrete_x=False)
    finally:
        api.nlars = original
    design, h = captured['design'], captured['h'].reshape(-1)
    penalty = float(model.lam[0, model.path.shape[1]-1])
    reference = model.beta.reshape(-1)
    timings = {'NN-LARS': [], 'FISTA': []}
    coefficients, diagnostics, gram_times, solve_times = {}, [], [], []
    for rep in range(3):
        order = ['NN-LARS', 'FISTA'] if (rep+job['split']+k) % 2 else ['FISTA', 'NN-LARS']
        for method in order:
            np.random.seed(seed)
            start = time.perf_counter()
            if method == 'NN-LARS':
                answer = original(design, captured['h'], k, model.max_neighbors)
                elapsed = time.perf_counter()-start
                alpha = answer[1].reshape(-1).copy()
                assert np.array_equal(alpha, reference)
            elif variant == 'dense':
                Q = design.T@design
                built = time.perf_counter()
                alpha, diag = solve(Q, h, lam=2*penalty, max_iter=MAX_ITER, tol=TOL)
                stopped = time.perf_counter()
                elapsed = stopped-start
                gram_times.append(built-start); solve_times.append(stopped-built)
                diagnostics.append(diag)
                del Q
            else:
                alpha, diag = matrix_free_fista(design, h, penalty)
                elapsed = diag['solve_seconds']
                diagnostics.append(diag)
            timings[method].append(elapsed)
            coefficients[method] = alpha
    values = []
    for method in ['NN-LARS', 'FISTA']:
        alpha = coefficients[method]
        a = np.asarray(alpha, dtype=np.float64)
        prediction = design@a
        values.append(dict(solver=method,
            timings_seconds=timings[method], median_solver_seconds=float(np.median(timings[method])),
            kkt_residual=projected_kkt(design, h, alpha, penalty),
            objective_without_label_constant=float(.5*(prediction@prediction)-h@a+penalty*a.sum()),
            **evaluation(alpha, X, y, tr, te, k)))
    a, b = [next(v for v in values if v['solver'] == name) for name in ['NN-LARS', 'FISTA']]
    return dict(matched_lambda=penalty, design_shape=list(design.shape),
        design_dtype=str(design.dtype), common_build_seconds=captured['common_build_seconds'],
        evaluations=values, fista_diagnostics=diagnostics,
        fista_gram_seconds=gram_times, fista_solve_seconds=solve_times,
        same_feature_set=set(a['selected_features']) == set(b['selected_features']),
        same_predictions=a['predictions'] == b['predictions'])


def label_kernels(job, X, y, tr, te):
    import numpy as np
    from hsic import factors, labels, q_from_factors, solve_system
    from io_utils import dimension_rule
    D, seed = dimension_rule(len(tr), X.shape[1]), 42+job['split']
    Z, h, Y, _ = factors(X[tr], y[tr], D, .5, seed)
    Q = q_from_factors(Z, D)
    dense = (y[tr, None] == y[None, tr]).astype(float)
    dense = dense-dense.mean(0)[None, :]-dense.mean(1)[:, None]+dense.mean()
    dense /= max(np.linalg.norm(dense), 1e-12)
    hd = np.asarray([np.sum((z@z.T)*dense) for z in Z])
    balanced = labels(y[tr], 'balanced')
    hb = np.asarray([np.sum((z.T@balanced)**2) for z in Z])
    values, coefficients = [], []
    for mode, relevance in [('onehot_factor', h), ('dense_delta', hd), ('class_balanced_delta', hb)]:
        alpha, diag = solve_system(Q, relevance)
        coefficients.append(alpha)
        values.append(dict(label_mode=mode, diagnostics=diag, **evaluation(alpha, X, y, tr, te, 10)))
    classes, indices = np.unique(y[tr], return_inverse=True)
    encoding = []
    for encoding_seed in [11, 22, 33, 44, 55]:
        mapping = np.random.RandomState(encoding_seed).permutation(len(classes))
        YY = labels(mapping[indices])
        hp = np.asarray([np.sum((z.T@YY)**2) for z in Z])
        encoding.append(dict(seed=encoding_seed, h_max_abs_error=float(np.max(np.abs(hp-h)))))
    return dict(D=D, evaluations=values,
        factor_vs_dense_label_kernel_max_abs_error=float(np.max(np.abs(Y@Y.T-dense))),
        factor_vs_dense_h_max_abs_error=float(np.max(np.abs(h-hd))),
        factor_vs_dense_coefficient_max_abs_error=float(np.max(np.abs(coefficients[0]-coefficients[1]))),
        encoding_permutations=encoding)


def exact_system(X, y, block=None):
    """Fortran-order explicit design or exact feature-block regeneration."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from hsic import labels
    X = StandardScaler().fit_transform(X)
    n, d = X.shape
    Y = labels(y); target = (Y@Y.T).reshape(-1)

    def matrix(start, end):
        A = np.empty((n*n, end-start), order='F')
        for offset, j in enumerate(range(start, end)):
            v = X[:, j]
            K = np.exp(-.5*(v[:, None]-v[None, :])**2)
            K = K-K.mean(0)[None, :]-K.mean(1)[:, None]+K.mean()
            A[:, offset] = (K/max(np.linalg.norm(K), 1e-12)).reshape(-1)
        return A

    if block is None:
        A = matrix(0, d)
        return A.T@A, A.T@target
    Q, h = np.empty((d, d)), np.empty(d)
    for start in range(0, d, block):
        end = min(start+block, d); left = matrix(start, end)
        h[start:end] = left.T@target
        for right_start in range(start, d, block):
            right_end = min(right_start+block, d)
            right = left if right_start == start else matrix(right_start, right_end)
            values = left.T@right
            Q[start:end, right_start:right_end] = values
            if right_start != start:
                Q[right_start:right_end, start:end] = values.T
                del right
        del left
    return Q, h


def storage_layout(job, X, y, tr, te):
    import numpy as np
    from hsic import JournalHSIC, solve_system
    from io_utils import dimension_rule
    from windows_resources import memory_info
    if os.name != 'nt':
        raise RuntimeError('The published peak-working-set control requires Windows.')
    n, d = len(tr), X.shape[1]
    D, method = dimension_rule(n, d), job['method']
    warm = np.ones((96, 96)); warm@warm
    before = memory_info(); began = time.perf_counter()
    if method in ['Exact-HSIC', 'RFF-HSIC']:
        obj = JournalHSIC(exact=method == 'Exact-HSIC', D=D, seed=42+job['split']).fit(X[tr], y[tr])
        alpha, diag = obj.alpha_, obj.diagnostics_
    else:
        Q, h = exact_system(X[tr], y[tr], block=20 if method == 'Exact-streamed' else None)
        alpha, diag = solve_system(Q, h)
    elapsed = time.perf_counter()-began
    after = memory_info()
    return dict(D=D, n=n, d=d, selected_features=np.argsort(-alpha, kind='stable')[:10].tolist(),
                selector_seconds=elapsed, diagnostics=diag,
                memory_before_selection=before, memory_after_selection=after)


def planned_jobs(args):
    defaults = {'block-solvers': ['Digits', 'Libras Movement', 'ORL'],
                'label-kernels': ['Breast Cancer', 'Digits', 'Libras Movement', 'ORL'],
                'storage-layout': ['Digits']}
    datasets = defaults[args.study] if args.datasets is None else args.datasets.split(',')
    folds = (list(range(50)) if args.study == 'block-solvers' and args.variant in ['dense', 'both']
             else FOLDS) if args.splits is None else [int(s) for s in args.splits.split(',')]
    if not folds or len(folds) != len(set(folds)) or any(s < 0 or s >= 50 for s in folds):
        raise ValueError('Choose distinct split indices from 0 to 49.')
    jobs = []
    for dataset in datasets:
        if dataset not in defaults[args.study]:
            raise ValueError('Dataset is outside the reported control: '+dataset)
        for split in folds:
            variants = ([dict(k=k, variant=v) for k in [5, 10] for v in
                         (['dense', 'matrix-free'] if args.variant == 'both' else [args.variant])]
                        if args.study == 'block-solvers' else
                        [dict(method=m) for m in ['Exact-HSIC', 'Exact-A-F', 'Exact-streamed', 'RFF-HSIC']]
                        if args.study == 'storage-layout' else [{}])
            jobs.extend(dict(study=args.study, dataset=dataset, split=split, **v) for v in variants
                        if not (args.study == 'block-solvers' and args.splits is None
                                and v['variant'] == 'matrix-free' and split not in FOLDS))
    return jobs


def worker(job):
    from windows_resources import enforce_memory_limit
    if os.name == 'nt':
        enforce_memory_limit(4096)
    from reproduce import load_case
    from threadpoolctl import threadpool_limits
    # Equal imports precede all timed storage-layout methods.
    from pyHSICLasso import HSICLasso
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True):
        X, y, tr, te, fold = load_case(job['dataset'], job['split'])
        result = {'block-solvers': block_solvers, 'label-kernels': label_kernels,
                  'storage-layout': storage_layout}[job['study']](job, X, y, tr, te)
    return dict(job, status='ok', split_sha256=fold['split_sha256'],
                seed=42+job['split'], memory_limit_mib=4096 if os.name == 'nt' else None, **result)


def verify_archive():
    """Check existing records and displayed tables; never fit a selector or classifier."""
    import numpy as np
    import re
    from scipy.stats import t
    from io_utils import records as archived_records
    from reproduce import load_case
    data = json.loads((ROOT/'reported_results.json').read_text(encoding='utf-8'))
    archive = list(archived_records())
    rows = data['source_data']['implementation_controls'] + [r for r in archive if r.get('study') == 'block-solvers']
    counts = {study: sum(r['study'] == study for r in rows)
              for study in ['block-solvers', 'label-kernels', 'storage-layout']}
    if counts != {'block-solvers': 330, 'label-kernels': 20, 'storage-layout': 20}:
        raise ValueError('Incomplete archived control records.')
    checked = 0
    for row in rows:
        X, y, tr, te, fold = load_case(row['dataset'], row['split'])
        if fold['split_sha256'] != row['split_sha256']:
            raise ValueError('Control split checksum mismatch.')
        for value in row.get('evaluations', []):
            pred = np.asarray(value['predictions'])
            accuracy = float(np.mean(pred == y[te]))
            balanced = float(np.mean([np.mean(pred[y[te] == c] == c) for c in np.unique(y[te])]))
            if abs(accuracy-value['accuracy']) > 1e-12 or abs(balanced-value['balanced_accuracy']) > 1e-12:
                raise ValueError('Archived prediction metric mismatch.')
            checked += 1
        if row['study'] == 'block-solvers':
            a, b = row['evaluations']
            same_set = set(a['selected_features']) == set(b['selected_features'])
            same_predictions = a['predictions'] == b['predictions']
            if row.get('same_feature_set', same_set) != same_set or row.get('same_predictions', same_predictions) != same_predictions:
                raise ValueError('Block control agreement flag mismatch.')
            if any(e['kkt_residual'] > TOL for e in [a, b]):
                raise ValueError('Block control KKT tolerance mismatch.')
    # Recompute Table 9 from the paired fits rather than from displayed summaries.
    solver_records = [r for r in archive if r.get('study') == 'solver']
    if len(solver_records) != 400:
        raise ValueError('Incomplete Table 9 solver records.')
    for displayed in data['tables'][8]['rows'][1:]:
        group = [r for r in solver_records if r['dataset'] == displayed[0]]
        if len(group) != 100:
            raise ValueError('Table 9 requires fifty pairs per dataset.')
        methods = {method: {r['split']: r for r in group if r['solver'] == method}
                   for method in ['FISTA', 'PG']}
        if any(set(values) != set(range(50)) for values in methods.values()):
            raise ValueError('Table 9 split coverage mismatch.')
        fs = [methods['FISTA'][s]['diagnostics'] for s in range(50)]
        ps = [methods['PG'][s]['diagnostics'] for s in range(50)]
        if any(not d['converged'] or d['kkt_residual'] > TOL for d in fs+ps):
            raise ValueError('Table 9 convergence mismatch.')
        ratio = np.exp(np.mean([np.log(p['solve_seconds']/f['solve_seconds']) for f,p in zip(fs,ps)]))
        expected = [f'{np.mean([d["solve_seconds"] for d in fs]):.4f}',
                    f'{np.mean([d["solve_seconds"] for d in ps]):.4f}',
                    f'{np.mean([d["iterations"] for d in fs]):.0f}',
                    f'{np.mean([d["iterations"] for d in ps]):.0f}', f'{ratio:.2f}']
        if expected != displayed[1:]:
            raise ValueError('Table 9 summary mismatch.')
    for table_row in data['tables'][9]['rows'][1:]:
        dataset, budget = table_row[:2]
        group = [r for r in rows if r['study'] == 'block-solvers' and r['variant'] == 'dense'
                 and r['dataset'] == dataset and r['k'] == int(budget)]
        if len(group) != 50 or {r['split'] for r in group} != set(range(50)):
            raise ValueError('Table 10 requires fifty distinct folds.')
        for column, name in [(2, 'NN-LARS'), (3, 'FISTA')]:
            values = [next(e for e in r['evaluations'] if e['solver'] == name)['median_solver_seconds'] for r in group]
            if f'{np.mean(values):.4f}' != table_row[column]:
                raise ValueError('Table 10 timing mismatch.')
    for displayed in data['additional_tables'][0]['rows'][1:]:
        dataset, budget, implementation = displayed[:3]
        variant = 'dense' if implementation == 'Dense system' else 'matrix-free'
        group = [r for r in rows if r['study'] == 'block-solvers' and r['variant'] == variant
                 and r['dataset'] == dataset and r['k'] == int(budget)]
        for column, name in [(3, 'NN-LARS'), (4, 'FISTA')]:
            values = [next(e for e in r['evaluations'] if e['solver'] == name)['median_solver_seconds'] for r in group]
            if len(values) != (50 if variant == 'dense' else 5) or f'{np.mean(values):.4f}' != displayed[column]:
                raise ValueError('Additional Block solver table mismatch.')
    for displayed in data['additional_tables'][1]['rows'][1:]:
        group = [r for r in rows if r['study'] == 'label-kernels' and r['dataset'] == displayed[0]]
        for column, name in [(1,'onehot_factor'),(2,'dense_delta'),(3,'class_balanced_delta')]:
            values = [next(e for e in r['evaluations'] if e['label_mode'] == name)['balanced_accuracy'] for r in group]
            if len(values) != 5 or f'{np.mean(values):.4f}' != displayed[column]:
                raise ValueError('Additional label-kernel table mismatch.')
    # Recompute paired intervals from the existing timing archive, using five
    # repetition means of ten log ratios, rather than treating all folds as independent.
    timing = {(r['dataset'], r['method'], r['k'], r['split']): r['selector_seconds']
              for r in archived_records() if r.get('study') == 'timing' and r.get('status') == 'ok'}
    time_intervals = 0
    for displayed in data['tables'][5]['rows'][1:]:
        for column, method, budget in [(1, 'Exact-HSIC', 10), (2, 'Block-HSIC', 10), (3, 'Block-HSIC', 5)]:
            if displayed[column] == 'Unavailable':
                continue
            logs = np.log([timing[(displayed[0], method, budget, s)]/
                           timing[(displayed[0], 'RFF-HSIC', budget, s)] for s in range(50)])
            means = logs.reshape(5, 10).mean(axis=1)
            center, half = float(means.mean()), float(t.ppf(.975, 4)*means.std(ddof=1)/np.sqrt(5))
            expected = [float(f'{v:.2f}') for v in np.exp([center, center-half, center+half])]
            shown = list(map(float, re.findall(r'\d+(?:\.\d+)?', displayed[column])))
            if shown != expected:
                raise ValueError('Table 6 paired timing interval mismatch.')
            time_intervals += 1
    resource_intervals = 0
    for displayed in data['tables'][6]['rows'][1:]:
        for column, method, field, precision in [(2,'Exact-HSIC','time',4),(3,'Block-HSIC','time',4),
                (4,'RFF-HSIC','time',4),(5,'Exact-HSIC','memory',1),(6,'Block-HSIC','memory',1),(7,'RFF-HSIC','memory',1)]:
            source = next(r for r in data['source_data']['subset_resources']
                          if int(r['n']) == int(displayed[0]) and r['method'] == method)
            mean, sd = float(source[field+'_mean']), float(source[field+'_sd'])
            half = float(t.ppf(.975, 2)*sd/np.sqrt(3))
            expected = [float(f'{v:.{precision}f}') for v in [mean,sd,mean-half,mean+half]]
            shown = list(map(float, re.findall(r'-?\d+(?:\.\d+)?', displayed[column])))
            if shown != expected:
                raise ValueError('Table 7 resource interval mismatch.')
            resource_intervals += 1
    for displayed in data['tables'][12]['rows'][1:]:
        source = next(r for r in data['source_data']['nested_dimension_selection'] if r['dataset'] == displayed[0])
        expected = [f"{float(source['original_rule_accuracy_mean_same_folds']):.4f}",
                    f"{float(source['inner_selected_accuracy_mean']):.4f}",
                    f"{float(source['inner_search_seconds_mean']):.2f}"]
        if displayed[1:] != expected:
            raise ValueError('Table 13 dimension-validation mismatch.')
    return dict(control_cases=counts, verified_prediction_vectors=checked,
                manuscript_tables=len(data['tables']), recomputed_time_ratio_intervals=time_intervals,
                recomputed_resource_intervals=resource_intervals, model_fits_started=0)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '_worker':
        with contextlib.redirect_stdout(sys.stderr):
            result = worker(json.loads(argv[1]))
        print(json.dumps(result, allow_nan=False))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('study', choices=['block-solvers', 'label-kernels', 'storage-layout', 'verify'])
    p.add_argument('--variant', choices=['dense', 'matrix-free', 'both'], default='both')
    p.add_argument('--datasets'); p.add_argument('--splits')
    p.add_argument('--output', type=Path); p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if args.study == 'verify':
        print(json.dumps(verify_archive(), indent=2))
        return
    try:
        jobs = planned_jobs(args)
    except ValueError as error:
        p.error(str(error))
    if args.dry_run:
        print(json.dumps(dict(study=args.study, jobs=len(jobs), examples=jobs[:2], fits_started=0), indent=2))
        return
    if args.output is None or args.output.exists():
        p.error('Choose a new --output JSONL file.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        for number, job in enumerate(jobs, 1):
            command = [sys.executable, '-B', str(Path(__file__).resolve()), '_worker', json.dumps(job)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=600)
            if result.returncode:
                raise RuntimeError(result.stderr)
            row = json.loads(result.stdout)
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            print(f'{number}/{len(jobs)} {job}', flush=True)


if __name__ == '__main__':
    main()
