"""Run the manuscript experiments. --dry-run lists jobs without fitting models."""
import argparse
import contextlib
import gc
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '2'
from io_utils import dimension_rule

ROOT = Path(__file__).resolve().parent
METHODS = ['ANOVA','MI','L1-Linear','Exact-HSIC','Block-HSIC','RFF-HSIC']
FIVE_FOLDS = [0,10,20,30,40]


def split_list(text, default):
    if text is None:
        return list(default)
    if text == 'all':
        return list(range(50))
    values = [int(v) for v in text.split(',')]
    if not values or len(values) != len(set(values)) or any(v < 0 or v >= 50 for v in values):
        raise ValueError('Splits must be distinct indices from 0 to 49.')
    return values


def planned_jobs(args):
    with ZipFile(ROOT/'benchmark_data.zip') as archive:
        manifest = json.loads(archive.read('manifest.json'))
    available = list(manifest)
    defaults = {
        'main': available,
        'grouped': [d for d in available if 'grouped_splits' in manifest[d]],
        'ablation': [d for d in available if d != 'ORL'],
        'budget': [d for d in available if d != 'ORL'],
        'rules': ['Breast Cancer','Digits','Libras Movement','Multiple Features'],
        'nested': ['Digits','Multiple Features'],
        'solver': ['Breast Cancer','Digits','Libras Movement','ORL'],
        'resources': ['Digits'],
        'construction': ['Multiple Features'],
    }
    datasets = defaults.get(args.study, available) if args.datasets == 'all' else args.datasets.split(',')
    if any(d not in manifest for d in datasets):
        raise ValueError('Unknown dataset name.')
    if args.study == 'grouped' and any('grouped_splits' not in manifest[d] for d in datasets):
        raise ValueError('Grouped splits exist for Ionosphere, Libras Movement, and Image Segmentation.')
    jobs = []
    if args.study in ['main','grouped','ablation','budget']:
        methods = METHODS if args.methods == 'all' else args.methods.split(',')
        if args.study == 'ablation':
            methods = ['RFF-HSIC','RFF-HSIC-screen']
        if args.study == 'budget':
            methods = ['RFF-HSIC']
        if any(m not in METHODS + ['RFF-HSIC-screen'] for m in methods):
            raise ValueError('Unknown selector.')
        budgets = [10] if args.study == 'ablation' else [5,10,20] if args.study == 'budget' else args.budgets
        for dataset in datasets:
            for split in split_list(args.splits, range(50)):
                for method in methods:
                    jobs.append(dict(study=args.study,dataset=dataset,split=split,method=method,
                                     budgets=budgets,protocol='grouped' if args.study=='grouped' else 'ordinary'))
    elif args.study == 'parameters':
        axes = ['lambda','D','gamma'] if args.axis == 'all' else [args.axis]
        for axis in axes:
            use = ['Breast Cancer','Digits','Libras Movement','Semeion']
            if args.datasets != 'all':
                use = datasets
            for dataset in use:
                for split in split_list(args.splits, range(50) if axis=='lambda' else FIVE_FOLDS):
                    jobs.append(dict(study='parameters',dataset=dataset,split=split,axis=axis))
    elif args.study in ['rules','nested','solver']:
        folds = range(10) if args.study == 'nested' else FIVE_FOLDS
        for dataset in datasets:
            for split in split_list(args.splits, folds):
                jobs.append(dict(study=args.study,dataset=dataset,split=split))
    elif args.study == 'resources':
        for dataset in datasets:
            for n in args.sizes:
                for seed in args.seeds:
                    for method in ['Exact-HSIC','Block-HSIC','RFF-HSIC']:
                        jobs.append(dict(study='resources',dataset=dataset,split=0,n=n,seed=seed,method=method))
    elif args.study == 'construction':
        for dataset in datasets:
            for split in split_list(args.splits, [0]):
                for unblocked in [False,True]:
                    jobs.append(dict(study='construction',dataset=dataset,split=split,unblocked=unblocked))
    for job in jobs:
        job['memory_limit_mib'] = args.memory_mib if args.memory_mib is not None else (
            3072 if args.study == 'construction' else 4096)
    return jobs


def fit_budgets(job, X, y, tr, te, split):
    import numpy as np
    from reproduce import select_features, predict_selected
    from windows_resources import memory_info
    from threadpoolctl import threadpool_limits
    method = job['method']
    seed = 42 + job['split']
    D = dimension_rule(len(tr),X.shape[1])
    cache = None
    output = []
    with threadpool_limits(limits=2):
        for budget in job['budgets']:
            k = min(budget,X.shape[1])
            if cache is None or method == 'Block-HSIC':
                count = k if method=='Block-HSIC' else min(max(job['budgets']),X.shape[1])
                started = time.perf_counter()
                indices,diag = select_features(X[tr],y[tr],method,count,seed,D)
                elapsed = time.perf_counter()-started
                peak = memory_info()
                cache = (indices,diag,elapsed,peak)
            indices,diag,elapsed,peak = cache
            idx = indices[:k]
            output.append(dict(study='main' if job['study']=='main' else job['study'],
                dataset=job['dataset'],method=method,split=job['split'],k=k,
                protocol=job['protocol'],seed=seed,D=D,n_train=len(tr),n_test=len(te),
                split_sha256=split['split_sha256'],selected_features=idx.tolist(),
                selector_seconds=elapsed,diagnostics=diag,memory_after_selection=peak,
                **predict_selected(X[tr],X[te],y[tr],y[te],idx)))
    return output


def parameters(job, X, y, tr, te):
    from hsic import rff_system, solve_system
    from reproduce import predict_selected
    import numpy as np
    D = dimension_rule(len(tr),X.shape[1])
    axis = job['axis']
    values = [.001,.01,.1] if axis=='lambda' else [5,10,D] if axis=='D' else [.125,.5,2.]
    output = []
    # Lambda timings follow the manuscript's fresh-fit protocol, without cached Q.
    if axis == 'lambda':
        from hsic import JournalHSIC
        offset = job['split'] % len(values)
        for lam in values[offset:] + values[:offset]:
            started = time.perf_counter()
            fitted = JournalHSIC(D=D, gamma=.5, lam=lam, seed=42+job['split']).fit(X[tr],y[tr])
            elapsed = time.perf_counter()-started
            idx = fitted.get_support(min(10,X.shape[1]))
            diag = fitted.diagnostics_
            output.append(dict(study='parameters',dataset=job['dataset'],split=job['split'],
                axis=axis,value=lam,D=D,gamma=.5,lambda_value=lam,selector_seconds=elapsed,
                construction_plus_solve_seconds=diag['build_seconds']+diag['solve_seconds'],
                positive_weights=int(np.sum(fitted.alpha_>1e-10)),diagnostics=diag,
                selected_features=idx.tolist(),
                **predict_selected(X[tr],X[te],y[tr],y[te],idx)))
        return output
    cache = {}
    for value in dict.fromkeys(values):
        dim = int(value) if axis=='D' else D
        gamma = value if axis=='gamma' else .5
        lam = value if axis=='lambda' else .01
        key = (dim,gamma)
        if key not in cache:
            cache[key] = rff_system(X[tr],y[tr],dim,gamma=gamma,seed=42+job['split'])
        Q,h,build = cache[key]
        alpha,diag = solve_system(Q,h,lam)
        idx = np.argsort(-alpha,kind='stable')[:min(10,X.shape[1])]
        output.append(dict(study='parameters',dataset=job['dataset'],split=job['split'],
            axis=axis,value=value,D=dim,gamma=gamma,lambda_value=lam,
            construction_plus_solve_seconds=build['build_seconds']+diag['solve_seconds'],
            positive_weights=int(np.sum(alpha>1e-10)),diagnostics=diag,
            selected_features=idx.tolist(),
            **predict_selected(X[tr],X[te],y[tr],y[te],idx)))
    return output


def rule_diagnostics(job, X, tr):
    base = dict(D_min=10,D_max=80,budget=15000)
    variants = [('default',{})]
    for key,values in [('D_min',[5,20]),('D_max',[40,160]),('budget',[7500,30000])]:
        variants.extend((f'{key}_{value}',{key:value}) for value in values)
    output = []
    for name,change in variants:
        setting = {**base,**change}
        D = dimension_rule(len(tr),X.shape[1],setting['D_min'],setting['D_max'],setting['budget'])
        output.append(dict(study='rules',dataset=job['dataset'],split=job['split'],name=name,
                           actual_D=D,d_times_D=X.shape[1]*D,**setting))
    return output


def nested_dimension(job, X, y, tr, te):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from hsic import rff_system, solve_system
    from reproduce import predict_selected
    split = job['split']
    folds = list(StratifiedKFold(n_splits=3,shuffle=True,random_state=1042+split).split(X[tr],y[tr]))
    started = time.perf_counter()
    inner = []
    for D in [10,20,40,80]:
        rows = []
        for i,(a,b) in enumerate(folds):
            training,validation = tr[a],tr[b]
            seed = 100000+100*split+i
            Q,h,build = rff_system(X[training],y[training],D,seed=seed)
            alpha,diag = solve_system(Q,h)
            idx = np.argsort(-alpha,kind='stable')[:min(10,X.shape[1])]
            ev = predict_selected(X[training],X[validation],y[training],y[validation],idx)
            rows.append(dict(inner_fold=i,seed=seed,diagnostics=diag,build=build,**ev))
            del Q,h
            gc.collect()
        inner.append(dict(D=D,accuracy_mean=float(np.mean([r['accuracy'] for r in rows])),
                          converged=all(r['diagnostics']['converged'] for r in rows),folds=rows))
    eligible = [r for r in inner if r['converged']]
    if not eligible:
        return [dict(study='nested',dataset=job['dataset'],split=split,
                     status='no_converged_inner_candidate',inner_candidates=inner)]
    best = max(r['accuracy_mean'] for r in eligible)
    chosen = min(r['D'] for r in eligible if abs(r['accuracy_mean']-best)<=1e-12)
    search_seconds = time.perf_counter()-started
    outer = []
    for policy,D in [('inner_selected',chosen),('original_rule',dimension_rule(len(tr),X.shape[1]))]:
        Q,h,build = rff_system(X[tr],y[tr],D,seed=42+split)
        alpha,diag = solve_system(Q,h)
        idx = np.argsort(-alpha,kind='stable')[:min(10,X.shape[1])]
        outer.append(dict(policy=policy,D=D,diagnostics=diag,build=build,
            selected_features=idx.tolist(),**predict_selected(X[tr],X[te],y[tr],y[te],idx)))
    return [dict(study='nested',dataset=job['dataset'],split=split,selected_D=chosen,
                 inner_search_seconds=search_seconds,inner_candidates=inner,outer_evaluations=outer)]


def solver_comparison(job, X, y, tr, te):
    import numpy as np
    from hsic import rff_system, solve_system
    from reproduce import predict_selected
    D = dimension_rule(len(tr),X.shape[1])
    Q,h,build = rff_system(X[tr],y[tr],D,seed=42+job['split'])
    output = []
    order = ['FISTA','PG'] if job['split']%20==0 else ['PG','FISTA']
    for method in order:
        alpha,diag = solve_system(Q,h,max_iter=100000,method=method)
        idx = np.argsort(-alpha,kind='stable')[:min(10,X.shape[1])]
        output.append(dict(study='solver',dataset=job['dataset'],split=job['split'],
            solver=method,D=D,build=build,diagnostics=diag,selected_features=idx.tolist(),
            **predict_selected(X[tr],X[te],y[tr],y[te],idx)))
    return output


def resource_measurement(job, X, y, tr):
    import numpy as np
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import StratifiedShuffleSplit
    from reproduce import select_features
    from hsic import rff_system
    from windows_resources import memory_info
    if job['study']=='resources':
        idx,_ = next(StratifiedShuffleSplit(n_splits=1,train_size=job['n'],
                    random_state=job['seed']).split(X,y))
        X,y = X[idx],y[idx]
        seed = job['seed']
    else:
        X,y = X[tr],y[tr]
        seed = 42+job['split']
    D = dimension_rule(len(y),X.shape[1])
    warm = np.ones((96,96))
    _ = warm@warm
    before = memory_info()
    started = time.perf_counter()
    if job['study']=='construction':
        Q,h,diag = rff_system(X,y,D,seed=seed,unblocked=job['unblocked'])
        selected = None
    else:
        selected,diag = select_features(X,y,job['method'],min(10,X.shape[1]),seed,D)
    elapsed = time.perf_counter()-started
    after = memory_info()
    return [dict(job,D=D,n=len(y),d=X.shape[1],selector_seconds=elapsed,
        memory_before_selection=before,memory_after_selection=after,diagnostics=diag,
        selected_features=selected.tolist() if selected is not None else None)]


def worker(job):
    """Each fitted job runs in a fresh, serial child process."""
    from reproduce import load_case
    from windows_resources import enforce_memory_limit, memory_info
    if job['study'] in ['resources','construction'] and os.name!='nt':
        raise RuntimeError('The published committed-memory and peak-working-set protocol requires Windows.')
    limit = job['memory_limit_mib']
    if os.name=='nt' and limit:
        enforce_memory_limit(limit)
    import numpy as np
    import scipy
    import sklearn
    from threadpoolctl import threadpool_limits
    # Common imports precede every timed resource method.
    if job['study'] in ['resources','construction']:
        from pyHSICLasso import HSICLasso
    try:
        with threadpool_limits(limits=2):
            X,y,tr,te,fold = load_case(job['dataset'],job['split'],job.get('protocol','ordinary'))
            study = job['study']
            if study in ['main','grouped','ablation','budget']:
                output = fit_budgets(job,X,y,tr,te,fold)
            elif study=='parameters':
                output = parameters(job,X,y,tr,te)
            elif study=='rules':
                output = rule_diagnostics(job,X,tr)
            elif study=='nested':
                output = nested_dimension(job,X,y,tr,te)
            elif study=='solver':
                output = solver_comparison(job,X,y,tr,te)
            else:
                output = resource_measurement(job,X,y,tr)
        for row in output:
            row.setdefault('status','ok')
            row['memory_limit_mib'] = limit if os.name=='nt' else None
        return output
    except MemoryError as error:
        return [dict(**job,status='memory_error',error=str(error),memory_at_failure=memory_info())]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0]=='_worker':
        job = json.loads(argv[1])
        with contextlib.redirect_stdout(sys.stderr):
            rows = worker(job)
        print(json.dumps(rows,ensure_ascii=False,allow_nan=False))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('study',choices=['main','grouped','ablation','budget','parameters','rules',
                                   'nested','solver','resources','construction'])
    p.add_argument('--datasets',default='all',help='Comma-separated exact names, or all.')
    p.add_argument('--methods',default='all')
    p.add_argument('--splits',help='Comma-separated indices, or all; defaults follow the published study.')
    p.add_argument('--budgets',default='5,10')
    p.add_argument('--axis',choices=['all','lambda','D','gamma'],default='all')
    p.add_argument('--sizes',default='100,200,400,600')
    p.add_argument('--seeds',default='42,43,44')
    p.add_argument('--memory-mib',type=int,default=None)
    p.add_argument('--timeout',type=float,default=None,help='Optional per-worker timeout in seconds.')
    p.add_argument('--output',type=Path)
    p.add_argument('--dry-run',action='store_true')
    args = p.parse_args(argv)
    try:
        args.budgets = [int(v) for v in args.budgets.split(',')]
        args.sizes = [int(v) for v in args.sizes.split(',')]
        args.seeds = [int(v) for v in args.seeds.split(',')]
        if any(v<=0 for v in args.budgets+args.sizes) or not args.seeds:
            raise ValueError('Budgets and sample sizes must be positive.')
        if args.memory_mib is not None and args.memory_mib<=0:
            raise ValueError('The memory limit must be positive.')
        jobs = planned_jobs(args)
    except ValueError as error:
        p.error(str(error))
    if args.dry_run:
        print(json.dumps({'study':args.study,'jobs':len(jobs),'examples':jobs[:2],
                          'fits_started':0},indent=2))
        return
    if args.output is None:
        p.error('--output is required unless --dry-run is used.')
    if args.output.exists():
        p.error('Choose a new output file; existing results are not replaced.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    failures = 0
    with args.output.open('x',encoding='utf-8') as stream:
        for number,job in enumerate(jobs,1):
            if job['study']=='rules':
                from reproduce import load_case
                X,y,tr,te,fold = load_case(job['dataset'],job['split'])
                rows = rule_diagnostics(job,X,tr)
            else:
                command = [sys.executable,'-B',str(Path(__file__).resolve()),'_worker',json.dumps(job)]
                try:
                    result = subprocess.run(command,capture_output=True,text=True,encoding='utf-8',
                        env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'},
                        timeout=args.timeout)
                    if result.returncode:
                        rows = [dict(**job,status='worker_error',error=result.stderr[-6000:])]
                    else:
                        rows = json.loads(result.stdout)
                except subprocess.TimeoutExpired:
                    rows = [dict(**job,status='timeout',timeout_seconds=args.timeout)]
            for row in rows:
                failures += row.get('status','ok') not in ['ok','memory_error']
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush()
            print(f'{args.study}: {number}/{len(jobs)}',file=sys.stderr,flush=True)
    print(json.dumps({'output':str(args.output),'jobs':len(jobs),'failed_records':failures}))
    if failures:
        raise SystemExit(1)


if __name__=='__main__':
    main()

