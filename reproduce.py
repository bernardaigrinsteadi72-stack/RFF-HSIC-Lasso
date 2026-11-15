"""Refit a paper selector and RBF-SVM on an unchanged frozen split."""
import argparse,gzip,hashlib,json,os,sys,time,warnings
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from io_utils import records
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
ROOT=Path(__file__).resolve().parent

def load_case(dataset,split,protocol='ordinary'):
    import numpy as np
    with ZipFile(ROOT/'benchmark_data.zip') as archive:
        manifest=json.loads(archive.read('manifest.json'))
        if dataset not in manifest:raise ValueError('Unknown dataset: '+dataset)
        info=manifest[dataset]
        with np.load(BytesIO(archive.read(info['array'])),allow_pickle=False) as data:
            X,y=data['X'],data['y']
        if hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()!=info['data_sha256']:
            raise ValueError('Input checksum mismatch')
        source=info['splits'] if protocol=='ordinary' else info['grouped_splits']
        row=next(r for r in map(json.loads,archive.read(source).decode('utf-8').splitlines()) if r['split']==split)
    train=np.asarray(row['train_indices'],dtype=np.int64)
    test=np.asarray(row['test_indices'],dtype=np.int64)
    if hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest()!=row['split_sha256']:
        raise ValueError('Split checksum mismatch')
    return X,y,train,test,row

def select_features(X, y, method, k, seed=42, D=None, gamma=.5, lam=.01):
    """Fit selection on the supplied training observations only."""
    import numpy as np
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from hsic import JournalHSIC
    from io_utils import dimension_rule
    k = min(k, X.shape[1])
    D = dimension_rule(len(y), X.shape[1]) if D is None else D
    diagnostics = {}
    if method in ['RFF-HSIC', 'Exact-HSIC', 'RFF-HSIC-screen']:
        obj = JournalHSIC(exact=method == 'Exact-HSIC', D=D, gamma=gamma,
                          lam=lam, seed=seed).fit(X, y)
        scores = obj.h_ if method == 'RFF-HSIC-screen' else obj.alpha_
        diagnostics = obj.diagnostics_
    elif method == 'Block-HSIC':
        from pyHSICLasso import HSICLasso
        np.random.seed(seed)
        obj = HSICLasso()
        obj.input(StandardScaler().fit_transform(X), y)
        obj.classification(k, B=20, M=3, n_jobs=1, discrete_x=False)
        scores = np.asarray(obj.beta).reshape(-1)
        diagnostics = {'solver': 'upstream NN-LARS', 'sample_block': 20, 'permutations': 3}
    elif method == 'ANOVA':
        scores = f_classif(X, y)[0]
    elif method == 'MI':
        scores = mutual_info_classif(X, y, random_state=seed)
    elif method == 'L1-Linear':
        obj = LinearSVC(penalty='l1', loss='squared_hinge', C=.1, dual=False,
                        tol=1e-4, max_iter=10000, random_state=seed)
        obj.fit(StandardScaler().fit_transform(X), y)
        scores = np.linalg.norm(obj.coef_, axis=0)
    else:
        raise ValueError('Unknown method: ' + method)
    idx = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind='stable')[:k]
    return idx, diagnostics


def predict_selected(Xtr, Xte, ytr, yte, indices):
    import numpy as np
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    model = make_pipeline(StandardScaler(), SVC(C=1, kernel='rbf', gamma='scale',
                                               decision_function_shape='ovr'))
    model.fit(Xtr[:, indices], ytr)
    pred = model.predict(Xte[:, indices])
    return {'predictions': pred.tolist(), 'accuracy': float(np.mean(pred == yte))}


def fit_case(dataset='Wine',split=0,method='RFF-HSIC',budget=10,protocol='ordinary',D=None,gamma=.5,lam=.01):
    import numpy as np
    from sklearn.feature_selection import f_classif,mutual_info_classif
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC,LinearSVC
    from threadpoolctl import threadpool_limits
    from hsic import JournalHSIC
    X,y,tr,te,fold=load_case(dataset,split,protocol)
    seed=42+split;k=min(budget,X.shape[1])
    dim=D if D is not None else min(80,max(10,int(np.floor(min(np.sqrt(len(tr)),15000/X.shape[1])))))
    diagnostics={}
    with threadpool_limits(limits=2),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');started=time.perf_counter()
        idx,diagnostics=select_features(X[tr],y[tr],method,k,seed,dim,gamma,lam)
        elapsed=time.perf_counter()-started
        model=make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',gamma='scale',decision_function_shape='ovr'))
        model.fit(X[tr][:,idx],y[tr]);pred=model.predict(X[te][:,idx])
    return {'dataset':dataset,'split':split,'method':method,'k':k,'protocol':protocol,'seed':seed,'D':dim,
        'gamma':gamma,'lambda':lam,'split_sha256':fold['split_sha256'],'selected_features':idx.tolist(),
        'predictions':pred.tolist(),'accuracy':float(np.mean(pred==y[te])),
        'selection_seconds':elapsed,'diagnostics':diagnostics,'warnings':[str(w.message) for w in caught]}

def check_reference(result):
    study='main' if result['protocol']=='ordinary' else 'grouped'
    for row in records():
            if all(row.get(k)==v for k,v in {'study':study,'dataset':result['dataset'],'split':result['split'],'method':result['method'],'k':result['k']}.items()):
                return {'selected_features_equal':row['selected_features']==result['selected_features'],
                    'predictions_equal':row['predictions']==result['predictions'],'accuracy_equal':abs(row['accuracy']-result['accuracy'])<1e-12}
    raise ValueError('No archived case for this configuration')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='Wine');p.add_argument('--split',type=int,default=0)
    p.add_argument('--method',choices=['ANOVA','MI','L1-Linear','Exact-HSIC','Block-HSIC','RFF-HSIC','RFF-HSIC-screen'],default='RFF-HSIC')
    p.add_argument('--budget',type=int,default=10)
    p.add_argument('--protocol',choices=['ordinary','grouped'],default='ordinary')
    p.add_argument('--D',type=int,default=None);p.add_argument('--gamma',type=float,default=.5);p.add_argument('--lam',type=float,default=.01)
    p.add_argument('--check-reference',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args()
    if not 0<=args.split<50:p.error('split must be between 0 and 49')
    if args.budget<1:p.error('budget must be positive')
    if args.D is not None and args.D<1:p.error('D must be positive')
    if args.gamma<=0 or args.lam<0:p.error('gamma must be positive and lam nonnegative')
    if args.check_reference and (args.D is not None or args.gamma!=.5 or args.lam!=.01):
        p.error('Reference checking requires the original dimension rule and parameters.')
    result=fit_case(args.dataset,args.split,args.method,args.budget,args.protocol,args.D,args.gamma,args.lam)
    if args.check_reference:
        result['reference_check']=check_reference(result)
        if not all(result['reference_check'].values()):raise RuntimeError(json.dumps(result))
    text=json.dumps(result,indent=2,ensure_ascii=False)
    if args.output:
        with args.output.open('x',encoding='utf-8') as out:out.write(text+'\n')
    print(text)
if __name__=='__main__':main()
