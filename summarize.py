"""Verify predictions, summarize paired comparisons, and export manuscript tables."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from reproduce import load_case
from io_utils import open_records

ROOT = Path(__file__).resolve().parent


def summarize(record_path=None, allow_partial=False):
    import numpy as np
    from scipy.stats import t
    cases = defaultdict(dict)
    tested = 0
    with open_records(record_path) as stream:
        for row in map(json.loads, stream):
            if row.get('study') not in ['main','grouped'] or row.get('status','ok')!='ok':
                continue
            key = (row['study'],row['dataset'],row['method'],row['k'])
            if row['split'] in cases[key]:
                raise ValueError('Duplicate case: '+str((key,row['split'])))
            cases[key][row['split']] = row
    summary = []
    cache = {}
    for key, rows in cases.items():
        study,ds,method,k = key
        for split,row in rows.items():
            ck = (study,ds,split)
            if ck not in cache:
                X,y,tr,te,fold = load_case(ds,split,'ordinary' if study=='main' else 'grouped')
                cache[ck] = (y[te],len(tr),len(te),fold['split_sha256'],X.shape[1])
            target,ntr,nte,sha,d = cache[ck]
            assert row['split_sha256']==sha, 'Split hash mismatch'
            indices = row['selected_features']
            assert len(indices)==len(set(indices))==k, 'Invalid feature count'
            assert all(0<=v<d for v in indices), 'Feature index out of range'
            assert len(row['predictions'])==len(target), 'Prediction length mismatch'
            assert abs(float(np.mean(np.asarray(row['predictions'])==target))-row['accuracy'])<1e-12
            tested += 1
        values = [row['accuracy'] for row in rows.values()]
        if not allow_partial and len(values)!=50:
            raise ValueError('Expected 50 folds. Use --allow-partial to summarize a subset.')
        summary.append(dict(study=study,dataset=ds,method=method,k=k,n=len(values),
            mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else None))
    paired = []
    for (study,ds,method,k),a in cases.items():
        if study!='main' or method!='RFF-HSIC':
            continue
        for other in ['Exact-HSIC','Block-HSIC']:
            b = cases.get((study,ds,other,k))
            if not b:
                continue
            splits = sorted(set(a)&set(b))
            if len(splits)<2:
                continue
            delta = np.array([a[s]['accuracy']-b[s]['accuracy'] for s in splits])
            ratio = np.mean([cache[(study,ds,s)][2]/cache[(study,ds,s)][1] for s in splits])
            se = float(np.sqrt((1/len(splits)+ratio)*np.var(delta,ddof=1)))
            mean = float(np.mean(delta))
            df = len(splits)-1
            raw = float(2*t.sf(abs(mean/se),df)) if se else float(mean==0)
            jac = np.mean([len(set(a[s]['selected_features'])&set(b[s]['selected_features']))/
                           len(set(a[s]['selected_features'])|set(b[s]['selected_features'])) for s in splits])
            paired.append(dict(dataset=ds,k=k,comparator=other,folds=len(splits),df=df,
                mean=mean,se=se,low=float(mean-t.ppf(.975,df)*se),
                high=float(mean+t.ppf(.975,df)*se),raw_p=raw,jaccard=float(jac)))
    previous = 0.
    for rank,row in enumerate(sorted(paired,key=lambda r:r['raw_p'])):
        previous = max(previous,min(1,(len(paired)-rank)*row['raw_p']))
        row['holm_p'] = previous
    if record_path is None and not allow_partial:
        assert len(paired)==40
    return dict(verified_prediction_records=tested,accuracy_summaries=summary,paired_comparisons=paired)


def export_reported_tables(directory):
    tables = json.loads((ROOT/'reported_results.json').read_text(encoding='utf-8'))['tables']
    paths = [directory/f'table{i}.csv' for i in range(1,len(tables)+1)]
    if any(p.exists() for p in paths):
        raise FileExistsError('Choose a new table directory.')
    directory.mkdir(parents=True,exist_ok=True)
    for path,table in zip(paths,tables):
        with path.open('x',encoding='utf-8-sig',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow([table['caption']])
            writer.writerows(table['rows'])
    return len(tables)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,help='A newly generated JSONL, GZIP, or ZIP file.')
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--output',type=Path,help='Save the summary to a new JSON file.')
    p.add_argument('--export-tables',type=Path,help='Export the eleven archived manuscript tables as CSV.')
    args = p.parse_args()
    if args.output and args.output.exists():
        p.error('Choose a new output file.')
    result = summarize(args.records,args.allow_partial)
    if args.export_tables:
        result['archived_tables_exported'] = export_reported_tables(args.export_tables)
    text = json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as stream:
            stream.write(text+'\n')
    print(text)


if __name__=='__main__':
    main()
