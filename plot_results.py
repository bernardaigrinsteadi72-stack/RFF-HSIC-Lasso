"""Draw Figures 1-11 from the distributed numerical records; no model fitting."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile
from io_utils import records

ROOT = Path(__file__).resolve().parent
METHODS = ['ANOVA','MI','L1-Linear','Exact-HSIC','Block-HSIC','RFF-HSIC']
NAMES = dict(zip(METHODS,['ANOVA','MI','L1-Linear','HSIC Lasso','Block HSIC Lasso','RFF-HSIC Lasso']))


def render(output, formats):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from summarize import summarize
    data = json.loads((ROOT/'reported_results.json').read_text(encoding='utf-8'))
    source = data['source_data']
    with ZipFile(ROOT/'benchmark_data.zip') as archive:
        datasets = list(json.loads(archive.read('manifest.json')))
    all_rows = list(records())
    stats = summarize()
    pairs = stats['paired_comparisons']
    plt.rcParams.update({'font.family':'Arial','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none'})

    def save(fig, number):
        for ext in formats:
            fig.savefig(output/f'figure{number}.{ext}',dpi=600,facecolor='white')
        plt.close(fig)

    for number,comparator in [(1,'Exact-HSIC'),(2,'Block-HSIC')]:
        ds = [d for d in datasets if any(r['dataset']==d and r['comparator']==comparator for r in pairs)]
        fig,ax = plt.subplots(figsize=(7.05,4.2),layout='constrained')
        for k,color,marker,offset in [(10,'#276c95','o',-.12),(5,'#b66b1e','s',.12)]:
            rs = [next(r for r in pairs if r['dataset']==d and r['comparator']==comparator and r['k']==k) for d in ds]
            mean = np.array([r['mean'] for r in rs])*100
            low = np.array([r['low'] for r in rs])*100
            high = np.array([r['high'] for r in rs])*100
            ax.errorbar(mean,np.arange(len(ds))+offset,xerr=[mean-low,high-mean],
                fmt=marker,color=color,capsize=2,ms=4,label=f'Top-{k}')
        ax.set_yticks(range(len(ds)),ds)
        ax.invert_yaxis()
        ax.axvline(0,color='#777',ls='--',lw=.8)
        ax.grid(axis='x',alpha=.18)
        ax.set_xlabel('Accuracy difference (percentage points)')
        ax.set_title('RFF-HSIC Lasso - '+NAMES[comparator],fontsize=11)
        ax.legend(frameon=False,loc='lower right')
        save(fig,number)

    for number,k in [(3,10),(4,5)]:
        fig,axes = plt.subplots(2,3,figsize=(7.05,4.25),layout='constrained')
        for ax,method in zip(axes.flat,METHODS):
            rows = [r for r in all_rows if r['study']=='main' and r['dataset']=='Digits'
                    and r['method']==method and r['k']==k]
            freq = np.zeros(64)
            for row in rows:
                freq[row['selected_features']] += 1/len(rows)
            im = ax.imshow(freq.reshape(8,8),vmin=0,vmax=1,cmap='Blues',interpolation='nearest')
            ax.set_title(NAMES[method],fontsize=10)
            ax.set_xticks([0,7])
            ax.set_yticks([0,7])
            ax.tick_params(length=0)
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Selection frequency (50 training folds)',shrink=.85)
        save(fig,number)

    for number,k in [(5,10),(6,5)]:
        rows = [r for r in pairs if r['comparator']=='Exact-HSIC' and r['k']==k]
        ds = [d for d in datasets if any(r['dataset']==d for r in rows)]
        fig,ax = plt.subplots(figsize=(7.05,3.8),layout='constrained')
        for i,dataset in enumerate(ds):
            value = next(r['jaccard'] for r in rows if r['dataset']==dataset)
            ax.barh(i,value,height=.57,color='#32749d')
            ax.text(value-.016,i,f'{value:.3f}',ha='right',va='center',color='white',fontsize=9)
        ax.set_yticks(range(len(ds)),ds)
        ax.invert_yaxis()
        ax.set_xlim(0,1)
        ax.set_xlabel(f'Mean top-{k} Jaccard overlap with RFF-HSIC Lasso')
        ax.grid(axis='x',alpha=.15)
        ax.set_axisbelow(True)
        save(fig,number)

    times = source['selection_times']
    for number,k in [(7,10),(8,5)]:
        fig,ax = plt.subplots(figsize=(7.05,4.8),layout='constrained')
        for i,dataset in enumerate(datasets):
            points = []
            for method,color,marker in [('Exact-HSIC','#6a4c93','D'),
                                        ('Block-HSIC','#b97829','s'),
                                        ('RFF-HSIC','#276c95','o')]:
                row = next((r for r in times if r['dataset']==dataset and r['method']==method and int(r['k'])==k),None)
                if row is not None:
                    value = float(row['mean'])
                    points.append(value)
                    ax.scatter(value,i,color=color,marker=marker,s=27,zorder=3,
                               label=NAMES[method] if i==0 else None)
            ax.plot(points,[i]*len(points),color='#aaa',lw=1,zorder=1)
        ax.set_yticks(range(len(datasets)),datasets)
        ax.invert_yaxis()
        ax.set_xscale('log')
        ax.set_xlabel('Mean selection time (s)')
        ax.grid(axis='x',alpha=.18)
        ax.set_title(f'Top-{k}',fontsize=11)
        handles,labels = ax.get_legend_handles_labels()
        fig.legend(handles,labels,frameon=False,loc='outside upper center',ncol=3,fontsize=9)
        save(fig,number)

    datasets_lambda = ['Breast Cancer','Libras Movement','Semeion']
    for number,metric,ylabel in [(9,'accuracy','Classification accuracy'),
                                (10,'selector_seconds','Selection time (s)'),
                                (11,'positive_weights','Nonzero coefficients')]:
        fig,axes = plt.subplots(1,3,figsize=(7.05,2.8),layout='constrained')
        for ax,dataset in zip(axes,datasets_lambda):
            means,errors = [],[]
            for lam in [.001,.01,.1]:
                rows = [r for r in all_rows if r['study']=='lambda_sensitivity'
                        and r['dataset']==dataset and r['lambda']==lam]
                values = [r[metric] for r in rows]
                if len(values)!=50:
                    raise ValueError('Expected 50 regularization records per configuration.')
                means.append(float(np.mean(values)))
                errors.append(float(np.std(values,ddof=1)))
            ax.errorbar([.001,.01,.1],means,yerr=errors,marker='o',capsize=3,color='#276c95')
            ax.set_xscale('log')
            ax.set_xticks([.001,.01,.1],['0.001','0.01','0.1'])
            ax.set_title(dataset,fontsize=10)
            ax.set_xlabel(r'$\lambda$')
            ax.grid(alpha=.18)
        axes[0].set_ylabel(ylabel)
        save(fig,number)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('figures'))
    p.add_argument('--formats',default='png,pdf')
    p.add_argument('--dry-run',action='store_true')
    args = p.parse_args()
    formats = args.formats.split(',')
    if any(ext not in ['png','pdf','svg'] for ext in formats):
        p.error('formats must be chosen from png,pdf,svg')
    targets = [args.output/f'figure{i}.{ext}' for i in range(1,12) for ext in formats]
    if args.dry_run:
        print(json.dumps({'figures':11,'files':[str(t) for t in targets],'fits_started':0},indent=2))
        return
    if any(t.exists() for t in targets):
        p.error('Choose a new output directory; existing figures are not replaced.')
    args.output.mkdir(parents=True,exist_ok=True)
    render(args.output,formats)
    print(json.dumps({'output':str(args.output),'figures':11,'formats':formats}))


if __name__=='__main__':
    main()

