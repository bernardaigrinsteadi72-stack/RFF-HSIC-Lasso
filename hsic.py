"""Algorithms used by the manuscript; extracted without changing numerical operations."""
import numpy as np

class RandomFourierFeatures:


    def __init__(
        self,
        n_components=1000,
        gamma=1.0,
        random_state=42
    ):

        self.D=n_components
        self.gamma=gamma
        self.random_state=random_state



    def fit(self,X):

        rng=np.random.RandomState(
            self.random_state
        )


        d=X.shape[1]


        self.W=np.sqrt(
            2*self.gamma
        )*rng.normal(
            size=(d,self.D)
        )


        self.b=rng.uniform(
            0,
            2*np.pi,
            self.D
        )


        return self



    def transform(self,X):


        Z=np.sqrt(
            2/self.D
        )*np.cos(
            X@self.W+self.b
        )


        return Z



    def fit_transform(self,X):

        self.fit(X)

        return self.transform(X)

class RFF_HSIC_Lasso:
    FEATURE_BLOCK_SIZE = 20

    @staticmethod
    def normalize_kernel_factor(Z):
        """Normalize ``Z Zᵀ`` to unit Frobenius norm without forming it.

        Standard HSIC-Lasso uses centered, Frobenius-normalized Gram matrices.
        For an RFF factor this norm equals ``||Zᵀ Z||_F`` and can be computed
        in the compact feature space.
        """
        gram = Z.T @ Z
        norm = np.linalg.norm(gram, ord="fro")
        return Z if norm < 1e-12 else Z / np.sqrt(norm)

    @classmethod
    def _build_Q_chunked(cls, Zx_list, d, D):
        """
        Build Q from feature blocks rather than one ``dD × dD`` Gram matrix.

        It computes exactly the same Q as the one-shot Gram construction, but
        bounds its largest temporary matrix by ``(B D)²`` instead of ``(d D)²``.
        """
        Q = np.empty((d, d), dtype=float)
        block = min(cls.FEATURE_BLOCK_SIZE, d)
        for start_j in range(0, d, block):
            end_j = min(start_j + block, d)
            Mj = np.hstack(Zx_list[start_j:end_j])
            bj = end_j - start_j
            for start_k in range(start_j, d, block):
                end_k = min(start_k + block, d)
                Mk = Mj if start_k == start_j else np.hstack(Zx_list[start_k:end_k])
                bk = end_k - start_k
                cross = Mj.T @ Mk
                q_block = (cross.reshape(bj, D, bk, D) ** 2).sum(axis=(1, 3))
                Q[start_j:end_j, start_k:end_k] = q_block
                if start_j != start_k:
                    Q[start_k:end_k, start_j:end_j] = q_block.T
                del cross, q_block
                if start_k != start_j:
                    del Mk
            del Mj
        return Q

"""Matched exact/RFF normalized HSIC objectives for the journal experiments.

Both solve alpha.T Q alpha - 2 h.T alpha + lambda * sum(alpha), alpha>=0.
The archived implementations remain unchanged for old-result traceability.
"""
import time
import numpy as np
from sklearn.preprocessing import StandardScaler

def solve(Q, h, lam=0.01, max_iter=10000, tol=1e-6):
    # Symmetric normalized Gram Q is PSD. Infinity norm bounds its spectral norm.
    L = max(2*np.abs(Q).sum(axis=1).max(), 1e-12)
    x = np.zeros(len(h)); z = x.copy(); t = 1.0
    for iteration in range(1, max_iter+1):
        new = np.maximum(z - (2*(Q@z-h)+lam)/L, 0)
        tn = (1+np.sqrt(1+4*t*t))/2
        z = new + (t-1)/tn*(new-x)
        x, t = new, tn
        if iteration % 10 == 0:
            residual = np.max(np.abs(x-np.maximum(x-(2*(Q@x-h)+lam),0)))
            if residual <= tol:
                break
    residual = np.max(np.abs(x-np.maximum(x-(2*(Q@x-h)+lam),0)))
    return x, {'iterations':iteration,'kkt_residual':float(residual), 'converged':bool(residual<=tol)}

class JournalHSIC:
    def __init__(self, exact=False, D=30, gamma=0.5, lam=0.01, seed=42):
        self.exact,self.D,self.gamma,self.lam,self.seed=exact,D,gamma,lam,seed

    def fit(self,X,y):
        start=time.perf_counter()
        X=StandardScaler().fit_transform(X)
        n,d=X.shape
        _,yi=np.unique(y,return_inverse=True)
        Y=np.eye(yi.max()+1)[yi]; Y-=Y.mean(axis=0)
        Y=RFF_HSIC_Lasso.normalize_kernel_factor(Y)
        if self.exact:
            A=np.empty((n*n,d))
            label=(Y@Y.T).reshape(-1)
            for j in range(d):
                v=X[:,j]
                K=np.exp(-self.gamma*(v[:,None]-v[None,:])**2)
                K=K-K.mean(axis=0)[None,:]-K.mean(axis=1)[:,None]+K.mean()
                norm=np.linalg.norm(K)
                A[:,j]=(K/max(norm,1e-12)).ravel()
            Q=A.T@A; h=A.T@label
        else:
            rff=RandomFourierFeatures(self.D,self.gamma,self.seed)
            rff.fit(X[:,:1]); Z=[]; h=np.empty(d)
            for j in range(d):
                z=rff.transform(X[:,j:j+1]); z-=z.mean(axis=0)
                z=RFF_HSIC_Lasso.normalize_kernel_factor(z)
                h[j]=np.sum((z.T@Y)**2); Z.append(z)
            class SmallBlocks(RFF_HSIC_Lasso):
                FEATURE_BLOCK_SIZE=20
            Q=SmallBlocks._build_Q_chunked(Z,d,self.D)
        built=time.perf_counter()
        self.alpha_,self.diagnostics_=solve(Q,h,self.lam)
        self.diagnostics_.update(build_seconds=built-start,solve_seconds=time.perf_counter()-built)
        self.Q_,self.h_=Q,h
        return self

    def get_support(self,k):
        return np.argsort(-self.alpha_,kind='stable')[:k]


def labels(y,mode="onehot"):
    import numpy as np
    _,index=np.unique(y,return_inverse=True)
    Y=np.eye(index.max()+1)[index]
    if mode=="balanced":Y=Y/np.sqrt(np.bincount(index))[None,:]
    Y-=Y.mean(axis=0)
    return RFF_HSIC_Lasso.normalize_kernel_factor(Y)


def factors(X,y,D,gamma,seed,label_mode="onehot"):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    scaled=StandardScaler().fit_transform(X);Y=labels(y,label_mode)
    mapping=RandomFourierFeatures(D,gamma,seed);mapping.fit(scaled[:,:1])
    Z=[];h=np.empty(scaled.shape[1]);norms=[]
    for j in range(scaled.shape[1]):
        z=mapping.transform(scaled[:,j:j+1]);z-=z.mean(axis=0)
        norms.append(float(np.linalg.norm(z.T@z)))
        z=RFF_HSIC_Lasso.normalize_kernel_factor(z);Z.append(z);h[j]=np.sum((z.T@Y)**2)
    return Z,h,Y,norms


def q_from_factors(Z,D,block=20,unblocked=False):
    import numpy as np
    d=len(Z)
    if unblocked:
        M=np.hstack(Z);cross=M.T@M
        return (cross.reshape(d,D,d,D)**2).sum(axis=(1,3))
    class ConfiguredBlocks(RFF_HSIC_Lasso):FEATURE_BLOCK_SIZE=block
    return ConfiguredBlocks._build_Q_chunked(Z,d,D)


def rff_system(X,y,D,gamma=.5,seed=42,label_mode="onehot",block=20,unblocked=False,screen=None):
    import numpy as np
    began=time.perf_counter();Z,h,Y,norms=factors(X,y,D,gamma,seed,label_mode)
    selected=np.arange(X.shape[1])
    if screen is not None:
        selected=np.argsort(-h,kind="stable")[:min(screen,X.shape[1])]
        Z=[Z[i] for i in selected];h=h[selected]
    Q=q_from_factors(Z,D,block,unblocked)
    return Q,h,{"build_seconds":time.perf_counter()-began,"original_indices":selected.tolist(),"D":D,"gamma":gamma,"label_mode":label_mode,"block_size":block,"unblocked":unblocked,"screening_budget":screen,"near_zero_feature_factor_norms":sum(v<1e-12 for v in norms),"min_factor_gram_norm":min(norms),"max_factor_gram_norm":max(norms)}


def solve_system(Q,h,lam=.01,max_iter=10000,tol=1e-6,method="FISTA"):
    import numpy as np
    began=time.perf_counter()
    if method=="FISTA":alpha,diag=solve(Q,h,lam,max_iter=max_iter,tol=tol)
    else:
        L=max(2*np.abs(Q).sum(axis=1).max(),1e-12);alpha=np.zeros(len(h))
        for iteration in range(1,max_iter+1):
            alpha=np.maximum(alpha-(2*(Q@alpha-h)+lam)/L,0)
            if iteration%10==0:
                residual=np.max(np.abs(alpha-np.maximum(alpha-(2*(Q@alpha-h)+lam),0)))
                if residual<=tol:break
        residual=np.max(np.abs(alpha-np.maximum(alpha-(2*(Q@alpha-h)+lam),0)))
        diag={"iterations":iteration,"kkt_residual":float(residual),"converged":bool(residual<=tol)}
    diag={**diag,"solve_seconds":time.perf_counter()-began,"solver":method,"lambda":lam,"objective_without_label_constant":float(alpha@Q@alpha-2*h@alpha+lam*alpha.sum()),"coefficient_l1":float(alpha.sum()),"positive_coefficients":int(np.sum(alpha>1e-10)),"max_iter":max_iter,"tol":tol}
    return alpha,diag

