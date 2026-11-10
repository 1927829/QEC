"""
No idea what the code is doing here, I'll just treat it as a black box
"""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from numba import njit
@njit(cache=True)
def _relay_kernel(indptr,cols,prior,syndrome,seed,sets,nphys,order,split,schedule,carry,aux_scale):
    """Frozen-per-sweep Relay bias; two-block Jacobi or serial layered min-sum.

    Every global sweep visits every check once. Priors and constraints are
    unchanged; only message scheduling, leg reset and auxiliary memory vary.
    """
    n=len(prior);m=len(syndrome);edges=len(cols);clip=1e6
    zero=True
    for r in range(m):
        if syndrome[r]:zero=False;break
    if zero:return np.zeros(n,np.uint8),True,0,0,0,0,False,prior.copy()
    np.random.seed(seed)
    post=prior.copy();check=np.zeros(edges);proposal=np.zeros(edges);bias=np.zeros(n);gamma=np.zeros(n)
    total=0;saturated=False
    for leg in range(sets+1):
        for c in range(n):
            g=.125 if leg==0 else -.24+.9*np.random.random()
            if c>=nphys:g*=aux_scale
            gamma[c]=g
        if leg==0 or not carry:check[:]=0.
        for it in range(80 if leg==0 else 60):
            # Evaluate memory once, not once per row, so a variable's degree
            # does not change how many times its Relay memory is applied.
            for c in range(n):bias[c]=(1-gamma[c])*prior[c]+gamma[c]*post[c]
            post[:]=bias
            for k in range(edges):post[cols[k]]+=check[k]
            for c in range(n):
                if abs(post[c])>clip:saturated=True
                post[c]=min(max(post[c],-clip),clip)
            groups=2 if schedule==1 else m
            for group in range(groups):
                start=(0 if group==0 else split) if schedule==1 else group
                end=(split if group==0 else m) if schedule==1 else group+1
                # All proposals in this group use the group's old marginals.
                for q in range(start,end):
                    r=order[q];a=indptr[r];b=indptr[r+1]
                    small=1e300;second=1e300;imin=-1;negative=int(syndrome[r])
                    for k in range(a,b):
                        v=post[cols[k]]-check[k]
                        if abs(v)>clip:saturated=True
                        v=min(max(v,-clip),clip);av=abs(v);negative+=int(v<0)
                        if av<small:second=small;small=av;imin=k
                        elif av<second:second=av
                    for k in range(a,b):
                        v=min(max(post[cols[k]]-check[k],-clip),clip)
                        value=second if k==imin else small
                        if (negative-int(v<0))%2:value=-value
                        if abs(value)>clip:saturated=True
                        proposal[k]=min(max(value,-clip),clip)
                for q in range(start,end):
                    r=order[q]
                    for k in range(indptr[r],indptr[r+1]):
                        c=cols[k];post[c]+=proposal[k]-check[k];check[k]=proposal[k]
                        if schedule==2:
                            if abs(post[c])>clip:saturated=True
                            post[c]=min(max(post[c],-clip),clip)
                if schedule==1:
                    for c in range(n):
                        if abs(post[c])>clip:saturated=True
                        post[c]=min(max(post[c],-clip),clip)
            total+=1;good=True
            for r in range(m):
                parity=0
                for k in range(indptr[r],indptr[r+1]):parity^=int(post[cols[k]]<0)
                if parity!=syndrome[r]:good=False;break
            if good:
                hard=np.zeros(n,np.uint8)
                for c in range(n):hard[c]=int(post[c]<0)
                return hard,True,total,leg+1,1,total,saturated,post
    return np.zeros(n,np.uint8),False,total,sets+1,0,0,saturated,post


@dataclass(frozen=True)
class DecodeResult:
    """Only use correction when converged is True.

    A converged correction matches the syndrome; this does not guarantee the
    correct logical class. iterations counts full check sweeps, not row updates.
    """
    correction: np.ndarray
    converged: bool
    iterations: int
    legs: int
    hit_llr_bound: bool
    belief_llr: np.ndarray


class AdaptedRelayDecoder:
    """Reusable decoder for an ORIGINAL graph, with one probability per column.

    This wrapper treats every column as a physical fault variable. Do not pass
    an auxiliary-augmented graph to this original-graph preset.

    Fixed settings from the experiment:
      - serial layered check updates in the input row order;
      - check-to-variable messages retained across relay legs;
      - physical gamma .125 initially, then iid uniform[-.24,.66] per leg;
      - 80 sweeps in the first leg, 60 in subsequent legs, LLR bound 1e6;
      - stop at the first syndrome-valid correction, no fallback.

    Messages reset for each NEW syndrome; retention is only between legs of
    the same decoding call. Use one decoder object per concurrent worker.
    """

    def __init__(self, parity_check, error_probabilities, *, max_legs=301, seed=271828):
        h=sp.csr_matrix(parity_check,copy=True)
        h.sum_duplicates();h.eliminate_zeros();h.sort_indices()
        if h.shape[0]<1 or h.shape[1]<1 or not np.all(h.data==1):
            raise ValueError('parity_check must be a nonempty binary matrix.')
        self.H=h.astype(np.uint8)
        p=np.asarray(error_probabilities,dtype=np.float64)
        if p.ndim!=1 or len(p)!=self.H.shape[1]:
            raise ValueError('Provide one fault probability per matrix column.')
        if not np.all(np.isfinite(p)&(p>0)&(p<1)):
            raise ValueError('Fault probabilities must be finite and strictly between 0 and 1.')
        if not isinstance(max_legs,(int,np.integer)) or isinstance(max_legs,(bool,np.bool_)) or max_legs<1:
            raise ValueError('max_legs must be a positive integer.')
        self.error_probabilities=p.copy()
        self._prior=np.ascontiguousarray(np.log((1-p)/p))
        self._order=np.arange(self.H.shape[0],dtype=np.int32)
        self.max_legs=int(max_legs);self.seed=int(seed);self._calls=0

    def decode(self, syndrome, *, seed=None):
        """Return correction and status for one binary syndrome.

        Default seeds are constructor seed + call number. Supply seed explicitly
        to reproduce a particular call. A failed result contains no valid
        correction; check result.converged before using result.correction.
        """
        s=np.asarray(syndrome)
        if s.ndim!=1 or len(s)!=self.H.shape[0] or not np.all((s==0)|(s==1)):
            raise ValueError('syndrome must be a binary vector with one entry per check row.')
        s=np.ascontiguousarray(s,dtype=np.uint8)
        used_seed=(self.seed+self._calls if seed is None else int(seed))%(2**32)
        self._calls+=1
        out=_relay_kernel(self.H.indptr,self.H.indices,self._prior,s,used_seed,
            self.max_legs-1,self.H.shape[1],self._order,0,2,True,.5)
        hard,ok,iterations,legs,_,_,saturated,belief=out
        if ok and not np.array_equal((self.H@hard)%2,s):
            raise RuntimeError('Internal error: returned correction does not match syndrome.')
        return DecodeResult(hard,bool(ok),int(iterations),int(legs),bool(saturated),belief)

    def warm_up(self):
        """Compile the kernel before timing; does not consume a decoding seed."""
        _relay_kernel(self.H.indptr,self.H.indices,self._prior,
            np.zeros(self.H.shape[0],dtype=np.uint8),0,0,self.H.shape[1],self._order,0,2,True,.5)
