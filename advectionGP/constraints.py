import numpy as np
from truncatedMVN import TruncMVNreparam as TMVN

class NonNegConstraint():
    def __init__(self,model,yTrain,Xconstrainlocs,jitter=0,thinning=1,burnin=None,verbose=False,usecaching=False,meanZ=None,covZ=None,startpointnormalised=False):
        """
        model = an instance of one of the models from the AdvectionGP module.
        yTrain = the observations at the model's measurement locations
        Xconstrainlocs = NxD matrix of where we're forcing it to be non-negative. Ideally should be dense enough.
        thinning, burnin = Gibb's sampling configuration
        """
        self.model = model
        self.usecaching = usecaching
        self.verbose = verbose
        if self.verbose: print("Computing mean and covariance of Z distribution")
        if meanZ is None:
            meanZ, covZ = model.computeZDistribution(yTrain)
        planes = np.array([phi for phi in model.kernel.getPhi(Xconstrainlocs.T)])
        if self.verbose: print("Instantiating Truncated MVN object")
        self.tm = TMVN(meanZ,covZ+np.eye(len(covZ))*jitter,planes.T,thinning=thinning,burnin=burnin,verbose=verbose,startpointnormalised=startpointnormalised)
        if self.verbose: print("Instantiation Complete")
        
    def sample(self,Nsamples=10):#,use_sparse_startpoint=False):
        if self.verbose: print("Sampling...")
        #use_sparse_startpoint disabled
          #passing W allows the sampler to use a sparsely computed start point (i.e. only uses a subset of dimensions
          #with low frequencies, setting the rest to zero).
        #if use_sparse_startpoint:
        #    W = self.model.kernel.W
        #else:
        #    W = None
        samps = self.tm.sample(samples=Nsamples,usecaching=self.usecaching)#,W=W)
        return samps
    
    def check_convergence(self,Nchains=10,Nsamples=10):
        """
        
        """
        return np.max(self.tm.compute_gelman_rubin(Nchains=Nchains,Nsamples=Nsamples,usecaching=self.usecaching))
    

def equality_constraint(self,model,m,c,knownS,newS):
    """
    Compute the new (posterior) mean and covariance of Z, given parameters:
     model = the model used (this allows us to get phi)
     meanZ, covZ = the mean and covariance of Z (without the constraint)
     knownS = locations, specified by indices of rows in model.coords, that we wish to specify.
     newS = values at these points
   
    Returns:
     meanZ, covZ = the new mean and covariance of Z
    """
    Phi = []
    for i,phi in enumerate(model.kernel.getPhi1D(model.coords)):
        Phi.append(phi[:,0])
    Phi = np.array(Phi).T
    
    inv = np.linalg.inv(Phi[knownS,:] @ c @ Phi[knownS,:].T + 0.1*np.eye(len(knownS)))
    newmean = m - (Phi[knownS,:] @ c).T @ inv @ (Phi[knownS,:] @ m - newS)
    newcov = c - (Phi[knownS,:] @ c).T @ inv @ (Phi[knownS,:] @ c)
    return newmean, newcov
