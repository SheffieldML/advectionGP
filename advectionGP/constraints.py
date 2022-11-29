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
        
    def sample(self,Nsamples=10):
        if self.verbose: print("Sampling...")
        samps = self.tm.sample(samples=Nsamples,usecaching=self.usecaching,W=self.model.kernel.W)
        return samps
    
    def check_convergence(self,Nchains=10,Nsamples=10):
        """
        
        """
        return np.max(self.tm.compute_gelman_rubin(Nchains=Nchains,Nsamples=Nsamples,usecaching=self.usecaching))
    
