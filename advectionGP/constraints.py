import numpy as np
from truncatedMVN import TruncMVNreparam as TMVN

class NonNegConstraint():
    def __init__(self,model,yTrain,Xconstrainlocs,jitter=0,thinning=1,burnin=None,verbose=False,usecaching=False):
        """
        model = an instance of one of the models from the AdvectionGP module.
        yTrain = the observations at the model's measurement locations
        Xconstrainlocs = NxD matrix of where we're forcing it to be non-negative. Ideally should be dense enough.
        thinning, burnin = Gibb's sampling configuration
        """
        self.model = model
        self.usecaching = usecaching
        if verbose: print("Computing mean and covariance of Z distribution")
        meanZ, covZ = model.computeZDistribution(yTrain)
        planes = np.array([phi for phi in model.kernel.getPhi(Xconstrainlocs.T)])
        self.tm = TMVN(meanZ,covZ+np.eye(len(covZ))*jitter,planes.T,thinning=thinning,burnin=burnin,verbose=verbose)
        
    def sample(self,Nsamples=10):
        samps = self.tm.sample(samples=Nsamples,usecaching=self.usecaching)
        return samps
    
    def check_convergence(self,Nchains=10,Nsamples=10):
        """
        
        """
        return np.max(self.tm.compute_gelman_rubin(Nchains=Nchains,Nsamples=Nsamples,usecaching=self.usecaching))
    
