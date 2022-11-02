import numpy as np
from truncatedMVN import TruncMVN

class NonNegConstraint():
    def __init__(self,model,yTrain,Xconstrainlocs,thinning=1,burnin=None):
        """
        model = an instance of one of the models from the AdvectionGP module.
        yTrain = the observations at the model's measurement locations
        Xconstrainlocs = NxD matrix of where we're forcing it to be non-negative. Ideally should be dense enough.
        thinning, burnin = Gibb's sampling configuration
        """
        self.model = model
        meanZ, covZ = model.computeZDistribution(yTrain)
        planes = np.array([phi for phi in model.kernel.getPhi(Xconstrainlocs.T)])
        self.tm = TruncMVN(meanZ,covZ,planes.T,thinning=thinning,burnin=None)
        
    def sample(self,Nsamples=10):
        samps = self.tm.sample(samples=Nsamples)
        return samps
    
    def check_convergence(self,Nchains=10):
        """
        
        """
        return np.max(self.tm.compute_gelman_rubin(Nchains=Nchains))
    
