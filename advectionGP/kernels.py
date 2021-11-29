import numpy as np

class Kernel():
    def __init__(self):
        assert False, "Not implemented" #TODO Turn into an exception
    def generateFeatures(self,N_D,N_feat):
        assert False, "Not implemented" #TODO Turn into an exception
    def computePhi(self):
        assert False, "Not implemented" #TODO Turn into an exception
        
class EQ(Kernel):
    def __init__(self,l2,sigma2):
        """
        A Exponentiated Quadratic kernel
        Arguments:
            l2 == lengthscale
            sigma2 == variance of kernel
        """
        self.l2 = l2
        self.sigma2 = sigma2
        self.W = None #need to be set by calling generateFeatures.
        self.b = None 
                
    def generateFeatures(self,N_D,N_feat):
        """
        Create a random basis for the kernel.
        Arguments:
            N_D = number of dimensions
            N_feat = number of features
        """
        self.W = np.random.normal(0,1.0,size=(N_feat,N_D))
        self.b = np.random.uniform(0.,2*np.pi,size=N_feat)
        self.N_D = N_D
        self.N_feat = N_feat
        
 
    def getPhi(self,coords):
        assert self.W is not None, "Need to call generateFeatures before computing phi."
        norm = 1./np.sqrt(self.N_feat)
        #c=np.sqrt(2.0)/(self.l2)
        c=1/(self.l2)
        for w,b in zip(self.W,self.b):
            phi=norm*np.sqrt(2*self.sigma2)*np.cos(c*np.einsum('i,ijkl->jkl',w,coords)+ b)
            yield phi
