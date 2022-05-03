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
        Create a random basis for the kernel sampled from the normal distribution.
        Here W is a list of weights for the t,x and y dimentions and b is a linear addition.
        Arguments:
            N_D = number of dimensions
            N_feat = number of features
        """
        self.W = np.random.normal(0,1.0,size=(N_feat,N_D))
        self.b = np.random.uniform(0.,2*np.pi,size=N_feat)
        self.N_D = N_D
        self.N_feat = N_feat
        
 
    def getPhi(self,coords):
        """
        Generates a (N_feat,Nt,Nx,Ny) matrix of basis vectors using features from generateFeatures 
        Arguments:
            coords: map of all (t,x,y) points in the grid
            
        CURRENTLY NOT USED
        """
        assert self.W is not None, "Need to call generateFeatures before computing phi."
        norm = 1./np.sqrt(self.N_feat)
        
        #We assume that we are using the e^-(1/2 * x^2/l^2) definition of the EQ kernel,
        #(in Mauricio's definition he doesn't use the 1/2 factor - but that's less standard).
        #c=np.sqrt(2.0)/(self.l2)
        c=1/(self.l2)
        for w,b in zip(self.W,self.b):
            phi=norm*np.sqrt(2*self.sigma2)*np.cos(c*np.einsum('i,ijkl->jkl',w,coords)+ b)
            yield phi
            
            
    def getPhi1D(self,coords):
        """
        Generates a (N_feat,Nt) matrix of basis vectors using features from generateFeatures 
        Arguments:
            coords: map of all (t) points in the grid

        CURRENTLY NOT USED
        """
        assert self.W is not None, "Need to call generateFeatures before computing phi."
        norm = 1./np.sqrt(self.N_feat)

        #We assume that we are using the e^-(1/2 * x^2/l^2) definition of the EQ kernel,
        #(in Mauricio's definition he doesn't use the 1/2 factor - but that's less standard).
        #c=np.sqrt(2.0)/(self.l2)
        c=1/(self.l2)
        for w,b in zip(self.W,self.b):
            phi=norm*np.sqrt(2*self.sigma2)*np.cos((c*w*np.array(coords))+b)
            yield phi
            
    #earlier experiment thinking that einsum would be slow. To delete.
    #def getPhiFast(self,coords):
    #    """
    #    Yields N_feat (Nt,Nx,Ny) phi matrices using features from generateFeatures 
    #    Arguments:
    #        coords: map of all (t,x,y) points in the grid
    #    """
    #    assert self.W is not None, "Need to call generateFeatures before computing phi."
    #    norm = 1./np.sqrt(self.N_feat)
    #    #c=1/(self.l2)
    #    c=np.sqrt(2.0)/(self.l2)
    #    for i in range(len(self.W)):
    #        phi = norm*np.sqrt(2*self.sigma2)*tf.math.cos(np.transpose(c*m.coords,axes=[1,2,3,0])@k.W[i,:]+ self.b[i])
    #        yield phi
