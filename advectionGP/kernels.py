import numpy as np

class Kernel():
    def __init__(self):
        assert False, "Not implemented" #TODO Turn into an exception
    def generateFeatures(self,N_D,N_feat):
        assert False, "Not implemented" #TODO Turn into an exception
    def computePhi(self):
        assert False, "Not implemented" #TODO Turn into an exception
    def getPhiValues(self,particles):
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
                
    def generateFeatures(self,N_D,N_feat,boundary):
        """
        Create a random basis for the kernel sampled from the normal distribution.
        Here W is a list of weights for the t,x and y dimentions and b is a linear addition.
        Arguments:
            N_D = number of dimensions
            N_feat = number of features
        """
        if np.isscalar(self.l2):
            self.l2 = np.repeat(self.l2,N_D)
        self.W = np.random.normal(0,1.0,size=(N_feat,N_D))
        self.b = np.random.uniform(0.,2*np.pi,size=N_feat)
        self.N_D = N_D
        self.N_feat = N_feat
        
 
    def getPhi(self,coords):
        """
        Generates a (N_feat,Nt,Nx,Ny) matrix of basis vectors using features from generateFeatures 
        Arguments:
            coords: map of all (t,x,y) points in the grid
        """
        assert self.W is not None, "Need to call generateFeatures before computing phi."
        norm = 1./np.sqrt(self.N_feat)
        
        #We assume that we are using the e^-(1/2 * x^2/l^2) definition of the EQ kernel,
        #(in Mauricio's definition he doesn't use the 1/2 factor - but that's less standard).
        #c=np.sqrt(2.0)/(self.l2)
        ####c=1/(self.l2)
        for w,b in zip(self.W,self.b):
            phi=norm*np.sqrt(2*self.sigma2)*np.cos(np.einsum('i,i...->...',w/self.l2,coords)+ b)
            yield phi                       

    def getPhiValues(self,particles):
        """
        Evaluates all features at location of all particles.
        
        
        Nearly a duplicate of getPhi, this returns phi for the locations in particles. 
        
        Importantly, particles is of shape N_ObsxN_Particlesx3,
        (typically N_Obs is the number of observations, N_ParticlesPerObs is the number of particles/observation. 3 is the dimensionality of the space).
        
        Returns array (Nfeats, N_ParticlesPerObs, N_Obs)
        
        """
        c=1/(self.l2)
        norm = 1./np.sqrt(self.N_feat)
        return norm*np.sqrt(2*self.sigma2)*np.cos(np.einsum('ij,lkj',self.W/self.l2,particles)+self.b[:,None,None])
  

    def getPhiDerivative(self,coords):
        """
        Generates a (N_feat,Nt,Nx,Ny) matrix of basis vectors using features from generateFeatures 
        Arguments:
            coords: map of all (t,x,y) points in the grid
        """
        assert self.W is not None, "Need to call generateFeatures before computing phi."
        norm = 1./np.sqrt(self.N_feat)

        #We assume that we are using the e^-(1/2 * x^2/l^2) definition of the EQ kernel,
        #(in Mauricio's definition he doesn't use the 1/2 factor - but that's less standard).
        #c=np.sqrt(2.0)/(self.l2)
        c=1/(self.l2)
        for w,b in zip(self.W,self.b):
            phi=(c**2)*np.einsum('i,i...->...',w,coords)*norm*np.sqrt(2*self.sigma2)*np.sin(np.einsum('i,i...->...',w/self.l2,coords)+ b)
            yield phi
            

            
class GaussianBases(Kernel):
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
        self.mu= None
                
    def generateFeatures(self,N_D,N_feat,boundary):
        """
        Create a random basis for the kernel sampled from the normal distribution.
        Here W is a list of weights for the t,x and y dimentions and b is a linear addition.
        Arguments:
            N_D = number of dimensions
            N_feat = number of features
        """
        if np.isscalar(self.l2):
            self.l2 = np.repeat(self.l2,N_D)
        self.mu = np.random.uniform(low=boundary[0],high=boundary[1],size=[N_feat,len(boundary[0])])

        self.N_D = N_D
        self.N_feat = N_feat
 
    def getPhi(self,coords):
        """
        Generates a series (of N_feat) matrices, shape (Nt,Nx,Ny) of compact basis vectors using features from generateFeatures 
        Arguments:
            coords: an array of D x [Nt, Nx, Ny, Nz...] coords of points
        Notes:
            uses self.mu, N_feat x D
        """
        for centre in self.mu:
            sqrdists = np.sum(((np.transpose(coords,list(range(1,coords.ndim))+[0])-centre)**2)/(self.l2**2),coords.ndim-1)
            phi = (1/np.sqrt(2*self.sigma2*np.pi))*np.exp(-sqrdists/2)
            yield phi
            
    def getPhiValues(self,particles):
        """
        Evaluates all features at location of all particles.
        
        
        Nearly a duplicate of getPhi, this returns phi for the locations in particles. 
        
        Importantly, particles is of shape N_ObsxN_Particlesx3,
        (typically N_Obs is the number of observations, N_ParticlesPerObs is the number of particles/observation. 3 is the dimensionality of the space).
        
        Returns array (Nfeats, N_ParticlesPerObs, N_Obs)
        
        """
        mu=self.mu
        coordList=particles
        phi=np.zeros([mu.shape[0],particles.shape[1],particles.shape[0]])
        for i,mus in enumerate(self.mu):
            phi[i,:,:]=((1/np.sqrt(2*self.sigma2*np.pi))*np.exp(-(1/(2*self.l2[0]**2))*((mus[0]-np.array(coordList[:,:,0]))**2))*(1/np.sqrt(2*self.sigma2*np.pi))*np.exp(-(1/(2*self.l2[1]**2))*((mus[1]-np.array(coordList[:,:,1]))**2))*(1/np.sqrt(2*self.sigma2*np.pi))*np.exp(-(1/(2*self.l2[2]**2))*((mus[2]-np.array(coordList[:,:,2]))**2))).T
        return phi


