import numpy as np
import threading

class Kernel():
    def __init__(self):
        assert False, "Not implemented" #TODO Turn into an exception
    def generateFeatures(self,N_D,N_feat):
        assert False, "Not implemented" #TODO Turn into an exception
    def computePhi(self):
        assert False, "Not implemented" #TODO Turn into an exception
    def getPhiValues(self,particles):
        assert False, "Not implemented" #TODO Turn into an exception





    
##@jit(nopython=True,parallel=True)
#def getEQPhiValues(coords,N_feat,N_ParticlesPerObs, N_Obs,sigma2,l2, W, b):
#    def computeEQitem(idx):
#        Phi[idx,:,:] = norm*np.sqrt(2*sigma2)*np.cos(np.dot(coords.T,(W[idx]/l2)).T + b[idx])
#    
#    Phi = np.zeros((N_feat, N_ParticlesPerObs, N_Obs))
#    norm = 1./np.sqrt(N_feat)
#    
#    with multiprocess.Pool() as pool:
#        for res in pool.imap_unordered(computeEQitem,range(N_feat)):
#            pass
#            
#    #for fi in range(N_feat): #,(wit,bit) in enumerate(zip(W,b)):        
#    #    #Phi[fi,:,:]=norm*np.sqrt(2*sigma2)*np.cos(np.einsum('i,i...->...',wit/l2,coords)+ bit)
#    #    Phi[fi,:,:]=computeEQitem(fi)#norm,sigma2,coords,wit,l2,bit)
#    return Phi
           
class EQ(Kernel):
    def __init__(self,l2,sigma2):
        """
        A Exponentiated Quadratic kernel
        Arguments:
            l2 == lengthscale (or lengthscales in a list of the length of the number of dimensions).
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
            boundary = a list of two lists describing the lower and upper corners of the domain [not used by this class]
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


    def getPhiPopResultsBlock(self,i):
        """
        Used by getPhiValues (helps with threading)
        """
        norm = 1./np.sqrt(self.N_feat)
        w = self.W[i:i+self.threadblocksize]
        b = self.b[i:i+self.threadblocksize]
        self.result[i:i+self.threadblocksize] = norm*np.sqrt(2*self.sigma2)*np.cos(((self.tempcoords.T@(w/self.l2).T) + b).T)


        
    def getPhiValues(self,particles):
        """
        Evaluates all features at location of all particles.
                
        Nearly a duplicate of getPhi, this returns phi for the locations in particles. 
        
        Importantly, particles is of shape N_ObsxN_Particlesx3,
        (typically N_Obs is the number of observations, N_ParticlesPerObs is the number of particles/observation. 3 is the dimensionality of the space).
        
        Returns array (Nfeats, N_ParticlesPerObs, N_Obs)
        
        """
        self.result = np.zeros([self.N_feat, particles.shape[1], particles.shape[0]])
        threads = []
        self.threadblocksize = int(self.N_feat / 16)+1 #e.g. if there are 1999 features --> self.threadblocksize = 125 (but with one with 124). if there are 2001 -> 126 (one will have 125)
        self.tempcoords = particles.T#transpose([2,1,0])
        for i in range(0,self.N_feat,self.threadblocksize):
            threads.append(threading.Thread(target=self.getPhiPopResultsBlock, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        result = self.result
        self.result = None #trying to avoid keeping this in memory
        return result


    def oldGetPhiValues(self,particles):
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
            

def meshgridndim(boundary,Nsteps,moveEdgeIn=False):
    """Returns points in a uniform grid within the boundary
    
    Parameters:
        boundary = a list of two lists describing the lower and upper corners of the domain.
            each list will be of Ndims long.
        Nsteps = number of steps in each dimension.
        moveEdgeIn = whether to slightly move the top edge in to ensure points lie inside boundary. [default = False]
    Returns:
        Returns a matrix of shape: (Nsteps^Ndims, Ndims)
    """    
    Ndims = len(boundary[0])
    if moveEdgeIn:
        newboundary = []
        newboundary.append(boundary[0])
        newboundary.append(boundary[1]-1e-5)
        boundary = newboundary
    g = np.array(np.meshgrid(*[np.linspace(a,b,Nsteps) for a,b in zip(boundary[0],boundary[1])]))
    return g.reshape(Ndims,Nsteps**Ndims).T

from advectionGP.kernels import Kernel
class GaussianBases(Kernel):
    def __init__(self,l2,sigma2,random=False):
        """
        A Exponentiated Quadratic kernel
        Arguments:
            l2 == lengthscale (or lengthscales in a list of the length of the number of dimensions).
            sigma2 == variance of kernel
            random = whether to sample the points randomly or in a uniform grid (default False)
        """
        self.l2 = l2
        self.sigma2 = sigma2
        self.W = None #need to be set by calling generateFeatures.
        self.b = None 
        self.mu= None
        self.random = random
                
    def generateFeatures(self,N_D,N_feat,boundary):
        """
        Create a basis for the kernel, distributed in a grid/random over part of domain defined by 'boundary'.
        
        Arguments:
            N_D = number of dimensions
            N_feat = number of features          
            boundary = a list of two lists describing the lower and upper corners of the domain.
        """    
        assert len(boundary[0])==N_D
        if np.isscalar(self.l2):
            self.l2 = np.repeat(self.l2,N_D)                    
        if self.random:
            self.mu = np.random.uniform(boundary[0],boundary[1],size=(N_feat,N_D))
        else:
            Nsteps = int(np.round(N_feat**(1/N_D))) #e.g. 100 features asked for, 2 dimensions -> 100^(1/2) = 10 steps in each dim.        
            self.mu = meshgridndim(boundary,Nsteps)
            #updated number of features...
            N_feat = len(self.mu)


        self.N_D = N_D
        self.N_feat = N_feat
        self.boundary = boundary
 
    def getPhi(self,coords):
            """
            Generates a series (of N_feat) matrices, shape (Nt,Nx,Ny) of compact basis vectors using features from generateFeatures 
            Arguments:
                coords: an array of D x [Nt, Nx, Ny, Nz...] coords of points
            Notes:
                uses self.mu, N_feat x D
            """

            norm_const = np.prod((0.5*self.l2**2 * np.pi)**(-0.25))
            
            #correct for density of features [subtract one from number in each dim, as the linspacing places points from one boundary to the other]
            norm_const *= np.sqrt((np.prod((np.array(self.boundary[1])-np.array(self.boundary[0]))/((self.N_feat**(1/self.N_D)-1)))))
            norm_const *= np.sqrt(self.sigma2)
            for centre in self.mu:
                sqrdists = np.sum(((coords.T - centre)/self.l2)**2,-1).T
                phi = norm_const * np.exp(-sqrdists)
                yield phi
            
    def getPhiValues(self,particles):
        """
        Evaluates all features at location of all particles.
        
        
        Nearly a duplicate of getPhi, this returns phi for the locations in particles. 
        
        Importantly, particles is of shape N_ObsxN_Particlesx3,
        (typically N_Obs is the number of observations, N_ParticlesPerObs is the number of particles/observation. 3 is the dimensionality of the space).
        
        Returns array (Nfeats, N_ParticlesPerObs, N_Obs)
        
        """
        #return np.array([p for p in self.getPhi(particles.transpose([2,1,0]))])
        return np.array([p for p in self.getPhi(particles.T)])
        #mu=self.mu
        #coordList=particles
        #phi=np.zeros([mu.shape[0],particles.shape[1],particles.shape[0]])
        #for i,mus in enumerate(self.mu):
        #    phi[i,:,:]=((1/np.sqrt(2*self.sigma2*np.pi))*np.exp(-(1/(2*self.l2[0]**2))*((mus[0]-np.array(coordList[:,:,0]))**2))*(1/np.sqrt(2*self.sigma2*np.pi))*np.exp(-(1/(2*self.l2[1]**2))*((mus[1]-np.array(coordList[:,:,1]))**2))*(1/np.sqrt(2*self.sigma2*np.pi))*np.exp(-(1/(2*self.l2[2]**2))*((mus[2]-np.array(coordList[:,:,2]))**2))).T
        #return phi
