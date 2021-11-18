import numpy as np

class Kernel():
    def __init__(self):
        assert False, "Not implemented" #TODO Turn into an exception
    def generate_features(self,N_D,N_feat):
        assert False, "Not implemented" #TODO Turn into an exception
    def compute_phi(self):
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
        self.W = None #need to be set by calling generate_features.
        self.b = None 
                
    def generate_features(self,N_D,N_feat):
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
        
    def compute_phi(self,coords):
        """
        Compute phi (posterior matrix of basis vectors)
        Arguments:
            coords = an N by N_D array of locations
        """
        assert self.W is not None, "Need to call generate_features before computing phi."
        
        norm = 1./np.sqrt(self.N_feat)
        c=np.sqrt(2.0)/(self.l2)
        phi_matrix = norm*np.sqrt(2*self.sigma2)*np.cos(c*(self.W@coords)+ self.b[:,None])
        phi_matrix = phi_matrix.reshape(N_feat,Nt,Nx,Ny)
        return phi_matrix
        
class Model():
    def __init__(self,X,y,boundary,resolution,kernel,noise_std,N_feat=25,spatial_averaging=1.0):
        """
        The Advection Diffusion Model.
        
        At the moment we assume a 3d grid [time, x, y].
        
        Parameters:
            X = an N by 4 matrix of sensor times and locations [time_start, time_end, x, y]
            y = an N long vector of the measurements associated with the sensors.
            boundary = a two element tuple of the corners of the grid. e.g. ([0,0,0],[10,10,10])        
            resolution = a list of the grid size in each dimension. e.g. [10,20,20]
            kernel = the kernel to use
            noise_std = the noise standard deviation
            N_feat = number of fourier features
            spatial_averaging = how big the volume the sensor measures (default 0.001).
        """
        #TODO URGENT: The spatial averaging doesn't make sense!
        
        self.N_D = len(resolution)
        assert self.N_D==3, "Currently advectionGP only supports a 3d grid: T,X,Y. Check your resolution parameter."
        assert self.N_D==len(boundary[0]), "The dimensions in the boundary don't match those in the grid resolution"
        assert self.N_D==len(boundary[1]), "The dimensions in the boundary don't match those in the grid resolution"

        self.spatial_averaging = spatial_averaging
        self.X = X
        self.y = y
        self.boundary = boundary
        self.resolution = resolution
        self.noise_std = noise_std
      

        assert self.X.shape[1]==4, "The X input matrix should be Nx4."
        assert self.y.shape[0]==self.X.shape[0], "The length of X should equal the length of y"
        self.kernel = kernel
        self.kernel.generate_features(self.N_D,N_feat) 
        self.N_feat = N_feat
        
    def compute_H(self,x):
        """Return a self.resolution array describing how the concentration is added up for an observation in x.
        Uses self.spatial_averaging to extend the part of the domain that is being observed.
        
        Parameters:
            x == a 4 element vector, time_start, time_end, x, y
            
        Returns:
            H = a self.resolution sized numpy array
        """
        
    def compute_TODONEEDNAME(): #TODO
        """compute the backward advection-diffusion PDE with finite difference method."""
        pass
       
    def compute_v(self):
        """
        Computes v.
        Returns
            v = an N by ? ... 
        """
        for x in X:
            H = self.compute_H(x)
            #TODO: URGENT: REALLY NEED CLEARER METHOD NAMES AND VARIABLE NAMES...
            y_Conv_Diff_back=-Convection_Diffusion_back(H)
            #TODO ...
            y_bvv.append(y_Conv_Diff_back*(xmax/Nx)*(ymax/Ny)*(tmax/Nt))

        return y_bvv#TODO...
     
