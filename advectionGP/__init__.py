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
        c=np.sqrt(2.0)/(self.l2)
        for w,b in zip(self.W,self.b):
            phi=norm*np.sqrt(2*self.sigma2)*np.cos(c*np.einsum('i,ijkl->jkl',w,coords)+ b)
            yield phi
       
class SensorModel():
    def __init__(self):
        """Builds H"""
        assert False, "Not implemented" #TODO Turn into an exception

        
    def getHs(self):
        assert False, "Not implemented" #TODO Turn into an exception
    
class FixedSensorModel(SensorModel):
    def __init__(self,obsLocations,spatialAveraging):
        """Return a self.resolution array describing how the concentration is added up for an observation in x.
        Uses self.spatial_averaging to extend the part of the domain that is being observed.
        
        Parameters:
            x == a 4 element vector, time_start, time_end, x, y
            
        The getHs method returns a model.resolution sized numpy array
        """
        self.obsLocs = obsLocations
        self.spatialAveraging = spatialAveraging
        #TO DO
       
    def getHs(self,model):
        halfGridTile = np.array([0,self.spatialAveraging/2,self.spatialAveraging/2])
        startOfHs = model.getGridCoord(self.obsLocs[:,[0,2,3]]-halfGridTile)
        endOfHs = model.getGridCoord(self.obsLocs[:,[1,2,3]]+halfGridTile)
        for start,end,tlength in zip(startOfHs,endOfHs,self.obsLocs[:,1]-self.obsLocs[:,0]):
            h = np.zeros(model.resolution)
            h[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = self.spatialAveraging**2 * tlength
            print(start[0],end[0],start[1],end[1],start[2],end[2])
            yield h
        
#X = an N by 4 matrix of sensor times and locations [time_start, time_end, x, y]
#y = an N long vector of the measurements associated with the sensors.
         
class Model():
    def __init__(self,boundary,resolution,kernel,noise_std,sensormodel,N_feat=25,spatial_averaging=1.0):
        """
        The Advection Diffusion Model.
        
        At the moment we assume a 3d grid [time, x, y].
        
        Parameters:
            boundary = a two element tuple of the corners of the grid. e.g. ([0,0,0],[10,10,10])        
            resolution = a list of the grid size in each dimension. e.g. [10,20,20]
            kernel = the kernel to use
            noise_std = the noise standard deviation
            sensormodel = an instatiation of a SensorModel class that implements the getHs method.
            N_feat = number of fourier features
            spatial_averaging = how big the volume the sensor measures (default 0.001).
        """
        #TODO URGENT: The spatial averaging doesn't make sense!
        
        self.N_D = len(resolution)
        assert self.N_D==3, "Currently advectionGP only supports a 3d grid: T,X,Y. Check your resolution parameter."
        assert self.N_D==len(boundary[0]), "The dimensions in the boundary don't match those in the grid resolution"
        assert self.N_D==len(boundary[1]), "The dimensions in the boundary don't match those in the grid resolution"

        #self.spatial_averaging = spatial_averaging
        #self.X = X
        #self.y = y
        self.boundary = [np.array(boundary[0]),np.array(boundary[1])]
        self.resolution = np.array(resolution)
        self.noise_std = noise_std
        self.sensormodel = sensormodel
        
        
        #coords is a D x (Nt,Nx,Ny) array of locations of the grid vertices.
        #TODO Maybe write more neatly...
        tt=np.linspace(0.0,self.boundary[1][0],self.resolution[0])
        xx=np.linspace(0.0,self.boundary[1][1],self.resolution[1])
        yy=np.linspace(0.0,self.boundary[1][2],self.resolution[2])
        self.coords=np.asarray(np.meshgrid(tt,xx,yy,indexing='ij'))
        #self.coords=coords.reshape(self.N_D,self.resolution[0]*self.resolution[1]*self.resolution[2])
      

        #assert self.X.shape[1]==4, "The X input matrix should be Nx4."
        #assert self.y.shape[0]==self.X.shape[0], "The length of X should equal the length of y"
        self.kernel = kernel
        self.kernel.generateFeatures(self.N_D,N_feat) 
        self.N_feat = N_feat
        
    def getGridCoord(self,realPos):
        """
        Gets the location on the mesh for a real position
        """
        return (self.resolution*(realPos - self.boundary[0])/(self.boundary[1]-self.boundary[0])).astype(int)
        
    def computeConcentration(self,source):
        #TODO
        #concentration
        self.concentration = None #TODO
      
    def computeObservations(self, sensorModel):
        """       
        Using the forward model
        ##todo        
        """
        #use self.concentration
        for h in self.sensorModel.getHs():
            pass
            #run forward model and get H
        self.ySimulated = None
        
    def computeSourceFromPhi(self):
        """
        """
        #uses self.phi and self.z to compute self.sources
        self.sources = None
        
    def computePhi(self):
        """
        """
        #use self.W and self.b to find self.phi
        self.phi = None
        
    def computeAdjoint(self,sensorModel):
        """
        """
        #for each feature computes the adjoint array. v
        self.v = None
        
    def computeModelRegressors(self):
        """
        """
        #phi * v, --> scale
        self.X = None
    
    def computeZDistribution(self,y):
        """
        """
        #uses self.X and observations y.
        self.zMean = None
        self.zCov = None
