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
        """
        Returns an interator providing indicator matrices for each observation.
        Should integrate to one over the `actual' space (but not necessarily over the grid).
        Arguments:
            model == is a model object (provides grid resolution etc)
            
        """
        halfGridTile = np.array([0,self.spatialAveraging/2,self.spatialAveraging/2])
        print(self.obsLocs[:,[0,2,3]]-halfGridTile)
        startOfHs = model.getGridCoord(self.obsLocs[:,[0,2,3]]-halfGridTile)
        endOfHs = model.getGridCoord(self.obsLocs[:,[1,2,3]]+halfGridTile)
        
        endOfHs[endOfHs==startOfHs]+=1 #TODO Improve this to ensure we enclose the sensor volume better with our grid.
        
        assert (np.all(startOfHs>=0)) & (np.all(startOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert (np.all(endOfHs>=0)) & (np.all(endOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert np.all(endOfHs>startOfHs), "Observation cell has zero volume: at least one axis has no length. startOfHs:"+str(startOfHs)+" endOfHs:"+str(endOfHs)
                
        for start,end,tlength in zip(startOfHs,endOfHs,self.obsLocs[:,1]-self.obsLocs[:,0]):
            h = np.zeros(model.resolution)
            h[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = 1/(self.spatialAveraging**2 * tlength)
            #h /= np.sum(h)
            print(start[0],end[0],start[1],end[1],start[2],end[2])
            yield h
        
#X = an N by 4 matrix of sensor times and locations [time_start, time_end, x, y]
#y = an N long vector of the measurements associated with the sensors.
         
class Model():
    def __init__(self,boundary,resolution,kernel,noiseSD,sensormodel,N_feat=25,spatial_averaging=1.0,u=0.001,k_0=0.001):
        """
        The Advection Diffusion Model.
        
        At the moment we assume a 3d grid [time, x, y].
        
        Parameters:
            boundary = a two element tuple of the corners of the grid. e.g. ([0,0,0],[10,10,10])        
            resolution = a list of the grid size in each dimension. e.g. [10,20,20]
            kernel = the kernel to use
            noiseSD = the noise standard deviation
            sensormodel = an instatiation of a SensorModel class that implements the getHs method.
            N_feat = number of fourier features
            spatial_averaging = how big the volume the sensor measures (default 0.001).
            u = wind speed
            k_0 = diffusion constant
        """
        #TODO URGENT: The spatial averaging doesn't make sense!
        #TODO The wind speed and diffusion might need to be vectors
        
        self.N_D = len(resolution)
        assert self.N_D==3, "Currently advectionGP only supports a 3d grid: T,X,Y. Check your resolution parameter."
        assert self.N_D==len(boundary[0]), "The dimensions in the boundary don't match those in the grid resolution"
        assert self.N_D==len(boundary[1]), "The dimensions in the boundary don't match those in the grid resolution"

        #self.spatial_averaging = spatial_averaging
        #self.X = X
        #self.y = y
        self.boundary = [np.array(boundary[0]),np.array(boundary[1])]
        self.resolution = np.array(resolution)
        self.noiseSD = noiseSD
        self.sensormodel = sensormodel
        
        
        #coords is a D x (Nt,Nx,Ny) array of locations of the grid vertices.
        #TODO Maybe write more neatly...
        tt=np.linspace(0.0,self.boundary[1][0],self.resolution[0])
        xx=np.linspace(0.0,self.boundary[1][1],self.resolution[1])
        yy=np.linspace(0.0,self.boundary[1][2],self.resolution[2])
        self.coords=np.asarray(np.meshgrid(tt,xx,yy,indexing='ij'))
        #self.coords=coords.reshape(self.N_D,self.resolution[0]*self.resolution[1]*self.resolution[2])
      
        #Compute some variables useful for PDEs
        
        self.u = u
        self.k_0 = k_0

        

        #assert self.X.shape[1]==4, "The X input matrix should be Nx4."
        #assert self.y.shape[0]==self.X.shape[0], "The length of X should equal the length of y"
        self.kernel = kernel
        self.kernel.generateFeatures(self.N_D,N_feat) 
        self.N_feat = N_feat
        
    def getGridStepSize(self):
        dt=(self.boundary[1][0]-self.boundary[0][0])/self.resolution[0]
        dx=(self.boundary[1][1]-self.boundary[0][1])/self.resolution[1]
        dy=(self.boundary[1][2]-self.boundary[0][2])/self.resolution[2]
        dx2=dx*dx
        dy2=dy*dy
        Nt=self.resolution[0]
        Nx=self.resolution[1]
        Ny=self.resolution[2]
        return dt,dx,dy,dx2,dy2,Nt,Nx,Ny
        
        
    def getGridCoord(self,realPos):
        """
        Gets the location on the mesh for a real position
        """
        return np.floor(self.resolution*(realPos - self.boundary[0])/(self.boundary[1]-self.boundary[0])).astype(int)
        
    def computeConcentration(self,source):        
        """
        Computes concentrations.
        Arguments:
         source == forcing function (shape: Nt x Nx x Ny). Can either be generated by ... or determine manually.
       
        returns array of concentrations (shape: Nt x Nx x Ny), given source. (also saved it in self.concentration)
        """
        
        #get the grid step sizes, their squares and the size of the grid
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        
        c=np.zeros(((self.resolution)))
        
        c[0,:,:]=0

        k_0 = self.k_0
        u = self.u
        for i in range(0,Nt-1):
            # Corner BCs 
            c[i+1,0,0]=c[i,0,0]+dt*( source[i,0,0] ) +dt*k_0*( 2*c[i,1,0]-2*c[i,0,0])/dx2 + dt*k_0*( 2*c[i,0,1]-2*c[i,0,0])/dy2
            c[i+1,Nx-1,Ny-1]=c[i,Nx-1,Ny-1]+dt*( source[i,Nx-1,Ny-1])+dt*k_0*( 2*c[i,Nx-2,Ny-1]-2*c[i,Nx-1,Ny-1])/dx2 + dt*k_0*( 2*c[i,Nx-1,Ny-2]-2*c[i,Nx-1,Ny-1])/dy2
            c[i+1,0,Ny-1]=c[i,0,Ny-1]+dt*( source[i,0,Ny-1] ) +dt*k_0*( 2*c[i,1,Ny-1]-2*c[i,0,Ny-1])/dx2 + dt*k_0*( 2*c[i,0,Ny-2]-2*c[i,0,Ny-1])/dy2
            c[i+1,Nx-1,0]=c[i,Nx-1,0]+dt*( source[i,Nx-1,0])+dt*k_0*( 2*c[i,Nx-2,0]-2*c[i,Nx-1,0])/dx2 + dt*k_0*( 2*c[i,Nx-1,1]-2*c[i,Nx-1,0])/dy2
    
            c[i+1,1:Nx-1,0]=c[i,1:Nx-1,0]+dt*(source[i,1:Nx-1,0]-u*(c[i,2:Nx,0]-c[i,0:Nx-2,0])/(2*dx)+k_0*(2*c[i,1:Nx-1,1]-2*c[i,1:Nx-1,0])/dy2 +k_0*(c[i,2:Nx,0]-2*c[i,1:Nx-1,0]+c[i,0:Nx-2,0] )/dx2     )
            c[i+1,1:Nx-1,Ny-1]=c[i,1:Nx-1,Ny-1]+dt*( source[i,1:Nx-1,Ny-1]-u*(c[i,2:Nx,Ny-1]-c[i,0:Nx-2,Ny-1])/(2*dx)+k_0*(2*c[i,1:Nx-1,Ny-2]-2*c[i,1:Nx-1,Ny-1])/dy2 +k_0*(c[i,2:Nx,Ny-1]-2*c[i,1:Nx-1,Ny-1]+c[i,0:Nx-2,Ny-1] )/dx2     )  
            #for k in range(1,Ny-1):
                # x edge bcs
            c[i+1,Nx-1,1:Ny-1]=c[i,Nx-1,1:Ny-1]+dt*( source[i,Nx-1,1:Ny-1]-u*(c[i,Nx-1,2:Ny]-c[i,Nx-1,0:Ny-2])/(2*dy)+k_0*(2*c[i,Nx-2,1:Ny-1]-2*c[i,Nx-1,1:Ny-1])/dx2 +k_0*(c[i,Nx-1,2:Ny]-2*c[i,Nx-1,1:Ny-1]+c[i,Nx-1,0:Ny-2] )/dy2     )
            c[i+1,0,1:Ny-1]=c[i,0,1:Ny-1]+dt*( source[i,0,1:Ny-1]-u*(c[i,0,2:Ny]-c[i,0,0:Ny-2])/(2*dy)+k_0*(2*c[i,1,1:Ny-1]-2*c[i,0,1:Ny-1])/dx2 +k_0*(c[i,0,2:Ny]-2*c[i,0,1:Ny-1]+c[i,0,0:Ny-2] )/dy2     )     
                # Internal Calc
            c[i+1,1:Nx-1,1:Ny-1]=c[i,1:Nx-1,1:Ny-1] +dt*(source[i,1:Nx-1,1:Ny-1]-u*(c[i,2:Nx,1:Ny-1]-c[i,0:Nx-2,1:Ny-1])/(2*dx) -u*(c[i,1:Nx-1,2:Ny]-c[i,1:Nx-1,0:Ny-2] )/(2*dy) +k_0*(c[i,2:Nx,1:Ny-1]-2*c[i,1:Nx-1,1:Ny-1]  +c[i,0:Nx-2,1:Ny-1])/dx2+k_0*(c[i,1:Nx-1,2:Ny]-2*c[i,1:Nx-1,1:Ny-1]  +c[i,1:Nx-1,0:Ny-2])/dy2 )
        concentration = c 
        return c
      
      
      
    def computeObservations(self, sensorModel):
        """       
        Using the forward model
        ##todo        
        """
        #use self.concentration
        
        obs = np.zeros(len(sensorModel.obsLocs))
        for it,h in enumerate(sensorModel.getHs(self)):
            obs[it]=sum(sum(sum(h*self.conc)))*self.dt*self.dx*self.dy+np.random.normal(0.0,self.noiseSD,1)
            
        self.ySimulated = obs
        
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
