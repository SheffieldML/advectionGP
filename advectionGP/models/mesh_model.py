import numpy as np

class MeshModel():
    def __init__(self,boundary,resolution,kernel,noiseSD,sensormodel,N_feat=25):
        """
        The base class for grid based PDE and ODE models in 1, 2 or 3 dimensions.
        
        Parameters:
            boundary = a two element tuple of the corners of the grid. e.g. ([0,0,0],[10,10,10])        
            resolution = a list of the grid size in each dimension. e.g. [10,20,20]
            kernel = the kernel to use
            noiseSD = the noise standard deviation
            sensormodel = an instatiation of a SensorModel class that implements the getHs method.
            N_feat = number of fourier features
            windmodel = an instance of a Wind class (to build u using)
            k_0 = diffusion constant
            
        When using real data (e.g. wind etc) we assume the units are:
         - km
         - hours
         - km/h
         - pollution can be anything, we assume at the moment PM2.5 in ug/m^3.
        
        """
        #TODO URGENT: The spatial averaging doesn't make sense!
        #TODO The wind speed and diffusion might need to be vectors
        
        self.N_D = len(resolution)
        assert self.N_D==len(boundary[0]), "The dimensions in the boundary don't match those in the grid resolution"
        assert self.N_D==len(boundary[1]), "The dimensions in the boundary don't match those in the grid resolution"


        self.boundary = [np.array(boundary[0]),np.array(boundary[1])]
        self.resolution = np.array(resolution)
        self.noiseSD = noiseSD
        self.sensormodel = sensormodel

        
        
        #coords is a D x (Nt,Nx,Ny) array of locations of the grid vertices.
        #TODO Maybe write more neatly...
        
        gg = []
        for d in range(self.N_D):
            gg.append(np.linspace(self.boundary[0][d],self.boundary[1][d],self.resolution[d]))
        
        if self.N_D==1: self.coords=np.asarray(np.meshgrid(gg[0],indexing='ij'))
        if self.N_D==2: self.coords=np.asarray(np.meshgrid(gg[0],gg[1],indexing='ij'))
        if self.N_D==3: self.coords=np.asarray(np.meshgrid(gg[0],gg[1],gg[2],indexing='ij'))
      
        #Compute some variables useful for PDEs
        
        self.kernel = kernel
        self.kernel.generateFeatures(self.N_D,N_feat,boundary) 
        self.N_feat = N_feat
        
        self.sourcecache = {}

    def getGridStepSize(self):
        """
        Calculates useful scalars for the PDE model
        outputs:
            dt: time grid size
            dx: x direction grid size
            dy: y direction grid size
            dx2 = dx**2
            dy2 = dy**2
            Nt: Number of evaluation points in time
            Nx: Number of evaluation points in x axis
            Ny: Number of evaluation points in y axis
        """
        
        delta=(self.boundary[1]-self.boundary[0])/self.resolution
        Ns=self.resolution
        return delta,Ns
        
    def getGridCoord(self,realPos):
        """
        Gets the location on the mesh for a real position
        I.e. Given a valume in m getGridCoord returns the location on the grid
        
        todo: assertion for out of bounds value
        """
        return np.floor(self.resolution*(realPos - self.boundary[0])/(self.boundary[1]-self.boundary[0])).astype(int)
    
                
                
    def computeObservations(self,addNoise=False):
        """       
        Generates test observations by calculating the inner product of the filter function from the senor model and a given self.conc.
        Arguments:
            addNoise: if addNoise is True then random noise is added to the observations from a normal distribution with mean 0 and standard deviation noiseSD. 
        """
        
        #TODO Need to write a unit test
        #use self.conc
        delta, _ = self.getGridStepSize()
        obs = np.zeros(len(self.sensormodel.obsLocs))
        for it,h in enumerate(self.sensormodel.getHs(self)):
            #TODO Make this faster - replacing the sums with matrix operations
            obs[it]=np.sum(h*self.conc)*np.prod(delta)
            if addNoise:
                obs[it]+=np.random.normal(0.0,self.noiseSD,1)  
        self.ySimulated = obs
        return obs
        

        
    def computeSourceFromPhi(self,z,coords=None):
        """
        uses getPhi from the kernel and a given z vector to generate a source function     
        set coords to a matrix: (3 x Grid Resolution), e.g. (3, 300, 80, 80)
                e.g. coords=np.asarray(np.meshgrid(tt,xx,yy,indexing='ij'))
        
        """
        if coords is None: coords = self.coords
        resolution = np.array(coords.shape[1:])
        self.source = np.zeros(resolution) 
        
        print("Computing Source from Phi...")
        for i,phi in enumerate(self.kernel.getPhi(coords)):
            print("%d/%d \r" % (i,self.kernel.N_feat),end="")
            self.source += phi*z[i]
        
        return self.source
        
    def computeModelRegressors(self):
        """
        Computes the regressor matrix X, using getHs from the sensor model and getPhi from the kernel.
        X here is used to infer the distribution of z (and hence the source).
        X is [features x observations]
        """
        delta, _ = self.getGridStepSize()
        
        X = np.zeros([self.N_feat,len(self.sensormodel.obsLocs)])
        
        adjs = []
        print("Calculating Adjoints...")
        for j,H in enumerate(self.sensormodel.getHs(self)):
            print("%d/%d \r" % (j,len(self.sensormodel.obsLocs)),end="")
            adjs.append(self.computeAdjoint(H))
        print("");
        #this will run out of memory...
        print("Calculating Phis...")
        for i,phi in enumerate(self.kernel.getPhi(self.coords)):
            print("%d/%d \r" % (i,self.N_feat),end="")
            for j,adj in enumerate(adjs):
                X[i,j] = np.sum((phi*adj))*np.prod(delta)
        print("");
        #phi * v, --> scale
        self.X = X
        return X
        
    def computeZDistribution(self,y):
        """
        Computes the z distribution using the regressor matrix and a vector of observations
        Arguments:
            y: a vector of observations (either generated using compute observations of given by the user in the real data case)
        """
        #uses self.X and observations y.
        print("Computing SS...",flush=True)
        SS = (1./(self.noiseSD**2))*(self.X@self.X.T) +np.eye(self.N_feat)
        print("Inverting SS...",flush=True)
        covZ =SSinv= np.linalg.inv(SS)
        print("Computing meanZ",flush=True)
        meanZ=(1./(self.noiseSD**2))*(SSinv@self.X@y) #sum_cc.flatten())
        print("Done",flush=True)
        return meanZ, covZ
        
        
    def computeSourceDistribution(self,meanZ,covZ):
        """
        Computes the S distribution (at each grid point) using the previously inferred mean and covariance of z. Does not compute joint distribution due to required size of covariance matrix
        Arguments:
            meanZ: an Nfeat long vector inferred using computeZDistribution
            covZ: an Nfeat x Nfeat matrix inferred using computeZDistribution
        """
        #uses self.X and observations y.
        #dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        delta, Ns = self.getGridStepSize()
        
        meanSource = self.computeSourceFromPhi(meanZ)
        varSource = np.zeros(Ns)#((Nt,Nx,Ny))
        for i,phii in enumerate(self.kernel.getPhi(self.coords)):
            for j,phij in enumerate(self.kernel.getPhi(self.coords)):
                varSource += covZ[i,j]*phii*phij
        
        
        return meanSource, varSource        
        

    def getSystemDerivative(self,conc,source):
        h_p=self.computeSystemDerivative(conc,source)
        return h_p
        
