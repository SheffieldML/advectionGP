class AdvectionDiffusion1DModel():
    def __init__(self,boundary,resolution,kernel,noiseSD,sensormodel,windmodel,N_feat=25,spatial_averaging=1.0,k_0=0.001):
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
        self.windmodel = windmodel
        
        
        #coords is a D x (Nt,Nx,Ny) array of locations of the grid vertices.
        #TODO Maybe write more neatly...
        tt=np.linspace(self.boundary[0][0],self.boundary[1][0],self.resolution[0])
        xx=np.linspace(self.boundary[0][1],self.boundary[1][1],self.resolution[1])
        self.coords=np.asarray(np.meshgrid(tt,xx,yy,indexing='ij'))
        #self.coords=coords.reshape(self.N_D,self.resolution[0]*self.resolution[1]*self.resolution[2])
        
        #Compute some variables useful for PDEs
        
        self.u = self.windmodel.getu(self) #advection term: size 2 x resolution grid
        self.k_0 = k_0

        

        #assert self.X.shape[1]==4, "The X input matrix should be Nx4."
        #assert self.y.shape[0]==self.X.shape[0], "The length of X should equal the length of y"
        self.kernel = kernel
        self.kernel.generateFeatures(self.N_D,N_feat) 
        self.N_feat = N_feat
        
        self.sourcecache = {}

        dt,dx,dx2,Nt,Nx = self.getGridStepSize()
        self.mu = np.array([np.random.uniform(boundary[0][0],boundary[1][0],N_feat),np.random.uniform(boundary[0][1],boundary[1][1],N_feat)]).T
        if (dx>=2*self.k_0/np.min(np.abs(self.u))): print("WARNING: spatial grid size does not meet the finite difference advection diffusion stability criteria")
        if (dt>=dx2/(2*self.k_0)): print("WARNING: temporal grid size does not meet the finite difference advection diffusion stability criteria")

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
        dt=(self.boundary[1][0]-self.boundary[0][0])/self.resolution[0]
        dx=(self.boundary[1][1]-self.boundary[0][1])/self.resolution[1]
        dx2=dx*dx
        Nt=self.resolution[0]
        Nx=self.resolution[1]

        return dt,dx,dx2,Nt,Nx
        
    def getGridCoord(self,realPos):
        """
        Gets the location on the mesh for a real position
        I.e. Given a valume in m getGridCoord returns the location on the grid
        
        todo: assertion for out of bounds value
        
        Q: why is this floor and not round?
        """
        return np.floor(self.resolution*(realPos - self.boundary[0])/(self.boundary[1]-self.boundary[0])).astype(int)
        #return np.round(self.resolution*(realPos - self.boundary[0])/(self.boundary[1]-self.boundary[0])).astype(int)
    
    def computeConcentration(self,source,enforce_nonnegative=False):        
        """
        Computes concentrations.
        Arguments:
         source == forcing function (shape: Nt x Nx x Ny). Can either be generated by ... or determine manually.
         enforce_nonnegative = default False,. Setting to true will force concentration to be non-negative each iteration.
        returns array of concentrations (shape: Nt x Nx x Ny), given source. (also saved it in self.concentration)
        """
        #source = self.source
        
        #get the grid step sizes, their squares and the size of the grid
        dt,dx,dx2,Nt,Nx = self.getGridStepSize()
        
        c=np.zeros(((self.resolution)))
        
        c[0,:]=0

        k_0 = self.k_0
        u = self.u
        for i in range(0,Nt-1):
            # Corner BCs 
            c[i+1,0]=c[i,0]+dt*( source[i,0] ) +dt*k_0*( 2*c[i,1]-2*c[i,0])/dx2
            c[i+1,Nx-1]=c[i,Nx-1]+dt*( source[i,Nx-1])+dt*k_0*( 2*c[i,Nx-2]-2*c[i,Nx-1])/dx2
            #for k in range(1,Ny-1):
                # Internal Calc
            c[i+1,1:Nx-1]=c[i,1:Nx-1] +dt*(source[i,1:Nx-1]-u[0][i,1:Nx-1]*(c[i,2:Nx]-c[i,0:Nx-2])/(2*dx) +k_0*(c[i,2:Nx]-2*c[i,1:Nx-1]  +c[i,0:Nx-2])/dx2)
            if enforce_nonnegative: c[c<0]=0
        concentration = c 
        
        self.conc = concentration
        return c
   
      
    def computeObservations(self,addNoise=False):
        """       
        Generates test observations by calculating the inner product of the filter function from the senor model and a given self.conc.
        Arguments:
            addNoise: if addNoise is True then random noise is added to the observations from a normal distribution with mean 0 and standard deviation noiseSD. 
        """
        
        #TODO Need to write a unit test
        #use self.conc
        dt,dx,dx2,Nt,Nx = self.getGridStepSize()
        obs = np.zeros(len(self.sensormodel.obsLocs))
        for it,h in enumerate(self.sensormodel.getHs(self)):
            #TODO Make this faster - replacing the sums with matrix operations
            obs[it]=np.sum(h*self.conc)*dt*dx
            if addNoise:
                obs[it]+=np.random.normal(0.0,self.noiseSD,1)  
        self.ySimulated = obs
        return obs
        
     
        
    def computeSourceFromPhi(self,z,coords=None,compact=False):
        """
        uses getPhi from the kernel and a given z vector to generate a source function     
        set coords to a matrix: (3 x Grid Resolution), e.g. (3, 300, 80, 80)
                e.g. coords=np.asarray(np.meshgrid(tt,xx,yy,indexing='ij'))
        
        """
        if coords is None: coords = self.coords
        resolution = np.array(coords.shape[1:])
        self.source = np.zeros(resolution) 
        if compact==True:
            print("Computing Source from Phi...")
            for i,phi in enumerate(self.kernel.getPhiCompact2D(self.mu,self.coords)):
                print("%d/%d \r" % (i,self.kernel.N_feat),end="")
                self.source += phi*z[i]
        else:
            print("Computing Source from Phi...")
            for i,phi in enumerate(self.kernel.getPhi(coords)):
                print("%d/%d \r" % (i,self.kernel.N_feat),end="")
                self.source += phi*z[i]
        return self.source
        
class AdjointAdvectionDiffusionModel(AdvectionDiffusionModel):
    def computeAdjoint(self,H):
        """
        Runs the backward PDE (adjoint problem)
        Gets called for an observation instance (H).
        (v is the result of the adjoint operation)
        """
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()

        v=np.zeros(((self.resolution)))
        v[-1,:]=0.0
        u=self.u
        k_0=self.k_0
        for i in range(1,Nt): #TODO might be better to rewrite as range(Nt-1,1,-1)...
    #Corner BCs   
            v[-i-1,0]=v[-i,0]+dt*(H[-i,0]) # BC at x=0, y=0
            v[-i-1,Nx-1]=v[-i,Nx-1]+dt*( H[-i,Nx-1]) # BC at x=xmax, y=ymax

    #Internal calculation (not on the boundary)
            v[-i-1,1:Nx-1]=v[-i,1:Nx-1] +dt*( H[-i,1:Nx-1]+u[0][-i,1:Nx-1]*(v[-i,2:Nx]-v[-i,0:Nx-2])/(2*dx) +k_0*(v[-i,2:Nx]-2*v[-i,1:Nx-1]  +v[-i,0:Nx-2])/dx2)
        return v


    def computeModelRegressors(self,compact=False):
        """
        Computes the regressor matrix X, using getHs from the sensor model and getPhi from the kernel.
        X here is used to infer the distribution of z (and hence the source).
        X is [features x observations]
        """
        dt,dx,dx2,Nt,Nx = self.getGridStepSize()
        X = np.zeros([self.N_feat,len(self.sensormodel.obsLocs)])
        
        adjs = []
        print("Calculating Adjoints...")
        for j,H in enumerate(self.sensormodel.getHs(self)):
            print("%d/%d \r" % (j,len(self.sensormodel.obsLocs)),end="")
            adjs.append(self.computeAdjoint(H))
        print("");
        #this will run out of memory...
        print("Calculating Phis...")
        if compact==True:
            print("Computing Source from Phi...")
            for i,phi in enumerate(self.kernel.getPhiCompact2D(self.mu,self.coords)):
                print("%d/%d \r" % (i,len(self.kernel.W)),end="")
                for j,adj in enumerate(adjs):
                    X[i,j] = np.sum((phi*adj))*dt*dx
        else:
            for i,phi in enumerate(self.kernel.getPhi(self.coords)):
                print("%d/%d \r" % (i,len(self.kernel.W)),end="")
                for j,adj in enumerate(adjs):
                    X[i,j] = np.sum((phi*adj))*dt*dx
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
        dt,dx,dx2,Nt,Nx = self.getGridStepSize()
        
        meanSource = self.computeSourceFromPhi(meanZ)
        varSource = np.zeros((Nt,Nx))
        for i,phii in enumerate(self.kernel.getPhi(self.coords)):
            for j,phij in enumerate(self.kernel.getPhi(self.coords)):
                varSource += covZ[i,j]*phii*phij
        
        
        return meanSource, varSource    