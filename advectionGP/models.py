import numpy as np
         
class AdvectionDiffusionModel():
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
        


        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        if (dx>=2*self.k_0/self.u): print("WARNING: spatial grid size does not meet the finite difference advection diffusion stability criteria")
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
        I.e. Given a valume in m getGridCoord returns the location on the grid
        
        todo: assertion for out of bounds value
        """
        return np.floor(self.resolution*(realPos - self.boundary[0])/(self.boundary[1]-self.boundary[0])).astype(int)
        
    def computeConcentration(self,source):        
        """
        Computes concentrations.
        Arguments:
         source == forcing function (shape: Nt x Nx x Ny). Can either be generated by ... or determine manually.
       
        returns array of concentrations (shape: Nt x Nx x Ny), given source. (also saved it in self.concentration)
        """
        #source = self.source
        
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
        
        self.conc = concentration
        return c
      
      
      
    def computeObservations(self,addNoise='FALSE'):
        """       
        Generates test observations by calculating the inner product of the filter function from the senor model and a given self.conc.
        Arguments:
            addNoise: if addNoise='TRUE' then random noise is added to the observations from a normal distribution with mean 0 and standard deviation noiseSD. 
        """
        
        #TODO Need to write a unit test
        #use self.conc
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        obs = np.zeros(len(self.sensormodel.obsLocs))
        for it,h in enumerate(self.sensormodel.getHs(self)):
            #TODO Make this faster - replacing the sums with matrix operations
            obs[it]=sum(sum(sum(h*self.conc)))*dt*dx*dy
            if addNoise=='TRUE':
                obs[it]+=np.random.normal(0.0,self.noiseSD,1)  
        self.ySimulated = obs
        return obs
        
     
        
    def computeSourceFromPhi(self,z):
        """
        uses getPhi from the kernel and a given z vector to generate a source function     
        """
        self.source = np.zeros(self.resolution) 
        for i,phi in enumerate(self.kernel.getPhi(self.coords)):
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

        v=np.zeros(((Nt,Nx,Ny)))
        v[-1,:,:]=0.0
        u=self.u
        k_0=self.k_0
        for i in range(1,Nt): #TODO might be better to rewrite as range(Nt-1,1,-1)...
    #Corner BCs   
            v[-i-1,0,0]=v[-i,0,0]+dt*(H[-i,0,0]) # BC at x=0, y=0
            v[-i-1,Nx-1,Ny-1]=v[-i,Nx-1,Ny-1]+dt*( H[-i,Nx-1,Ny-1]) # BC at x=xmax, y=ymax
            v[-i-1,0,Ny-1]=v[-i,0,Ny-1]+dt*( H[-i,0,Ny-1]) # BC at x=0, y=ymax
            v[-i-1,Nx-1,0]=v[-i,Nx-1,0]+dt*( H[-i,Nx-1,0]) # BC at x=xmax, y=0


    #Edge BCs   
            v[-i-1,Nx-1,1:Ny-1]=v[-i,Nx-1,1:Ny-1]+dt*(H[-i,Nx-1,1:Ny-1] +u*(v[-i,Nx-1,2:Ny]-v[-i,Nx-1,0:Ny-2] )/(2*dy) +k_0*(v[-i,Nx-1,2:Ny]-2*v[-i,Nx-1,1:Ny-1]+v[-i,Nx-1,0:Ny-2])/dy2) # BC at x=xmax        
            v[-i-1,0,1:Ny-1]=v[-i,0,1:Ny-1]+dt*(H[-i,0,1:Ny-1]+u*(v[-i,0,2:Ny]-v[-i,0,0:Ny-2] )/(2*dy) +k_0*(v[-i,0,2:Ny]-2*v[-i,0,1:Ny-1]+v[-i,0,0:Ny-2])/dy2 ) # BC at x=0

            v[-i-1,1:Nx-1,0]=v[-i,1:Nx-1,0]+dt*(   H[-i,1:Nx-1,0]+u*(v[-i,2:Nx,0]-v[-i,0:Nx-2,0] )/(2*dx) +k_0*(v[-i,2:Nx,0]-2*v[-i,1:Nx-1,0]+v[-i,0:Nx-2,0])/dx2  )# BC at y=0
            v[-i-1,1:Nx-1,Ny-1]=v[-i,1:Nx-1,Ny-1]+dt*(H[-i,1:Nx-1,Ny-1]+u*(v[-i,2:Nx,Ny-1]-v[-i,0:Nx-2,Ny-1] )/(2*dx)+k_0*(v[-i,2:Nx,Ny-1]-2*v[i,1:Nx-1,Ny-1]+v[-i,0:Nx-2,Ny-1])/dx2) # BC at y=ymax

    #Internal calculation (not on the boundary)
            v[-i-1,1:Nx-1,1:Ny-1]=v[-i,1:Nx-1,1:Ny-1] +dt*( H[-i,1:Nx-1,1:Ny-1]+u*(v[-i,2:Nx,1:Ny-1]-v[-i,0:Nx-2,1:Ny-1])/(2*dx) +u*(v[-i,1:Nx-1,2:Ny]-v[-i,1:Nx-1,0:Ny-2] )/(2*dy)+k_0*(v[-i,2:Nx,1:Ny-1]-2*v[-i,1:Nx-1,1:Ny-1]  +v[-i,0:Nx-2,1:Ny-1])/dx2+k_0*(v[-i,1:Nx-1,2:Ny]-2*v[-i,1:Nx-1,1:Ny-1]  +v[-i,1:Nx-1,0:Ny-2])/dy2 )
        return v


    def computeModelRegressors(self):
        """
        Computes the regressor matrix X, using getHs from the senor model and getPhi from the kernel.
        X here is used to infer the distribution of z (and hence the source)
        """
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        X = np.zeros([self.N_feat,len(self.sensormodel.obsLocs)])
        for j,H in enumerate(self.sensormodel.getHs(self)):
            adj=self.computeAdjoint(H)
            for i,phi in enumerate(self.kernel.getPhi(self.coords)):
                X[i,j] = sum((phi*adj*dt*dx*dy).flatten())
            
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
        
        SS = (1./(self.noiseSD**2))*(self.X@self.X.T) +np.eye(self.N_feat)
        covZ =SSinv= np.linalg.inv(SS)
        meanZ=(1./(self.noiseSD**2))*(SSinv @self.X@y) #sum_cc.flatten())
        return meanZ, covZ
        
        
        
        
        
        
        
        
class MCMCAdvectionDiffusionModel(AdvectionDiffusionModel):
    def computeLikelihood(self,pred_y,act_y):
        #compute how likely act_y is given our predictions in pred_y.
        pass
        
    def computeZDistribution(self):
        """
        """
        #uses self.X and observations y.
      
        z = np.random.randn(self.N_feat)
        
        #do MCMC looping here?
        source = self.computeSourceFromPhi(z)
        self.computeConcentration(source)
        pred_y = self.computeObservations()
        p = self.computeLikelihood(pred_y,y)
        #do MCMC decision stuff...?
        #return samples of z, or np.mean(z,1),np.cov(z,1) ##???!
        #return meanZ, covZ
        return np.zeros_like(z), np.eye(len(z))

