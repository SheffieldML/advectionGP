import numpy as np
from scipy.interpolate import griddata

def gethash(z):
    return hash(z.tobytes())
        
#def squash(M):
#    if M.shape[0]==3:
#        return M.reshape(M.shape[0],np.prod(M.shape[1:]))
#    if M.shape[-1]==3:
#        return M.reshape(np.prod(M.shape[1:],M.shape[-1]))


class AdvectionDiffusionModel():
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
        assert self.N_D==3, "Currently AdvectionDiffusionModel only supports a 3d grid: T,X,Y. Check your resolution parameter."
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
        yy=np.linspace(self.boundary[0][2],self.boundary[1][2],self.resolution[2])
        self.coords=np.asarray(np.meshgrid(tt,xx,yy,indexing='ij'))
        #self.coords=coords.reshape(self.N_D,self.resolution[0]*self.resolution[1]*self.resolution[2])
      
        #Compute some variables useful for PDEs
        
        self.u = self.windmodel.getu(self) #advection term: size 2 x resolution grid
        self.k_0 = k_0

        

        #assert self.X.shape[1]==4, "The X input matrix should be Nx4."
        #assert self.y.shape[0]==self.X.shape[0], "The length of X should equal the length of y"
        self.kernel = kernel
        self.kernel.generateFeatures(self.N_D,N_feat,boundary) 
        self.N_feat = N_feat
        
        self.sourcecache = {}

        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
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
    
            c[i+1,1:Nx-1,0]=c[i,1:Nx-1,0]+dt*(source[i,1:Nx-1,0]-u[0][i,1:Nx-1,0]*(c[i,2:Nx,0]-c[i,0:Nx-2,0])/(2*dx)+k_0*(2*c[i,1:Nx-1,1]-2*c[i,1:Nx-1,0])/dy2 +k_0*(c[i,2:Nx,0]-2*c[i,1:Nx-1,0]+c[i,0:Nx-2,0] )/dx2     )
            c[i+1,1:Nx-1,Ny-1]=c[i,1:Nx-1,Ny-1]+dt*( source[i,1:Nx-1,Ny-1]-u[0][i,1:Nx-1,Ny-1]*(c[i,2:Nx,Ny-1]-c[i,0:Nx-2,Ny-1])/(2*dx)+k_0*(2*c[i,1:Nx-1,Ny-2]-2*c[i,1:Nx-1,Ny-1])/dy2 +k_0*(c[i,2:Nx,Ny-1]-2*c[i,1:Nx-1,Ny-1]+c[i,0:Nx-2,Ny-1] )/dx2     )  
            #for k in range(1,Ny-1):
                # x edge bcs
            c[i+1,Nx-1,1:Ny-1]=c[i,Nx-1,1:Ny-1]+dt*( source[i,Nx-1,1:Ny-1]-u[1][i,Nx-1,1:Ny-1]*(c[i,Nx-1,2:Ny]-c[i,Nx-1,0:Ny-2])/(2*dy)+k_0*(2*c[i,Nx-2,1:Ny-1]-2*c[i,Nx-1,1:Ny-1])/dx2 +k_0*(c[i,Nx-1,2:Ny]-2*c[i,Nx-1,1:Ny-1]+c[i,Nx-1,0:Ny-2] )/dy2     )
            c[i+1,0,1:Ny-1]=c[i,0,1:Ny-1]+dt*( source[i,0,1:Ny-1]-u[1][i,0,1:Ny-1]*(c[i,0,2:Ny]-c[i,0,0:Ny-2])/(2*dy)+k_0*(2*c[i,1,1:Ny-1]-2*c[i,0,1:Ny-1])/dx2 +k_0*(c[i,0,2:Ny]-2*c[i,0,1:Ny-1]+c[i,0,0:Ny-2] )/dy2     )     
                # Internal Calc
            c[i+1,1:Nx-1,1:Ny-1]=c[i,1:Nx-1,1:Ny-1] +dt*(source[i,1:Nx-1,1:Ny-1]-u[0][i,1:Nx-1,1:Ny-1]*(c[i,2:Nx,1:Ny-1]-c[i,0:Nx-2,1:Ny-1])/(2*dx) -u[1][i,1:Nx-1,1:Ny-1]*(c[i,1:Nx-1,2:Ny]-c[i,1:Nx-1,0:Ny-2] )/(2*dy) +k_0*(c[i,2:Nx,1:Ny-1]-2*c[i,1:Nx-1,1:Ny-1]  +c[i,0:Nx-2,1:Ny-1])/dx2+k_0*(c[i,1:Nx-1,2:Ny]-2*c[i,1:Nx-1,1:Ny-1]  +c[i,1:Nx-1,0:Ny-2])/dy2 )
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
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        obs = np.zeros(len(self.sensormodel.obsLocs))
        for it,h in enumerate(self.sensormodel.getHs(self)):
            #TODO Make this faster - replacing the sums with matrix operations
            obs[it]=np.sum(h*self.conc)*dt*dx*dy
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
        
        
    def computeSourceFromPhiInterpolated(self,z,coords=None):
        """
        uses getPhi from the kernel and a given z vector to generate a source function     
        set coords to a matrix: (3 x Grid Resolution), e.g. (3, 300, 80, 80)
                e.g. coords=np.asarray(np.meshgrid(tt,xx,yy,indexing='ij'))
        
        """
        if coords is None: coords = self.coords.transpose([1,2,3,0])
        
        zhash = gethash(z)
        if zhash not in self.sourcecache:
            print("cache miss, computing source from phi...")
            source = self.computeSourceFromPhi(z)
            self.sourcecache[zhash] = source
        else:
            #print("cache hit")
            source = self.sourcecache[zhash]
        
        gcs = self.getGridCoord(coords)
        keep = (gcs<source.shape) & (gcs>=0)
        gcs[~keep]=0 #just set to something that won't break stuff
        s = source[gcs[...,0],gcs[...,1],gcs[...,2]]
        s[~np.all(keep,-1)]=0
        return s
        
        
        #resolution = np.array(coords.shape[1:])
        #self.source = np.zeros(resolution) 
        
        
        #self.temp = self.coords, source, coords
        #return self.computeSourceFromPhi(z,coords.transpose([4,0,1,2,3]))
        
        #print("Interpolating...")
        #sourceatcoords = griddata(squash(self.coords).T, source.flatten(), coords, method='linear')
        #print("Done")
        #return sourceatcoords




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
            v[-i-1,Nx-1,1:Ny-1]=v[-i,Nx-1,1:Ny-1]+dt*(H[-i,Nx-1,1:Ny-1] +u[1][-i,Nx-1,1:Ny-1]*(v[-i,Nx-1,2:Ny]-v[-i,Nx-1,0:Ny-2] )/(2*dy) +k_0*(v[-i,Nx-1,2:Ny]-2*v[-i,Nx-1,1:Ny-1]+v[-i,Nx-1,0:Ny-2])/dy2) # BC at x=xmax        
            v[-i-1,0,1:Ny-1]=v[-i,0,1:Ny-1]+dt*(H[-i,0,1:Ny-1]+u[1][-i,0,1:Ny-1]*(v[-i,0,2:Ny]-v[-i,0,0:Ny-2] )/(2*dy) +k_0*(v[-i,0,2:Ny]-2*v[-i,0,1:Ny-1]+v[-i,0,0:Ny-2])/dy2 ) # BC at x=0

            v[-i-1,1:Nx-1,0]=v[-i,1:Nx-1,0]+dt*(   H[-i,1:Nx-1,0]+u[0][-i,1:Nx-1,0]*(v[-i,2:Nx,0]-v[-i,0:Nx-2,0] )/(2*dx) +k_0*(v[-i,2:Nx,0]-2*v[-i,1:Nx-1,0]+v[-i,0:Nx-2,0])/dx2  )# BC at y=0
            v[-i-1,1:Nx-1,Ny-1]=v[-i,1:Nx-1,Ny-1]+dt*(H[-i,1:Nx-1,Ny-1]+u[0][-i,1:Nx-1,Ny-1]*(v[-i,2:Nx,Ny-1]-v[-i,0:Nx-2,Ny-1] )/(2*dx)+k_0*(v[-i,2:Nx,Ny-1]-2*v[i,1:Nx-1,Ny-1]+v[-i,0:Nx-2,Ny-1])/dx2) # BC at y=ymax

    #Internal calculation (not on the boundary)
            v[-i-1,1:Nx-1,1:Ny-1]=v[-i,1:Nx-1,1:Ny-1] +dt*( H[-i,1:Nx-1,1:Ny-1]+u[0][-i,1:Nx-1,1:Ny-1]*(v[-i,2:Nx,1:Ny-1]-v[-i,0:Nx-2,1:Ny-1])/(2*dx) +u[1][-i,1:Nx-1,1:Ny-1]*(v[-i,1:Nx-1,2:Ny]-v[-i,1:Nx-1,0:Ny-2] )/(2*dy)+k_0*(v[-i,2:Nx,1:Ny-1]-2*v[-i,1:Nx-1,1:Ny-1]  +v[-i,0:Nx-2,1:Ny-1])/dx2+k_0*(v[-i,1:Nx-1,2:Ny]-2*v[-i,1:Nx-1,1:Ny-1]  +v[-i,1:Nx-1,0:Ny-2])/dy2 )
        return v


    def computeModelRegressors(self):
        """
        Computes the regressor matrix X, using getHs from the sensor model and getPhi from the kernel.
        X here is used to infer the distribution of z (and hence the source).
        X is [features x observations]
        """
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
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
                X[i,j] = np.sum((phi*adj))*dt*dx*dy
        print("");
        #phi * v, --> scale
        self.X = X
        return X
        
        
    def computeAdjointTest(self,H):
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
            v[-i-1,0,0]= v[-i,0,0] +dt*(H[-i,0,0]-(u[0][-i,0,0]**2)*v[-i,0,0]/k_0-(u[1][-i,0,0]**2)*v[-i,0,0]/k_0+k_0*v[-i,1,0]/dx2-2*k_0*v[-i,0,0]/dx2+k_0*(v[-i,1,0]+2*u[0][-i,0,0]*dx*v[-i,0,0]/k_0)/dx2+k_0*v[-i,0,1]/dy2-2*k_0*v[-i,0,0]/dy2+k_0*(v[-i,0,1]+2*u[1][-i,0,0]*dy*v[-i,0,0]/k_0)/dy2) # BC at x=0, y=0
            v[-i-1,Nx-1,Ny-1]= v[-i,Nx-1,Ny-1] +dt*(H[-i,Nx-1,Ny-1]-(u[0][-i,Nx-1,Ny-1]**2)*v[-i,Nx-1,Ny-1]/k_0-(u[1][-i,Nx-1,Ny-1]**2)*v[-i,Nx-1,Ny-1]/k_0+k_0*(v[-i,Nx-2,Ny-1]-2*u[0][-i,Nx-1,Ny-1]*dx*v[-i,Nx-1,Ny-1]/k_0)/dx2-2*k_0*v[-i,Nx-1,Ny-1]/dx2+k_0*v[-i,Nx-2,Ny-1]/dx2+k_0*(v[-i,Nx-1,Ny-2]-2*u[1][-i,Nx-1,Ny-1]*dy*v[-i,Nx-1,Ny-1]/k_0)/dy2-2*k_0*v[-i,Nx-1,Ny-1]/dy2+k_0*v[-i,Nx-1,Ny-2]/dy2) # BC at x=xmax, y=ymax
            v[-i-1,0,Ny-1]=v[-i,0,Ny-1]+dt*( H[-i,0,Ny-1]-(u[0][-i,0,Ny-1]**2)*v[-i,0,Ny-1]/k_0-(u[1][-i,0,Ny-1]**2)*v[-i,0,Ny-1]/k_0+k_0*v[-i,1,Ny-1]/dx2-2*k_0*v[-i,0,Ny-1]/dx2+k_0*(v[-i,1,Ny-1]+2*u[0][-i,0,Ny-1]*dx*v[-i,0,Ny-1]/k_0)/dx2+k_0*(v[-i,0,Ny-2]-2*u[1][-i,0,Ny-1]*dy*v[-i,0,Ny-1]/k_0)/dy2-2*k_0*v[-i,0,Ny-1]/dy2+k_0*v[-i,0,Ny-2]/dy2) # BC at x=0, y=ymax
            v[-i-1,Nx-1,0]=v[-i,Nx-1,0]+dt*( H[-i,Nx-1,0]-(u[0][-i,Nx-1,0]**2)*v[-i,Nx-1,0]/k_0-(u[1][-i,Nx-1,0]**2)*v[-i,Nx-1,0]/k_0+k_0*(v[-i,Nx-2,0]-2*u[0][-i,Nx-1,0]*dx*v[-i,Nx-1,0]/k_0)/dx2-2*k_0*v[-i,Nx-1,0]/dx2+k_0*v[-i,Nx-2,0]/dx2+k_0*v[-i,Nx-1,1]/dy2-2*k_0*v[-i,Nx-1,0]/dy2+k_0*(v[-i,Nx-1,1]+2*u[1][-i,Nx-1,0]*dy*v[-i,Nx-1,0]/k_0)/dy2) # BC at x=xmax, y=0
            
    #Edge BCs   
            v[-i-1,Nx-1,1:Ny-1]=v[-i,Nx-1,1:Ny-1]+dt*(H[-i,Nx-1,1:Ny-1] +u[1][-i,Nx-1,1:Ny-1]*(v[-i,Nx-1,2:Ny]-v[-i,Nx-1,0:Ny-2] )/(2*dy) +k_0*(v[-i,Nx-1,2:Ny]-2*v[-i,Nx-1,1:Ny-1]+v[-i,Nx-1,0:Ny-2])/dy2-(u[0][-i,Nx-1,1:Ny-1]**2)*v[-i,Nx-1,1:Ny-1]/k_0+k_0*(v[-i,Nx-2,1:Ny-1]-2*u[0][-i,Nx-1,1:Ny-1]*dx*v[-i,Nx-1,1:Ny-1]/k_0)/dx2-2*k_0*v[-i,Nx-1,1:Ny-1]/dx2+k_0*v[-i,Nx-2,1:Ny-1]/dx2) # BC at x=xmax        
            v[-i-1,0,1:Ny-1]=v[-i,0,1:Ny-1]+dt*(H[-i,0,1:Ny-1]+u[1][-i,0,1:Ny-1]*(v[-i,0,2:Ny]-v[-i,0,0:Ny-2] )/(2*dy) +k_0*(v[-i,0,2:Ny]-2*v[-i,0,1:Ny-1]+v[-i,0,0:Ny-2])/dy2 -(u[0][-i,0,1:Ny-1]**2)*v[-i,0,0]/k_0+k_0*v[-i,1,1:Ny-1]/dx2-2*k_0*v[-i,0,1:Ny-1]/dx2+k_0*(v[-i,1,1:Ny-1]+2*u[0][-i,0,1:Ny-1]*dx*v[-i,0,1:Ny-1]/k_0)/dx2) # BC at x=0

            v[-i-1,1:Nx-1,0]=v[-i,1:Nx-1,0]+dt*(H[-i,1:Nx-1,0]+u[0][-i,1:Nx-1,0]*(v[-i,2:Nx,0]-v[-i,0:Nx-2,0] )/(2*dx) +k_0*(v[-i,2:Nx,0]-2*v[-i,1:Nx-1,0]+v[-i,0:Nx-2,0])/dx2 -(u[1][-i,1:Nx-1,0]**2)*v[-i,1:Nx-1,0]/k_0 +k_0*v[-i,1:Nx-1,1]/dy2-2*k_0*v[-i,1:Nx-1,0]/dy2+k_0*(v[-i,1:Nx-1,1]+2*u[1][-i,1:Nx-1,0]*dy*v[-i,1:Nx-1,0]/k_0)/dy2)# BC at y=0
            v[-i-1,1:Nx-1,Ny-1]=v[-i,1:Nx-1,Ny-1]+dt*(H[-i,1:Nx-1,Ny-1]+u[0][-i,1:Nx-1,Ny-1]*(v[-i,2:Nx,Ny-1]-v[-i,0:Nx-2,Ny-1] )/(2*dx)+k_0*(v[-i,2:Nx,Ny-1]-2*v[i,1:Nx-1,Ny-1]+v[-i,0:Nx-2,Ny-1])/dx2-(u[1][-i,1:Nx-1,Ny-1]**2)*v[-i,1:Nx-1,Ny-1]/k_0+k_0*(v[-i,1:Nx-1,Ny-2]-2*u[1][-i,1:Nx-1,Ny-1]*dy*v[-i,1:Nx-1,Ny-1]/k_0)/dy2-2*k_0*v[-i,1:Nx-1,Ny-1]/dy2+k_0*v[-i,1:Nx-1,Ny-2]/dy2) # BC at y=ymax

    #Internal calculation (not on the boundary)
            v[-i-1,1:Nx-1,1:Ny-1]=v[-i,1:Nx-1,1:Ny-1] +dt*( H[-i,1:Nx-1,1:Ny-1]+u[0][-i,1:Nx-1,1:Ny-1]*(v[-i,2:Nx,1:Ny-1]-v[-i,0:Nx-2,1:Ny-1])/(2*dx) +u[1][-i,1:Nx-1,1:Ny-1]*(v[-i,1:Nx-1,2:Ny]-v[-i,1:Nx-1,0:Ny-2] )/(2*dy)+k_0*(v[-i,2:Nx,1:Ny-1]-2*v[-i,1:Nx-1,1:Ny-1]  +v[-i,0:Nx-2,1:Ny-1])/dx2+k_0*(v[-i,1:Nx-1,2:Ny]-2*v[-i,1:Nx-1,1:Ny-1]  +v[-i,1:Nx-1,0:Ny-2])/dy2 )
        return v


    def computeModelRegressorsTest(self):
        """
        Computes the regressor matrix X, using getHs from the sensor model and getPhi from the kernel.
        X here is used to infer the distribution of z (and hence the source).
        X is [features x observations]
        """
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        X = np.zeros([self.N_feat,len(self.sensormodel.obsLocs)])
        
        adjs = []
        print("Calculating Adjoints...")
        for j,H in enumerate(self.sensormodel.getHs(self)):
            print("%d/%d \r" % (j,len(self.sensormodel.obsLocs)),end="")
            adjs.append(self.computeAdjointTest(H))
        print("");
        #this will run out of memory...
        print("Calculating Phis...")
        for i,phi in enumerate(self.kernel.getPhi(self.coords)):
            print("%d/%d \r" % (i,self.N_feat),end="")
            for j,adj in enumerate(adjs):
                X[i,j] = np.sum((phi*adj))*dt*dx*dy
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
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        
        meanSource = self.computeSourceFromPhi(meanZ)
        varSource = np.zeros((Nt,Nx,Ny))
        for i,phii in enumerate(self.kernel.getPhi(self.coords)):
            for j,phij in enumerate(self.kernel.getPhi(self.coords)):
                varSource += covZ[i,j]*phii*phij
        
        
        return meanSource, varSource    
        
        
class AdvectionDiffusionReactionModel(AdvectionDiffusionModel):
    def __init__(self,boundary,resolution,kernel,noiseSD,sensormodel,windmodel,N_feat=25,spatial_averaging=1.0,k_0=0.001,R=0.001):
        """
        The Advection Diffusion Reaction Model.
        
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
            R = reaction constant
        """
        super().__init__(boundary,resolution,kernel,noiseSD,sensormodel,windmodel,N_feat,spatial_averaging,k_0)        
        self.R=R
                
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
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        
        c=np.zeros(((self.resolution)))
        
        c[0,:,:]=0

        k_0 = self.k_0
        u = self.u
        R = self.R
        for i in range(0,Nt-1):
            # Corner BCs 
            c[i+1,0,0]=c[i,0,0]+dt*( source[i,0,0] ) +dt*k_0*( 2*c[i,1,0]-2*c[i,0,0])/dx2 + dt*k_0*( 2*c[i,0,1]-2*c[i,0,0])/dy2 -dt*R*c[i,0,0]
            c[i+1,Nx-1,Ny-1]=c[i,Nx-1,Ny-1]+dt*( source[i,Nx-1,Ny-1])+dt*k_0*( 2*c[i,Nx-2,Ny-1]-2*c[i,Nx-1,Ny-1])/dx2 + dt*k_0*( 2*c[i,Nx-1,Ny-2]-2*c[i,Nx-1,Ny-1])/dy2-dt*R*c[i,Nx-1,Ny-1]
            c[i+1,0,Ny-1]=c[i,0,Ny-1]+dt*( source[i,0,Ny-1] ) +dt*k_0*( 2*c[i,1,Ny-1]-2*c[i,0,Ny-1])/dx2 + dt*k_0*( 2*c[i,0,Ny-2]-2*c[i,0,Ny-1])/dy2-dt*R*c[i,0,Ny-1]
            c[i+1,Nx-1,0]=c[i,Nx-1,0]+dt*( source[i,Nx-1,0])+dt*k_0*( 2*c[i,Nx-2,0]-2*c[i,Nx-1,0])/dx2 + dt*k_0*( 2*c[i,Nx-1,1]-2*c[i,Nx-1,0])/dy2-dt*R*c[i,Nx-1,0]
    
            c[i+1,1:Nx-1,0]=c[i,1:Nx-1,0]+dt*(source[i,1:Nx-1,0]-u[0][i,1:Nx-1,0]*(c[i,2:Nx,0]-c[i,0:Nx-2,0])/(2*dx)+k_0*(2*c[i,1:Nx-1,1]-2*c[i,1:Nx-1,0])/dy2 +k_0*(c[i,2:Nx,0]-2*c[i,1:Nx-1,0]+c[i,0:Nx-2,0] )/dx2 - R*c[i,1:Nx-1,0]     )
            c[i+1,1:Nx-1,Ny-1]=c[i,1:Nx-1,Ny-1]+dt*( source[i,1:Nx-1,Ny-1]-u[0][i,1:Nx-1,Ny-1]*(c[i,2:Nx,Ny-1]-c[i,0:Nx-2,Ny-1])/(2*dx)+k_0*(2*c[i,1:Nx-1,Ny-2]-2*c[i,1:Nx-1,Ny-1])/dy2 +k_0*(c[i,2:Nx,Ny-1]-2*c[i,1:Nx-1,Ny-1]+c[i,0:Nx-2,Ny-1] )/dx2 - R*c[i,1:Nx-1,Ny-1]     )  
            #for k in range(1,Ny-1):
                # x edge bcs
            c[i+1,Nx-1,1:Ny-1]=c[i,Nx-1,1:Ny-1]+dt*( source[i,Nx-1,1:Ny-1]-u[1][i,Nx-1,1:Ny-1]*(c[i,Nx-1,2:Ny]-c[i,Nx-1,0:Ny-2])/(2*dy)+k_0*(2*c[i,Nx-2,1:Ny-1]-2*c[i,Nx-1,1:Ny-1])/dx2 +k_0*(c[i,Nx-1,2:Ny]-2*c[i,Nx-1,1:Ny-1]+c[i,Nx-1,0:Ny-2] )/dy2  - R*c[i,Nx-1,1:Ny-1]   )
            c[i+1,0,1:Ny-1]=c[i,0,1:Ny-1]+dt*( source[i,0,1:Ny-1]-u[1][i,0,1:Ny-1]*(c[i,0,2:Ny]-c[i,0,0:Ny-2])/(2*dy)+k_0*(2*c[i,1,1:Ny-1]-2*c[i,0,1:Ny-1])/dx2 +k_0*(c[i,0,2:Ny]-2*c[i,0,1:Ny-1]+c[i,0,0:Ny-2] )/dy2 - R*c[i,0,1:Ny-1]    )     
                # Internal Calc
            c[i+1,1:Nx-1,1:Ny-1]=c[i,1:Nx-1,1:Ny-1] +dt*(source[i,1:Nx-1,1:Ny-1]-u[0][i,1:Nx-1,1:Ny-1]*(c[i,2:Nx,1:Ny-1]-c[i,0:Nx-2,1:Ny-1])/(2*dx) -u[1][i,1:Nx-1,1:Ny-1]*(c[i,1:Nx-1,2:Ny]-c[i,1:Nx-1,0:Ny-2] )/(2*dy) +k_0*(c[i,2:Nx,1:Ny-1]-2*c[i,1:Nx-1,1:Ny-1]  +c[i,0:Nx-2,1:Ny-1])/dx2+k_0*(c[i,1:Nx-1,2:Ny]-2*c[i,1:Nx-1,1:Ny-1]  +c[i,1:Nx-1,0:Ny-2])/dy2 - R*c[i,1:Nx-1,1:Ny-1])
            if enforce_nonnegative: c[c<0]=0
        concentration = c 
        
        self.conc = concentration
        return c      

class AdjointAdvectionDiffusionReactionModel(AdvectionDiffusionReactionModel,AdjointAdvectionDiffusionModel):
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
        R=self.R
        for i in range(1,Nt): #TODO might be better to rewrite as range(Nt-1,1,-1)...
    #Corner BCs   
            v[-i-1,0,0]=v[-i,0,0]+dt*(H[-i,0,0]-R*v[-i,0,0]) # BC at x=0, y=0
            v[-i-1,Nx-1,Ny-1]=v[-i,Nx-1,Ny-1]+dt*( H[-i,Nx-1,Ny-1]-R*v[-i,Nx-1,Ny-1]) # BC at x=xmax, y=ymax
            v[-i-1,0,Ny-1]=v[-i,0,Ny-1]+dt*( H[-i,0,Ny-1]-R*v[-i,0,Ny-1]) # BC at x=0, y=ymax
            v[-i-1,Nx-1,0]=v[-i,Nx-1,0]+dt*( H[-i,Nx-1,0]-R*v[-i,Nx-1,0]) # BC at x=xmax, y=0


    #Edge BCs   
            v[-i-1,Nx-1,1:Ny-1]=v[-i,Nx-1,1:Ny-1]+dt*(H[-i,Nx-1,1:Ny-1] +u[1][-i,Nx-1,1:Ny-1]*(v[-i,Nx-1,2:Ny]-v[-i,Nx-1,0:Ny-2] )/(2*dy) +k_0*(v[-i,Nx-1,2:Ny]-2*v[-i,Nx-1,1:Ny-1]+v[-i,Nx-1,0:Ny-2])/dy2-R*v[-i,Nx-1,1:Ny-1]) # BC at x=xmax        
            v[-i-1,0,1:Ny-1]=v[-i,0,1:Ny-1]+dt*(H[-i,0,1:Ny-1]+u[1][-i,0,1:Ny-1]*(v[-i,0,2:Ny]-v[-i,0,0:Ny-2] )/(2*dy) +k_0*(v[-i,0,2:Ny]-2*v[-i,0,1:Ny-1]+v[-i,0,0:Ny-2])/dy2 -R*v[-i,0,1:Ny-1]) # BC at x=0

            v[-i-1,1:Nx-1,0]=v[-i,1:Nx-1,0]+dt*(   H[-i,1:Nx-1,0]+u[0][-i,1:Nx-1,0]*(v[-i,2:Nx,0]-v[-i,0:Nx-2,0] )/(2*dx) +k_0*(v[-i,2:Nx,0]-2*v[-i,1:Nx-1,0]+v[-i,0:Nx-2,0])/dx2  -R*v[-i,1:Nx-1,0])# BC at y=0
            v[-i-1,1:Nx-1,Ny-1]=v[-i,1:Nx-1,Ny-1]+dt*(H[-i,1:Nx-1,Ny-1]+u[0][-i,1:Nx-1,Ny-1]*(v[-i,2:Nx,Ny-1]-v[-i,0:Nx-2,Ny-1] )/(2*dx)+k_0*(v[-i,2:Nx,Ny-1]-2*v[i,1:Nx-1,Ny-1]+v[-i,0:Nx-2,Ny-1])/dx2-R*v[-i,1:Nx-1,Ny-1]) # BC at y=ymax

    #Internal calculation (not on the boundary)
            v[-i-1,1:Nx-1,1:Ny-1]=v[-i,1:Nx-1,1:Ny-1] +dt*( H[-i,1:Nx-1,1:Ny-1]+u[0][-i,1:Nx-1,1:Ny-1]*(v[-i,2:Nx,1:Ny-1]-v[-i,0:Nx-2,1:Ny-1])/(2*dx) +u[1][-i,1:Nx-1,1:Ny-1]*(v[-i,1:Nx-1,2:Ny]-v[-i,1:Nx-1,0:Ny-2] )/(2*dy)+k_0*(v[-i,2:Nx,1:Ny-1]-2*v[-i,1:Nx-1,1:Ny-1]  +v[-i,0:Nx-2,1:Ny-1])/dx2+k_0*(v[-i,1:Nx-1,2:Ny]-2*v[-i,1:Nx-1,1:Ny-1]  +v[-i,1:Nx-1,0:Ny-2])/dy2 -R*v[-i,1:Nx-1,1:Ny-1])
        return v

class SimpleODEModel(AdvectionDiffusionModel):
    def __init__(self,boundary,resolution,kernel,noiseSD,sensormodel,N_feat=25,spatial_averaging=1.0):
        """
        The Advection Diffusion Reaction Model.
        
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
            R = reaction constant
        """
        super().__init__(boundary,resolution,kernel,noiseSD,sensormodel,N_feat,spatial_averaging)        
        
                
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
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        
        c=np.zeros(((self.resolution)))
        
        c[0,:,:]=0

        for i in range(0,Nt-1):
            # Internal Calc
            c[i+1,:,:]=c[i,:,:] +dt*(source[i,:,:])
            if enforce_nonnegative: c[c<0]=0
        concentration = c 
        
        self.conc = concentration
        return c      

class AdjointSimpleODEModel(SimpleODEModel,AdjointAdvectionDiffusionModel):
    def computeAdjoint(self,H):
        """
        Runs the backward PDE (adjoint problem)
        Gets called for an observation instance (H).
        (v is the result of the adjoint operation)
        """
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()

        v=np.zeros(((Nt,Nx,Ny)))
        v[-1,:,:]=0.0
        for i in range(1,Nt): #TODO might be better to rewrite as range(Nt-1,1,-1)...
    #Internal calculation (not on the boundary)
            v[-i-1,:,:]=v[-i,:,:] +dt*( H[-i,:,:])
        return v
        
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

    
class SecondOrderODEModel():
    def __init__(self,boundary,resolution,kernel,noiseSD,sensormodel,N_feat=25,spatial_averaging=1.0,k_0=0.001,u=0.001,eta=0.001):
        """
        Second order linear ODE model of form
        
        -k_0*y"(t) + u*y'(t) + eta*y(t) = f(t)
        
        At the moment we assume a 1d grid [time].
        
        Parameters:
            boundary = a two element tuple of the corners of the grid. e.g. ([0],[10])        
            resolution = a list of the grid size in each dimension. e.g. [10]
            kernel = the kernel to use
            noiseSD = the noise standard deviation
            sensormodel = an instatiation of a SensorModel class that implements the getHs1D method.
            N_feat = number of fourier features
            k_0 = parameter multiplying second derivative
            u = parameter multiplying first derivative
            eta  = parameter multiplying y
           
        
        """
        #TODO URGENT: The spatial averaging doesn't make sense!
        
        
        self.N_D = len(resolution)
        assert self.N_D==1, "Currently SecondOrderODEModel only supports a 1d grid: T. Check your resolution parameter."
        assert self.N_D==len(boundary[0]), "The dimensions in the boundary don't match those in the grid resolution"
        assert self.N_D==len(boundary[1]), "The dimensions in the boundary don't match those in the grid resolution"


        self.boundary = [np.array(boundary[0]),np.array(boundary[1])]
        self.resolution = np.array(resolution)
        self.noiseSD = noiseSD
        self.sensormodel = sensormodel
        
        
        #coords is a D x Nt array of locations of the grid vertices.
        #TODO Maybe write more neatly...
        self.coords=np.linspace(self.boundary[0],self.boundary[1],self.resolution[0]).T
      
        #Establish ODE variables
        
        self.k_0 = k_0
        self.u=u
        self.eta=eta
        
        self.kernel = kernel
        self.kernel.generateFeatures(self.N_D,N_feat,boundary) 
        self.N_feat = N_feat
        
        dt,dt2,Nt = self.getGridStepSize()
        
    def getGridStepSize(self):
        """
        Calculates useful scalars for the PDE model
        outputs:
            dt: time grid size
            dt2 = dt**2
            Nt: Number of evaluation points in time
        """
        dt=(self.boundary[1]-self.boundary[0])/self.resolution[0]
        dt2=dt*dt
        Nt=self.resolution[0]
        return dt,dt2,Nt
        
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
        source == forcing function (shape: Nt). Can either be generated by ... or determine manually.
        returns array of concentrations (shape: Nt x Nx x Ny), given source. (also saved it in self.concentration)
        """
        #source = self.source
        
        #get the grid step sizes, their squares and the size of the grid
        
        # TODO - need to work out if there are options for other initial conditions?
        dt,dt2,Nt = self.getGridStepSize()
        
        x=np.zeros(((self.resolution)))

        k_0 = self.k_0
        u = self.u
        eta=self.eta

        x[0] = 0
        #x[1] = 0
        x[1] = ( 1.0/(-k_0+u*dt/2))*(k_0*(-2*x[0] + x[0] )+  (u*dt*x[0])/2.0- eta*x[0]*(dt2)+ source[0]*(dt2))
        for i in range(1, Nt-1):
            x[i+1] = ( 1.0/(-k_0+u*dt/2))*(k_0*(-2*x[i] + x[i-1] )+  (u*dt*x[i-1])/2.0- eta*x[i]*(dt2)+ source[i]*(dt2))
        
        self.conc = x
        return x
   
      
    def computeObservations(self,addNoise='FALSE'):
        """       
        Generates test observations by calculating the inner product of the filter function from the senor model and a given self.conc.
        Arguments:
            addNoise: if addNoise='TRUE' then random noise is added to the observations from a normal distribution with mean 0 and standard deviation noiseSD. 
        """
        
        #TODO Need to write a unit test
        #use self.conc
        dt,dt2,Nt = self.getGridStepSize()
        obs = np.zeros(len(self.sensormodel.obsLocs))
        for it,h in enumerate(self.sensormodel.getHs1D(self)):
            #TODO Make this faster - replacing the sums with matrix operations
            obs[it]=np.sum(h*self.conc)*dt
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
    
class AdjointSecondOrderODEModel(SecondOrderODEModel):
    def computeAdjoint(self,H):
        """
        Runs the backward second order ODE (adjoint problem)
        Gets called for an observation instance (H).
        (v is the result of the adjoint operation)
        """
        dt,dt2,Nt = self.getGridStepSize()
        k_0=self.k_0
        u=self.u
        eta=self.eta
        v=np.zeros(((self.resolution)))
        v[Nt-1] = 0
        #v[Nt-2]=0
        v[Nt-2] = (1.0/(-k_0+u*dt/2))*( H[Nt-1]*(dt**2)+ k_0*v[Nt-1]+u*dt*v[Nt-1]/2.0- 2*k_0*v[Nt-1]-eta*(dt**2)*v[Nt-1]     )
    
        for i in reversed(range(1,Nt-1)):
            #i=Nt-j-2
            v[i-1]=(1.0/(-k_0+u*dt/2))*( H[i]*(dt2)+ k_0*v[i+1]+u*dt*v[i+1]/2.0- 2*k_0*v[i]-eta*(dt2)*v[i]     )
        
        return v


    def computeModelRegressors(self):
        """
        Computes the regressor matrix X, using getHs1D from the senor model and getPhi from the kernel.
        X here is used to infer the distribution of z (and hence the source)
        """
        print("Getting Grid Step Size",flush=True)
        dt,dt2,Nt = self.getGridStepSize()
        print("Building X matrix",flush=True)
        X = np.zeros([self.N_feat,len(self.sensormodel.obsLocs)])
        
        adjs = []
        print("Calculating Adjoints...",flush=True)
        for j,H in enumerate(self.sensormodel.getHs1D(self)):
            print("%d/%d \r" % (j,len(self.sensormodel.obsLocs)),end="",flush=True)
            adjs.append(self.computeAdjoint(H))
        print("");
        #this will run out of memory...
        print("Calculating Phis...",flush=True)
        for i,phi in enumerate(self.kernel.getPhi(self.coords)):
            print("%d/%d \r" % (i,self.N_feat),end="",flush=True)
            for j,adj in enumerate(adjs):
                X[i,j] = np.sum(phi*adj)*dt
                
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
        
        print("Calculating SS")
        SS = (1./(self.noiseSD**2))*(self.X@self.X.T) +np.eye(self.N_feat)
        print("Inverting SS")
        covZ =SSinv= np.linalg.inv(SS)
        print("Computing meanZ")
        meanZ=(1./(self.noiseSD**2))*(SSinv @self.X@y) #sum_cc.flatten())
        print("Done")
        return meanZ, covZ   
    
    
class ShiftOperatorModel():
    def __init__(self,boundary,resolution,kernel,noiseSD,sensormodel,N_feat=25,spatial_averaging=1.0):
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
        assert self.N_D==3, "Currently AdvectionDiffusionModel only supports a 3d grid: T,X,Y. Check your resolution parameter."
        assert self.N_D==len(boundary[0]), "The dimensions in the boundary don't match those in the grid resolution"
        assert self.N_D==len(boundary[1]), "The dimensions in the boundary don't match those in the grid resolution"


        self.boundary = [np.array(boundary[0]),np.array(boundary[1])]
        self.resolution = np.array(resolution)
        self.noiseSD = noiseSD
        self.sensormodel = sensormodel
        
        
        #coords is a D x (Nt,Nx,Ny) array of locations of the grid vertices.
        #TODO Maybe write more neatly...
        tt=np.linspace(self.boundary[0][0],self.boundary[1][0],self.resolution[0])
        xx=np.linspace(self.boundary[0][1],self.boundary[1][1],self.resolution[1])
        yy=np.linspace(self.boundary[0][2],self.boundary[1][2],self.resolution[2])
        self.coords=np.asarray(np.meshgrid(tt,xx,yy,indexing='ij'))
        #self.coords=coords.reshape(self.N_D,self.resolution[0]*self.resolution[1]*self.resolution[2])
      
        #Compute some variables useful for PDEs

        

        #assert self.X.shape[1]==4, "The X input matrix should be Nx4."
        #assert self.y.shape[0]==self.X.shape[0], "The length of X should equal the length of y"
        self.kernel = kernel
        self.kernel.generateFeatures(self.N_D,N_feat,boundary) 
        self.N_feat = N_feat
        


        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()

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
        
        Q: why is this floor and not round?
        """
        return np.floor(self.resolution*(realPos - self.boundary[0])/(self.boundary[1]-self.boundary[0])).astype(int)
        #return np.round(self.resolution*(realPos - self.boundary[0])/(self.boundary[1]-self.boundary[0])).astype(int)
   
      
    def computeObservations(self,addNoise=False):
        """       
        Generates test observations by calculating the inner product of the filter function from the senor model and a given self.conc.
        Arguments:
            addNoise: if addNoise is True then random noise is added to the observations from a normal distribution with mean 0 and standard deviation noiseSD. 
        """
        
        #TODO Need to write a unit test
        #use self.conc
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        obs = np.zeros(len(self.sensormodel.obsLocs))
        for it,h in enumerate(self.sensormodel.getHs(self)):
            #TODO Make this faster - replacing the sums with matrix operations
            obs[it]=np.sum(h*self.conc)*dt*dx*dy
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
        
        for i,phi in enumerate(self.kernel.getPhi(self.coords)):
            self.source += phi*z[i]
        
        return self.source
                
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
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()
        
        c=np.zeros(((self.resolution)))
        
        

        for i in range(5,Nt):
            # Internal Calc
            c[i,:,:]=(source[i-5,:,:])
            if enforce_nonnegative: c[c<0]=0
        concentration = c 
        
        self.conc = concentration
        return c      

class AdjointShiftOperatorModel(ShiftOperatorModel,AdjointAdvectionDiffusionModel):
    def computeAdjoint(self,H):
        """
        Runs the backward PDE (adjoint problem)
        Gets called for an observation instance (H).
        (v is the result of the adjoint operation)
        """
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize()

        v=np.zeros(((Nt,Nx,Ny)))
     
        for i in range(1,Nt-5): #TODO might be better to rewrite as range(Nt-1,1,-1)...
    #Internal calculation (not on the boundary)
            v[i,:,:]= H[i+5,:,:]
        return v
    
class ShiftOperator1DModel():
    def __init__(self,boundary,resolution,kernel,noiseSD,sensormodel,N_feat=25,spatial_averaging=1.0,a=1):
        """
        1D shift operator model of the form u(t+a)=f(t)
        
        At the moment we assume a 1d grid [time].
        
        Parameters:
            boundary = a two element tuple of the corners of the grid. e.g. ([0],[10])        
            resolution = a list of the grid size in each dimension. e.g. [10]
            kernel = the kernel to use
            noiseSD = the noise standard deviation
            sensormodel = an instatiation of a SensorModel class that implements the getHs1D method.
            N_feat = number of fourier features
            a = parameter denoting the shift
           
        
        """
        #TODO URGENT: The spatial averaging doesn't make sense!
        
        
        self.N_D = len(resolution)
        assert self.N_D==1, "Currently SecondOrderODEModel only supports a 1d grid: T. Check your resolution parameter."
        assert self.N_D==len(boundary[0]), "The dimensions in the boundary don't match those in the grid resolution"
        assert self.N_D==len(boundary[1]), "The dimensions in the boundary don't match those in the grid resolution"


        self.boundary = [np.array(boundary[0]),np.array(boundary[1])]
        self.resolution = np.array(resolution)
        self.noiseSD = noiseSD
        self.sensormodel = sensormodel
        dt,dt2,Nt = self.getGridStepSize()
        self.a = a
        #coords is a D x Nt array of locations of the grid vertices.
        #TODO Maybe write more neatly...
        self.coords=np.linspace(self.boundary[0],self.boundary[1],self.resolution[0]).T
        self.shiftCoords=np.linspace(self.boundary[0]-self.a,self.boundary[1],self.resolution[0]+int(self.a/dt)).T
      
        #Establish ODE variables
        
        
        
        self.kernel = kernel
        self.kernel.generateFeatures(self.N_D,N_feat,boundary) 
        self.N_feat = N_feat
        
        
        
    def getGridStepSize(self):
        """
        Calculates useful scalars for the PDE model
        outputs:
            dt: time grid size
            dt2 = dt**2
            Nt: Number of evaluation points in time
        """
        dt=(self.boundary[1]-self.boundary[0])/self.resolution[0]
        dt2=dt*dt
        Nt=self.resolution[0]
        return dt,dt2,Nt
        
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
        source == forcing function (shape: Nt). Can either be generated by ... or determine manually.
        returns array of concentrations (shape: Nt x Nx x Ny), given source. (also saved it in self.concentration)
        """
        #source = self.source
        
        #get the grid step sizes, their squares and the size of the grid
        
        # TODO - need to work out if there are options for other initial conditions?
        dt,dt2,Nt = self.getGridStepSize()
        
        x=np.zeros(((self.resolution)))

        a = np.int(self.a/dt)
        for i in range (0,Nt):
            x[i]=source[i]

        self.conc = x
        return x
   
      
    def computeObservations(self,addNoise='FALSE'):
        """       
        Generates test observations by calculating the inner product of the filter function from the senor model and a given self.conc.
        Arguments:
            addNoise: if addNoise='TRUE' then random noise is added to the observations from a normal distribution with mean 0 and standard deviation noiseSD. 
        """
        
        #TODO Need to write a unit test
        #use self.conc
        dt,dt2,Nt = self.getGridStepSize()
        obs = np.zeros(len(self.sensormodel.obsLocs))
        for it,h in enumerate(self.sensormodel.getHs1D(self)):
            #TODO Make this faster - replacing the sums with matrix operations
            obs[it]=np.sum(h*self.conc)*dt
            if addNoise=='TRUE':
                obs[it]+=np.random.normal(0.0,self.noiseSD,1)  
        self.ySimulated = obs
        return obs
        
     
        
    def computeSourceFromPhi(self,z,coords):
        """
        uses getPhi from the kernel and a given z vector to generate a source function     
        """
        dt,dt2,Nt = self.getGridStepSize()
        self.source = np.zeros(coords.shape[1]) 
        for i,phi in enumerate(self.kernel.getPhi(coords)):
            
            self.source += phi*z[i]
            
        return self.source
    
class AdjointShiftOperator1DModel(ShiftOperator1DModel):
    def computeAdjoint(self,H):
        """
        Runs the backward second order ODE (adjoint problem)
        Gets called for an observation instance (H).
        (v is the result of the adjoint operation)
        """
        dt,dt2,Nt = self.getGridStepSize()
        a = np.int(self.a/dt)
        b=int(self.resolution+a)
        v=np.zeros(self.resolution+a) 
        for i in range(1,Nt): #TODO might be better to rewrite as range(Nt-1,1,-1)...
    #Internal calculation (not on the boundary)
            v[i]= H[i]
        #v=v[a:b]
        return v

    def computeModelRegressors(self):
        """
        Computes the regressor matrix X, using getHs1D from the senor model and getPhi from the kernel.
        X here is used to infer the distribution of z (and hence the source)
        """
        print("Getting Grid Step Size",flush=True)
        dt,dt2,Nt = self.getGridStepSize()
        print("Building X matrix",flush=True)
        X = np.zeros([self.N_feat,len(self.sensormodel.obsLocs)])
        
        adjs = []
        print("Calculating Adjoints...",flush=True)
        for j,H in enumerate(self.sensormodel.getHs1D(self)):
            print("%d/%d \r" % (j,len(self.sensormodel.obsLocs)),end="",flush=True)
            adjs.append(self.computeAdjoint(H))
        print("");
        #this will run out of memory...
        print("Calculating Phis...",flush=True)
        for i,phi in enumerate(self.kernel.getPhi(self.shiftCoords)):
            print("%d/%d \r" % (i,self.N_feat),end="",flush=True)
            for j,adj in enumerate(adjs):
                X[i,j] = np.sum(phi*adj)*dt
                
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
        
        print("Calculating SS")
        SS = (1./(self.noiseSD**2))*(self.X@self.X.T) +np.eye(self.N_feat)
        print("Inverting SS")
        covZ =SSinv= np.linalg.inv(SS)
        print("Computing meanZ")
        meanZ=(1./(self.noiseSD**2))*(SSinv @self.X@y) #sum_cc.flatten())
        print("Done")
        return meanZ, covZ   
    

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
        self.coords=np.asarray(np.meshgrid(tt,xx,indexing='ij'))
        #self.coords=coords.reshape(self.N_D,self.resolution[0]*self.resolution[1]*self.resolution[2])
        
        #Compute some variables useful for PDEs
        
        self.u = self.windmodel.getu(self) #advection term: size 2 x resolution grid
        self.k_0 = k_0

        

        #assert self.X.shape[1]==4, "The X input matrix should be Nx4."
        #assert self.y.shape[0]==self.X.shape[0], "The length of X should equal the length of y"
        self.kernel = kernel
        self.kernel.generateFeatures(self.N_D,N_feat,boundary) 
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
        for it,h in enumerate(self.sensormodel.getHs2D(self)):
            #TODO Make this faster - replacing the sums with matrix operations
            obs[it]=np.sum(h*self.conc)*dt*dx
            if addNoise:
                obs[it]+=np.random.normal(0.0,self.noiseSD,1)  
        self.ySimulated = obs
        return obs
        
     
        
    def computeSourceFromPhi(self,z,coords=None,gaussian=False):
        """
        uses getPhi from the kernel and a given z vector to generate a source function     
        set coords to a matrix: (3 x Grid Resolution), e.g. (3, 300, 80, 80)
                e.g. coords=np.asarray(np.meshgrid(tt,xx,yy,indexing='ij'))
        
        """
        if coords is None: coords = self.coords
        resolution = np.array(coords.shape[1:])
        self.source = np.zeros(resolution) 

        print("Computing Source from Phi...")
        for i,phi in enumerate(self.kernel.getPhi2D(coords)):
            print("%d/%d \r" % (i,self.kernel.N_feat),end="")
            self.source += phi*z[i]
        return self.source
        
class AdjointAdvectionDiffusion1DModel(AdvectionDiffusion1DModel):
    def computeAdjoint(self,H):
        """
        Runs the backward PDE (adjoint problem)
        Gets called for an observation instance (H).
        (v is the result of the adjoint operation)
        """
        dt,dx,dx2,Nt,Nx = self.getGridStepSize()

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


    def computeModelRegressors(self,gaussian=False):
        """
        Computes the regressor matrix X, using getHs from the sensor model and getPhi from the kernel.
        X here is used to infer the distribution of z (and hence the source).
        X is [features x observations]
        """
        dt,dx,dx2,Nt,Nx = self.getGridStepSize()
        X = np.zeros([self.N_feat,len(self.sensormodel.obsLocs)])
        
        adjs = []
        print("Calculating Adjoints...")
        for j,H in enumerate(self.sensormodel.getHs2D(self)):
            print("%d/%d \r" % (j,len(self.sensormodel.obsLocs)),end="")
            adjs.append(self.computeAdjoint(H))
        print("");
        #this will run out of memory...
        print("Calculating Phis...")


        for i,phi in enumerate(self.kernel.getPhi2D(self.coords)):
            print("%d/%d \r" % (i,self.N_feat),end="")
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
