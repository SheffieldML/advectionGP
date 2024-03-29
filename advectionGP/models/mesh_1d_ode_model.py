import numpy as np
from advectionGP.models.mesh_model import MeshModel

class SecondOrderODEModel(MeshModel):
    def __init__(self,boundary,resolution,kernel,noiseSD,sensormodel,N_feat=25,k_0=0.001,u=0.001,eta=0.001):
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
        super().__init__(boundary,resolution,kernel,noiseSD,sensormodel,N_feat)
        
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
        
    def computeResponse(self,source):        
        """
        Computes the system response over time (returns a vector)
        Arguments:
        source == forcing function (shape: Nt). Can either be generated by ... or determine manually.
        returns array of concentrations (shape: Nt), given source. (also saved it in self.conc)
        """
        delta, Ns = self.getGridStepSize()
        Nt = Ns[0]
        dt = delta[0]
        dt2 = dt**2
        
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
  
    
class AdjointSecondOrderODEModel(SecondOrderODEModel):
    def computeAdjoint(self,H):
        """
        Runs the backward second order ODE (adjoint problem)
        Gets called for an observation instance (H).
        (v is the result of the adjoint operation)
        """
        delta, Ns = self.getGridStepSize()
        Nt = Ns[0]        
        dt = delta[0]
        dt2 = dt**2
        
        k_0=self.k_0
        u=self.u
        eta=self.eta
        v=np.zeros(((self.resolution)))
        v[Nt-1] = 0

        v[Nt-2] = (1.0/(-k_0+u*dt/2))*( H[Nt-1]*(dt**2)+ k_0*v[Nt-1]+u*dt*v[Nt-1]/2.0- 2*k_0*v[Nt-1]-eta*(dt**2)*v[Nt-1]     )
    
        for i in reversed(range(1,Nt-1)):
            #i=Nt-j-2
            v[i-1]=(1.0/(-k_0+u*dt/2))*( H[i]*(dt2)+ k_0*v[i+1]+u*dt*v[i+1]/2.0- 2*k_0*v[i]-eta*(dt2)*v[i]     )
        
        return v
    
    def computeSystemDerivative(self,conc,source):
        # return * self.k_0 due to slight discrepancy in adjoint equation between systems
        delta, Ns = self.getGridStepSize()
        dmH=np.array([np.gradient(conc,delta[0])/self.k_0,conc/self.k_0,(-self.u*np.gradient(conc,delta[0])-self.eta*conc+source)/self.k_0**2])*self.k_0
        return dmH
    
    def assignParameters(self,params):
        self.k_0=params[2]
        self.u=params[0]
        self.eta=params[1]
        
    def computeSourceLengthscaleDerivative(self,samples,obs,samp):
        dmH=-self.computeSourceDerivative(samples,obs,samp)
        return dmH    

