#from advectionGP.models import AdjointAdvectionDiffusionModel
from advectionGP.models.mesh_model import MeshModel
from scipy.interpolate import griddata
import numpy as np

def gethash(z):
    return hash(z.tobytes())
        
class MeshFreeAdjointAdvectionDiffusionModel(MeshModel):
    def __init__(self,boundary,resolution,kernel,noiseSD,sensormodel,windmodel,k_0,R=0,N_feat=25):
        super().__init__(boundary,resolution,kernel,noiseSD,sensormodel,N_feat)
        self.windmodel = windmodel
        self.k_0 = k_0
        #self.R=R  
        if R!=0: assert False, "Not yet implemented reaction term, set R to zero."      
        
    def computeAdjoint(self,H):
        assert False, "This isn't used in this child class, as we compute the Phi array in a single step, see computeModelRegressors()."
        
    def genParticlesFromObservations(self,Nparticles,sensormodel=None):
        if sensormodel is None:
            sensormodel = self.sensormodel
        particles = []
        N_obs = len(sensormodel.obsLocs)
        for obsi in range(N_obs):
            locA = sensormodel.obsLocs[obsi,[0,2,3]]
            locB = sensormodel.obsLocs[obsi,[1,2,3]]
            newparticles = np.repeat(locA[None,:],Nparticles,0).astype(float)
            newparticles[:,0]+=np.random.rand(len(newparticles))*(locB[0]-locA[0])
            particles.append(newparticles)
        particles = np.array(particles)
        return particles

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
                
    def computeModelRegressors(self,Nparticles=10):
        """
        Computes the regressor matrix X, using the sensor model and getPhi from the kernel.
        X here is used to infer the distribution of z (and hence the source).
        X is [features x observations]
        
        Nparticles = number of particles PER OBSERVATION.
        
        uses just dt, Nt and boundary[0][0].
        """
        delta, Ns = self.getGridStepSize()
        dt = delta[0]
        Nt = Ns[0]
        scale = Nparticles / dt

        particles = self.genParticlesFromObservations(Nparticles)
        
        #Place particles at the observations...
        print("Initialising particles...")
        
        N_obs = len(self.sensormodel.obsLocs)
        
        X = np.zeros([self.N_feat,N_obs])
        print("Diffusing particles...")
        for nit in range(Nt): 
            print("%d/%d \r" % (nit,Nt),end="")
            wind = self.windmodel.getwind(particles[:,:,1:])*dt #how much each particle moves due to wind [backwards]
            particles[:,:,1:]+=np.random.randn(particles.shape[0],particles.shape[1],2)*np.sqrt(2*dt*self.k_0) - wind
            particles[:,:,0]-=dt

            keep = particles[:,0,0]>self.boundary[0][0] #could extend to be within grid space
            X[:,keep] += np.sum(self.kernel.getPhiValues(particles),axis=(1))[:,keep]
            if np.sum(keep)==0: 
                break
        X = np.array(X)/scale
        self.X = X
        self.particles = particles
        return X
        
    def computeConcentration(self,meanZ=None,covZ=None,Nsamps=10,Nparticles=30,coords=None,particles=None,interpolateSource=False,Zs=None):
        """
        meanZ,covZ = mean and covariance of Z (used to sample Z, Nsamps times)
          or specify 'Zs' directly.
        use Nparticles to say how many particles per coord or observation to use
        if coords is None, then we use self.coords (i.e. the whole grid)
        alternatively set particles to specify their start directly...
        Compute the concentration using the particle approach.
        Nsamps = number of samples to take of Z. If you use just one, it uses the mean.
        Nparticles = number of particles to use
        returns mean and variance
        """
            
        delta, Ns = self.getGridStepSize() #only bit we use is dt and Nt
        dt = delta[0]
        Nt = Ns[0]

        #meanZ, covZ = self.computeZDistribution(Y) # Infers z vector mean and covariance using regressor matrix
        #sourceInfer = self.computeSourceFromPhi(meanZ) # Generates estimated source using mean of the inferred distribution
        if Zs is None:
            if Nsamps==1:
                Zs = meanZ[None,:]
            else:
                Zs = np.random.multivariate_normal(meanZ,covZ,Nsamps)
        else:
            Nsamps = len(Zs)

        #Place particles at the places of interest...
        print("Initialising particles...")
        
        if coords is None:
            coords = self.coords
            
        if particles is None:
            particles = coords.transpose([1,2,3,0]).copy()
            particles = particles[None,:].repeat(Nparticles,axis=0)
        print("Particle shape:")
        print(particles.shape)
        conc = np.zeros((Nsamps,)+particles.shape[:4]) #SAMPLING FROM Z
        print("Diffusing particles...")
        for nit in range(Nt):
            print("%d/%d \r" % (nit,Nt),end="")
            wind = self.windmodel.getwind(particles[...,1:])*dt #how much each particle moves due to wind [backwards]
            particles[...,1:]+=np.random.randn(particles.shape[0],particles.shape[1],particles.shape[2],particles.shape[3],2)*np.sqrt(2*dt*self.k_0) - wind
            particles[...,0]-=dt

            keep = particles[...,0]>self.boundary[0][0] #could extend to be within grid space
            
            if interpolateSource:
                sources = np.array([self.computeSourceFromPhiInterpolated(z, particles) for z in Zs])
            else:
                sources = np.array([self.computeSourceFromPhi(z, particles.transpose([4 ,0,1,2,3])) for z in Zs])            
            conc[:,keep] += sources[:,keep] #np.sum(sources)#[:,keep]
            if np.sum(keep)==0: 
                break
            
        conc = np.array(conc) #/scale
        conc = np.mean(conc,axis=1)*dt #average over particles
        return np.mean(conc,axis=0),np.var(conc,axis=0),conc
