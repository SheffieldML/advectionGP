#from advectionGP.models import AdjointAdvectionDiffusionModel
from advectionGP.models.mesh_model import MeshModel
from scipy.interpolate import griddata
import numpy as np

def gethash(z):
    return hash(z.tobytes())
        
class MeshFreeAdjointAdvectionDiffusionModel(MeshModel):
    def __init__(self,boundary,resolution,kernel,noiseSD,sensormodel,windmodel,k_0,R=0,N_feat=25):
        """
        This model uses particle approximation to compute the adjoints for the advection/diffusion model

        The Adjoint in all approaches in this module is used to compute the Phi matrix,
        (the 'design matrix' we can use to compute the posterior distribution or ML
        solution). Each element of Phi consists of the inner product of the solution to
        the adjoint for a given sensor, and one of the basis vectors.
        
        In the mesh-free approach we approximate these inner products by allowing
        particles to diffuse and advect following the adjoint (i.e. using L* rather
        than L). We effectively step backwards in time, and at each time step, evaluate
        all the bases at all the particle locations.

        As a consequence the computeAdjoint method is not implemented.
        
        Parameters
            boundary = a two element tuple of the corners of the grid 
                  ([start_time,start_spaceX,start_spaceY...],[end_time,end_spaceX,end_spaceY...]). e.g. ([0,0,0],[10,10,10])        
            resolution = a list of the grid size in each dimension. e.g. [10,20,20] (time first)
            kernel = the kernel to use
            noiseSD = the noise standard deviation
            sensormodel = an instatiation of a SensorModel class that implements the getHs method.
            windmodel = an instance of a Wind class (to build u using)
            k_0 = diffusion constant
            R = has to be zero at the moment, a reaction term is not implemented.
            N_feat = Number of features in the approximation.
        """
        super().__init__(boundary,resolution,kernel,noiseSD,sensormodel,N_feat)
        self.windmodel = windmodel
        self.k_0 = k_0
        #self.R=R
        if R!=0: assert False, "Not yet implemented reaction term, set R to zero."      
        
    def computeAdjoint(self,H):
        assert False, "This isn't used in this child class, as we compute the Phi array in a single step, see computeModelRegressors()."
        
    def genParticlesFromObservations(self,Nparticles,sensormodel=None):
        """
        We need to place particles at the sensors, which will then be iteratively
        moved around (based on the adjoint, L*).
    
        Parameters    
            Nparticles = Specify the number of particles per observation
            sensormodel = If you want to use different "sensors" (e.g. query
                            a point not used in training), set this parameter,
                            default to None - which uses the sensormodel used
                            for training.
        
        The method returns an array of [N_obs, Nparticles, ... ? ] #TODO
        """
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
        particles = particles.transpose([1,0,2])
        return particles

    def computeSourceFromPhiInterpolated(self,z,coords=None):
        """
        uses getPhi from the kernel and a given z vector to generate a source function     
        set coords to a matrix: (Grid Resolution x 3), e.g. (300, 80, 80, 3)
                e.g. coords=np.asarray(np.meshgrid(tt,xx,yy,indexing='ij'))
        if coords is not set, we use self.coords (transposed)
        
        coords gets transposed internally to be [tt,xx,yy,cc]       
        
        
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
        s[~np.all(keep,-1)]=0 #no contribution from space outside
        return s        
                
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
        #particles is Nparticles_per_obs x NumObservations x NumDims [e.g. 3]

        #Place particles at the observations...
        print("Initialising particles...")

        N_obs = len(self.sensormodel.obsLocs)

        X = np.zeros([self.N_feat,N_obs])
        print("Diffusing particles...")
        for nit in range(Nt):
            print("%d/%d \r" % (nit,Nt),end="",flush=True)
            wind = self.windmodel.getwind(particles[:,:,1:])*dt #how much each particle moves due to wind [backwards]
            particles[:,:,1:]+=np.random.randn(particles.shape[0],particles.shape[1],2)*np.sqrt(2*dt*self.k_0) - wind
            particles[:,:,0]-=dt

            #We remove a whole observation if it leaves the domain, not just single particles.
            #we do this by testing the first particle in the observation
            keep = particles[0,:,0]>self.boundary[0][0] #could extend to be within grid space
            #self.kernel.getPhiValues(particles) produces Nfeat x Nobs x Nparticles
            #we want to sum over the particles, so sum(above,axis=2), this will give us Nfeat x Nobs
            X[:,keep] += np.sum(self.kernel.getPhiValues(particles),axis=-1)[:,keep]
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
            ds = list(range(1,coords.ndim)); ds.insert(len(ds),0)
            particles = coords.copy() #transpose(ds).copy()
            print(particles.shape)
            particles = particles[None,:].repeat(Nparticles,axis=0)
            print(particles.shape)
        print("Particle shape:")
        assert particles.shape[-1]==len(self.resolution), "The last dimension of the particles array should be the dimensionality of the domain (e.g. 3 if [time,x,y])"
        print(particles.shape)
        conc = np.zeros((Nsamps,)+particles.shape[:-1]) #SAMPLING FROM Z
        print("Diffusing particles...")
        for nit in range(Nt):
            print("%d/%d \r" % (nit,Nt),end="",flush=True)
            
            wind = self.windmodel.getwind(particles[...,1:])*dt #how much each particle moves due to wind [backwards]
            particles[...,1:]+=np.random.randn(*particles.shape[:-1],2)*np.sqrt(2*dt*self.k_0) - wind
            particles[...,0]-=dt

            keep = particles[...,0]>self.boundary[0][0] #could extend to be within grid space

            if interpolateSource:
                sources = np.array([self.computeSourceFromPhiInterpolated(z, particles) for z in Zs])
            else:
                ds = list(range(particles.ndim-1)); ds.insert(0,particles.ndim-1)                
                sources = np.array([self.computeSourceFromPhi(z, particles.transpose(ds)) for z in Zs])                   
            conc[:,keep] += sources[:,keep] #np.sum(sources)#[:,keep]
            if np.sum(keep)==0: 
                break

        conc = np.array(conc) #/scale
        conc = np.mean(conc,axis=1)*dt #average over particles
        return np.mean(conc,axis=0),np.var(conc,axis=0),conc
