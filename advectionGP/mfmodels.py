from advectionGP.models import AdjointAdvectionDiffusionModel
import numpy as np

class MeshFreeAdjointAdvectionDiffusionModel(AdjointAdvectionDiffusionModel):
    def computeAdjoint(self,H):
        assert False, "This isn't used in this child class, as we compute the Phi array in a single step, see computeModelRegressors()."
        
    def computeModelRegressors(self,Nparticles=10):
        """
        Computes the regressor matrix X, using the sensor model and getPhi from the kernel.
        X here is used to infer the distribution of z (and hence the source).
        X is [features x observations]
        
        Nparticles = number of particles PER OBSERVATION.
        
        uses just dt, Nt and boundary[0][0].
        """
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = self.getGridStepSize() #only bit we use is dt and Nt
        
        scale = Nparticles / dt

        particles = []
        N_obs = len(self.sensormodel.obsLocs)
        
        #Place particles at the observations...
        print("Initialising particles...")
        for obsi in range(N_obs):
            print("%d/%d \r" % (obsi,N_obs),end="")
            locA = self.sensormodel.obsLocs[obsi,[0,2,3]]
            locB = self.sensormodel.obsLocs[obsi,[1,2,3]]
            newparticles = np.repeat(locA[None,:],Nparticles,0).astype(float)
            newparticles[:,0]+=np.random.rand(len(newparticles))*(locB[0]-locA[0])
            particles.append(newparticles)
        particles = np.array(particles)

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
        return X
        
        
#MFModel = MeshFreeAdjointAdvectionDiffusionModel
