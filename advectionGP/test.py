from advectionGP.models import AdvectionDiffusionModel
from advectionGP.sensors import FixedSensorModel
from advectionGP.kernels import EQ

import numpy as np

import numpy as np
import unittest


class TestKernels(unittest.TestCase):

    def test_kernel(self):
        X = np.array([[2,3,6,4]]) #not used really.
        boundary = ([0,0,0],[10,10,10])
        
        k = EQ(3.0, 2.0)
        
        sensors = FixedSensorModel(X,2)
        m = AdvectionDiffusionModel(resolution=[40,4,4],boundary=boundary,N_feat=10000,noiseSD=5.0,kernel=k,sensormodel=sensors)

        Phi = np.zeros(np.r_[m.N_feat,m.resolution])
        for i,phi in enumerate(m.kernel.getPhi(m.coords)):
            Phi[i,:,:,:] = phi


        x = m.coords[0,:,0,0]
        largest_error = np.max(np.abs(np.sum(Phi[:,:,0,0]*Phi[:,20:21,0,0],0)-2*np.exp(-.5*(x-x[20])**2/(3.0**2))))
        
        #import matplotlib.pyplot as plt
        #plt.plot(x,np.sum(Phi[:,:,0,0]*Phi[:,20:21,0,0],0))
        #plt.plot(x,2*np.exp(-.5*(x-x[20])**2/(3.0**2)))
        self.assertLess(largest_error,0.1,"Approximation to EQ kernel is wrong.")
        #self.assertAlmostEqual(approx,exact,1,"Approximation to EQ kernel is wrong.")

    def test_grid_volume(self):
        X = np.array([[0,10,3,5]])
        y = np.array([12])#unused

        boundary = ([0,0,0],[10,10,10])
        k = EQ(1.0, 2.0)

        sensors = FixedSensorModel(X,2)
        m = AdvectionDiffusionModel(resolution=[200,200,200],boundary=boundary,N_feat=15,noiseSD=5.0,kernel=k,sensormodel=sensors)

        volume_of_grid_tile = np.prod((np.array(boundary[1])-np.array(boundary[0]))/m.resolution)
        
        #get first/only h
        for h in sensors.getHs(m):
            break
            
        self.assertAlmostEqual(np.sum(h)*volume_of_grid_tile,1,5,"Integral of h over space and time doesn't equal one.")
       
    def test_forward_model(self):

        X = np.array([[0,10,3,5]])#not used
        y = np.array([12])#not used

        boundary = ([0,0,0],[20,20,20])
        k = EQ(1.0, 2.0) #not used
        sensors = FixedSensorModel(X,2)#not used
        
        #given the advection and diffusion parameters, we can compute the expected Gaussian pollution after the 20s.
        m = AdvectionDiffusionModel(resolution=[100,100,100],boundary=boundary,N_feat=15,noiseSD=5.0,kernel=k,sensormodel=sensors,u=0.09,k_0=0.01)

        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = m.getGridStepSize()
        
        #impulse response of single pike of pollution, over a time period of dt.
        source = np.zeros(m.resolution)
        source[0,int(Nx/2),int(Ny/2)] = 1.0

        #estimate using our model
        estimated_concentration = m.computeConcentration(source)

        #compute the predicted analytic solution for an infinitesimal spike of pollution
        x = np.linspace(boundary[0][0],boundary[1][0],Nx)
        t = ((Nt-1)/m.resolution[0])*m.boundary[1][0]
        new_centre = (m.boundary[1][1]+dx)/2+m.u*t
        c = np.exp(-(x-new_centre)**2/(4*m.k_0*t))

        #the height isn't relevant as we have to normalise things anyway.

        conc_snapshot = estimated_concentration[Nt-1,int(Nx/2),:].copy()
        conc_snapshot/=np.sum(conc_snapshot)
        conc_snapshot*=np.sum(c)

        #plt.plot(x,conc_snapshot)
        #plt.plot(x,c)
        #plt.xlim([10.5,12.5])
        largest_error = np.max(np.abs(conc_snapshot-c))
        self.assertLess(largest_error,0.1,"Advection/diffusion of impulse response not equal to expected Gaussian")
        
        #we also expect the pollution added to add up to 1, 
        self.assertAlmostEqual(np.sum(estimated_concentration[2,:,:])/dt,1)
        
        #and not change during diffusion
        self.assertAlmostEqual(np.sum(estimated_concentration[2,:,:])/dt,1)
    
    def testAdjoint(self):
        X = np.array([[17,18,10,10]])
        y = np.array([12])

        boundary = ([0,0,0],[20,20,20])
        k = EQ(1.0, 2.0)
        sensors = FixedSensorModel(X,1)
        m = AdvectionDiffusionModel(resolution=[100,20,20],boundary=boundary,N_feat=150,noiseSD=5.0,kernel=k,sensormodel=sensors,u=0.01,k_0=0.005)

        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = m.getGridStepSize()
        source = np.zeros(m.resolution)
        source[1,int(Nx/2)-1,int(Ny/2)-1] = 1.0
        estimated_concentration = m.computeConcentration(source)
        v = m.computeAdjoint(list(sensors.getHs(m))[0])
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = m.getGridStepSize()
        self.assertAlmostEqual(np.sum(v*source)*dt*dx*dy,m.computeObservations()[0])
    
    def testRegressor(self):
        X = np.array([[17,18,10,10],[7,8,5,5],[10,15,12,15]])
        y = np.array([np.nan,np.nan,np.nan])

        boundary = ([0,0,0],[20,20,20])
        k = EQ(2, 2.0)
        sensors = FixedSensorModel(X,1)
        m = AdvectionDiffusionModel(resolution=[30,30,30],boundary=boundary,N_feat=100,noiseSD=5.0,kernel=k,sensormodel=sensors,u=0.01,k_0=0.05)

        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = m.getGridStepSize()

        z = np.random.randn(m.N_feat,1)
        source = m.computeSourceFromPhi(z)

        m.computeConcentration(source)
        #predicted_observations = m.computeObservations()

        v = np.array([m.computeAdjoint(h) for h in sensors.getHs(m)])
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = m.getGridStepSize()
        X = m.computeModelRegressors()


        calc_usingAdjoint = np.sum(v*source,(1,2,3))*dt*dx*dy
        calc_usingForwardModel = m.computeObservations()
        calc_usingRegressors = (X.T@z)[:,0]


        self.assertLess(np.max(np.abs(calc_usingAdjoint-calc_usingForwardModel)),0.001)
        self.assertLess(np.max(np.abs(calc_usingAdjoint-calc_usingRegressors)),0.001)  
        
    def testDistribution(self):
        X = np.array([[17,18,10,10],[7,8,5,5],[10,15,12,15]])
        y = np.array([np.nan,np.nan,np.nan])

        boundary = ([0,0,0],[20,20,20])
        k = EQ(2, 2.0)
        sensors = FixedSensorModel(X,1)
        m = AdvectionDiffusionModel(resolution=[30,30,30],boundary=boundary,N_feat=150,noiseSD=5.0,kernel=k,sensormodel=sensors,u=0.01,k_0=0.05)
        X2=np.identity(m.N_feat)
        y2=np.ones(m.N_feat)
        varZ, meanZ = m.computeZDistribution(X2,y2)
        varTest=m.N_feat*(1+1/(m.noiseSD**2))
        meanTest=m.N_feat*(1/m.noiseSD**2)*1/(1+1/(m.noiseSD**2))
        self.assertAlmostEqual(1,1)
        self.assertAlmostEqual(sum(meanZ),meanTest)
        self.assertAlmostEqual(np.sum(np.linalg.inv(varZ)),varTest)
        

    
if __name__ == '__main__':
    unittest.main()
