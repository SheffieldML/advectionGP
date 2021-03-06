from advectionGP.models import AdjointAdvectionDiffusionModel
from advectionGP.models import AdjointSecondOrderODEModel
from advectionGP.sensors import FixedSensorModel
from advectionGP.kernels import EQ
from advectionGP.wind import WindFixU 

import numpy as np
import unittest


class TestKernels(unittest.TestCase):

    def test_kernel(self):
        """
        Checks that the fourier expansion approximates an EQ kernel
        """
        X = np.array([[2,3,6,4]]) #not used really.
        boundary = ([0,0,0],[10,10,10])
        
        k = EQ(3.0, 2.0)
        windmodel=WindFixU(1)
        sensors = FixedSensorModel(X,2)
        m = AdjointAdvectionDiffusionModel(resolution=[40,4,4],boundary=boundary,windmodel=windmodel,N_feat=10000,noiseSD=5.0,kernel=k,sensormodel=sensors)

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
        """
        Checks that the integral of H (filter function) equals one over space and time. Can act as a check for H calculation and grid cell calculation.
        """
        X = np.array([[0,10,3,5]])
        y = np.array([12])#unused
        windmodel=WindFixU(1)
        boundary = ([0,0,0],[10,10,10])
        k = EQ(1.0, 2.0)

        sensors = FixedSensorModel(X,2)
        m = AdjointAdvectionDiffusionModel(resolution=[200,200,200],boundary=boundary,windmodel=windmodel,N_feat=15,noiseSD=5.0,kernel=k,sensormodel=sensors)

        volume_of_grid_tile = np.prod((np.array(boundary[1])-np.array(boundary[0]))/m.resolution)
        
        #get first/only h
        for h in sensors.getHs(m):
            break
            
        self.assertAlmostEqual(np.sum(h)*volume_of_grid_tile,1,5,"Integral of h over space and time doesn't equal one.")
       
    def test_adv_diff_forward_model(self):
        """
        Tests the calculation of the advection-diffusion PDE with a point source. Pollution distribution has a gaussian shape: http://web.mit.edu/1.061/www/dream/FIVE/FIVETHEORY.PDF
        """

        X = np.array([[0,10,3,5]])#not used
        y = np.array([12])#not used

        boundary = ([0,0,0],[20,20,20])
        k = EQ(1.0, 2.0) #not used
        sensors = FixedSensorModel(X,2)#not used
        u=[]
        u.append(np.ones([100,100,100])*0.09) #x direction wind
        u.append(np.ones([100,100,100])*0.09) # y direction wind
        windmodel=WindFixU(u)
        #given the advection and diffusion parameters, we can compute the expected Gaussian pollution after the 20s.
        m = AdjointAdvectionDiffusionModel(resolution=[100,100,100],boundary=boundary,N_feat=15,noiseSD=5.0,kernel=k,sensormodel=sensors,windmodel=windmodel,k_0=0.01)

        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = m.getGridStepSize()
        
        #impulse response of single pike of pollution, over a time period of dt.
        source = np.zeros(m.resolution)
        source[0,int(Nx/2),int(Ny/2)] = 1.0

        #estimate using our model
        estimated_concentration = m.computeConcentration(source)

        #compute the predicted analytic solution for an infinitesimal spike of pollution
        x = np.linspace(boundary[0][0],boundary[1][0],Nx)
        t = ((Nt-1)/m.resolution[0])*m.boundary[1][0]
        new_centre = (m.boundary[1][1]+dx)/2+np.max(m.u)*t
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
        
        
    def test_second_order_ODE_forward_model(self):
        """
        Tests the calculation of the second order ODE model with a constant source (=1) and fixed k_0=-0.5, u=1, eta=5. Solution distribution can be solved using variation of parameters method
        """

        tlocL = np.linspace(0,9.9,75) #not used
        X= np.zeros((len(tlocL),2)) #not used
        # Build sensor locations
        X[:,0] = tlocL #not used
        X[:,1] = X[:,0]+0.1 #not used
        sensors = FixedSensorModel(X,1) #not used
        k_0 = -0.5 #p1
        u=1 #p2
        eta=5 #p3
        noiseSD = 0.05 #not used
        N_feat=2000 # not used
        boundary = ([0],[10])# corners of the grid - in units of space
        kForward = EQ(0.6, 4.0) # generate EQ kernel arguments are lengthscale and variance
        res = [100] # grid size for time, x and y
        m = AdjointSecondOrderODEModel(resolution=res,boundary=boundary,N_feat=N_feat,noiseSD=noiseSD,kernel=kForward,sensormodel=sensors,k_0=k_0,u=u,eta=eta) #initiate ODE model to build concentration

        dt,dt2,Nt = m.getGridStepSize() # useful numbers!
        sourceGT = np.ones(m.resolution) # set constant source
        estimated_concentration=m.computeConcentration(sourceGT) # Compute concentration - runs ODE forward model

        #compute the predicted analytic solution for a constant source
        analytic = -(0.2/3)*np.exp(-m.coords)*np.sin(3*m.coords)-0.2*np.exp(-m.coords)*np.cos(3*m.coords)+0.2
        
        largest_error = np.max(np.abs(estimated_concentration[:,None]-analytic)) # calculate largest error between the estimated and analytic
        
        self.assertLess(largest_error,0.1,"Estimated concentration field does not match analytic solution")
    
    def testAdvDiffAdjoint(self):
        """
        Tests the calculation of the adjoint problem by using <c,h> = <f,v> where c=concentration field, h=filter function, f=source and v=adjoint solution
        """
        X = np.array([[18,19,19.5,19]])
        y = np.array([12])
        boundary = ([0,0,0],[20,20,20])
        k = EQ(1.0, 2.0)
        sensors = FixedSensorModel(X,1)
        u=[]
        u.append(np.ones([100,100,100])*0.01) #x direction wind
        u.append(np.ones([100,100,100])*0.01) # y direction wind
        windmodel=WindFixU(u)
        m = AdjointAdvectionDiffusionModel(resolution=[100,100,100],boundary=boundary,N_feat=150,noiseSD=5.0,kernel=k,sensormodel=sensors,windmodel=windmodel,k_0=0.005)

        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = m.getGridStepSize()
        source = np.zeros(m.resolution)
        source[1,int(Nx/2)-1,int(Ny/2)-1] = 1.0
        estimated_concentration = m.computeConcentration(source)
        v = m.computeAdjoint(list(sensors.getHs(m))[0])
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = m.getGridStepSize()
        self.assertAlmostEqual(np.sum(v*source)*dt*dx*dy,m.computeObservations()[0])
        
        
    def testSecondOrderODEAdjoint(self):
            """
            Tests the calculation of the adjoint problem by using <c,h> = <f,v> where c=concentration field, h=filter function, f=source and v=adjoint solution
            """
            tlocL=np.linspace(5,5,1)
            X= np.zeros((len(tlocL),2)) #obs matrix
            # Build sensor locations
            X[:,0] = tlocL # lower bound for obs
            X[:,1] = X[:,0]+0.1 # upper bound
            sensors = FixedSensorModel(X,1) # sensor model
            k_0 = -0.5 #p1
            u=1 #p2
            eta=5 #p3
            noiseSD = 0.05 
            N_feat=2000 
            boundary = ([0],[10])# corners of the grid - in units of space
            kForward = EQ(0.6, 4.0) # generate EQ kernel arguments are lengthscale and variance
            res = [1000] # grid size for time, x and y
            m = AdjointSecondOrderODEModel(resolution=res,boundary=boundary,N_feat=N_feat,noiseSD=noiseSD,kernel=kForward,sensormodel=sensors,k_0=k_0,u=u,eta=eta) #initiate ODE model to build concentration

            dt,dt2,Nt = m.getGridStepSize() # useful numbers!
            #source = np.ones(m.resolution) # set constant source
            z=np.random.normal(0,1.0,N_feat) # Generate z to compute source
            source=m.computeSourceFromPhi(z)# Compute ground truth source by approximating GP
            estimated_concentration=m.computeConcentration(source) # Compute concentration - runs ODE forward model

            v = m.computeAdjoint(list(sensors.getHs1D(m))[0])
            dt,dt2,Nt = m.getGridStepSize()
            self.assertAlmostEqual((np.sum(v*source)*dt)[0],np.array(m.computeObservations())[0])

    def testRegressor(self):
        """
        Checks the calculation of the regressor matrix by comparing it to results from the adjoint model and the forward model (<c,h> = <f,v>  = X.Tz) (X regressor matrix, z vector that defines the source function)
        """
        X = np.array([[17,18,10,10],[7,8,5,5],[10,15,12,15]])
        y = np.array([np.nan,np.nan,np.nan])

        boundary = ([0,0,0],[20,20,20])
        k = EQ(2, 2.0)
        sensors = FixedSensorModel(X,1)
        u=[]
        u.append(np.ones([30,30,30])*0.01) #x direction wind
        u.append(np.ones([30,30,30])*0.01) # y direction wind
        windmodel=WindFixU(u)
        m = AdjointAdvectionDiffusionModel(resolution=[30,30,30],boundary=boundary,N_feat=100,noiseSD=5.0,kernel=k,sensormodel=sensors,windmodel=windmodel,k_0=0.05)

        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = m.getGridStepSize()

        z = np.random.randn(m.N_feat,1)
        source = m.computeSourceFromPhi(z)

        m.computeConcentration(source)
        #predicted_observations = m.computeObservations()

        v = np.array([m.computeAdjoint(h) for h in sensors.getHs(m)])
        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = m.getGridStepSize()
        X1 = m.computeModelRegressors()


        calc_usingAdjoint = np.sum(v*source,(1,2,3))*dt*dx*dy
        calc_usingForwardModel = m.computeObservations()
        calc_usingRegressors = (X1.T@z)[:,0]


        self.assertLess(np.max(np.abs(calc_usingAdjoint-calc_usingForwardModel)),0.001)
        self.assertLess(np.max(np.abs(calc_usingAdjoint-calc_usingRegressors)),0.001)  
        
    def testDistribution(self):
        """
        Tests the matrix multiplication used to calculate the mean and covariance of z
        """
        X = np.array([[17,18,10,10],[7,8,5,5],[10,15,12,15]])
        y = np.array([np.nan,np.nan,np.nan])

        boundary = ([0,0,0],[20,20,20])
        k = EQ(2, 2.0)
        sensors = FixedSensorModel(X,1)
        u=[]
        u.append(np.ones([30,30,30])*0.01) #x direction wind
        u.append(np.ones([30,30,30])*0.01) # y direction wind
        windmodel=WindFixU(u)
        m = AdjointAdvectionDiffusionModel(resolution=[30,30,30],boundary=boundary,N_feat=150,noiseSD=5.0,kernel=k,sensormodel=sensors,windmodel=windmodel,k_0=0.05)
        m.X=np.identity(m.N_feat)
        y2=np.ones(m.N_feat)
        
        meanZ, covZ = m.computeZDistribution(y2)
        varTest=m.N_feat*(1+1/(m.noiseSD**2))
        meanTest=m.N_feat*(1/m.noiseSD**2)*1/(1+1/(m.noiseSD**2))
        self.assertAlmostEqual(np.sum(meanZ),meanTest)
        self.assertAlmostEqual(np.sum(np.linalg.inv(covZ)),varTest)
        
    def testSourceDistribution(self):
        """
        Tests the variance of the source calculated by computeSourceDistribution
        """
        # generate sensor locations with shape [total observations, 4], where each row has elements 
        #[lower time location, upper time location, x location, y location]

        tlocL = np.linspace(2,15,1) # lower time
        xloc=np.linspace(5,15,2) # x locations
        yloc=np.linspace(5,15,2) # y locations
        sensN = len(xloc)*len(yloc) # total number of sensors 
        obsN = len(tlocL) # total time points at which an observation is taken
        X= np.zeros((obsN*sensN,4)) # obsN*sensN is total observations over all sensors and all times
        # Build sensor locations
        X[:,0] = np.asarray(np.meshgrid(tlocL,xloc,yloc)).reshape(3,sensN*obsN)[0] 
        X[:,2] = np.asarray(np.meshgrid(tlocL,xloc,yloc)).reshape(3,sensN*obsN)[1]
        X[:,3] = np.asarray(np.meshgrid(tlocL,xloc,yloc)).reshape(3,sensN*obsN)[2]
        X[:,1] = X[:,0]+1
        
        
        k_0 = 0.0001
        noiseSD = 0.05

        N_feat=1000 # number of features used to approximate GP
        boundary = ([0,0,0],[20,20,20]) # corners of the grid
        k = EQ(4.0, 2.0) # generate EQ kernel
        sensors = FixedSensorModel(X,1) # establish sensor model
        res = [80,40,40]
        u=[]
        u.append(np.ones(res)*0.0004) #x direction wind
        u.append(np.ones(res)*0.0004) # y direction wind
        windmodel=WindFixU(u)
        m = AdjointAdvectionDiffusionModel(resolution=res,boundary=boundary,N_feat=N_feat,noiseSD=noiseSD,kernel=k,sensormodel=sensors,windmodel=windmodel,k_0=k_0) #initiate PDE model

        dt,dx,dy,dx2,dy2,Nt,Nx,Ny = m.getGridStepSize() # useful numbers!

        z=np.random.normal(0,1.0,N_feat) # Generate z to compute test source
        source=m.computeSourceFromPhi(z) # Compute test source
        conc=m.computeConcentration(source) # Compute test concentration
        y= m.computeObservations(addNoise='TRUE') # Compute observations with noise
        k = EQ(4.0, 2.0) # generate EQ kernel
        sensors = FixedSensorModel(X,1)
        N_feat = 10
        m = AdjointAdvectionDiffusionModel(resolution=res,boundary=boundary,N_feat=N_feat,noiseSD=noiseSD,kernel=k,sensormodel=sensors,windmodel=windmodel,k_0=k_0) #initiate PDE model
        X1 = m.computeModelRegressors() # Compute regressor matrix
        meanZ, covZ = m.computeZDistribution(y) # Infers z vector mean and covariance
        source2 = m.computeSourceFromPhi(meanZ) # Generates estimated source using inferred mean
        Nsamps = 500
        results = np.zeros(np.r_[res,Nsamps])
        for sample_i in range(Nsamps):
            z = np.random.multivariate_normal(meanZ,covZ)
            results[:,:,:,sample_i] = m.computeSourceFromPhi(z)
        meanSource, varSource = m.computeSourceDistribution(meanZ,covZ)
        self.assertAlmostEqual(np.sum(np.abs(np.sqrt(varSource)-np.std(results,3)))/(Nx*Ny*Nt),0,1)    
        

    
if __name__ == '__main__':
    unittest.main()
