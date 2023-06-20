import numpy as np
from advectionGP.wind import WindSimple

class SquaredErrorSamplingCost():
    def generateQSampleLocations(model,nSamp):
        samples = np.random.randn(nSamp,model.N_feat)
        
        return samples
    def generateQSamples(obs,sample,model):
        meanZ, covZ = model.computeZDistribution(obs)
        #X = np.random.randn(nSamp,model.N_feat)
        chol = np.linalg.cholesky(covZ)
        samp = (meanZ[:,None] + chol @ sample.T).T

        return samp
    
    def cost(conc,coords,obs,M,S):
        cost = ((conc[tuple([*coords.T])]-obs)**2)*(1/M)*(1/S)
        return cost
    
    def dcost(conc,coords,obs,M,S):
        dcost=2*(conc[tuple([*coords.T])]-obs)*(1/M)*(1/S)
        
        return dcost
        
    def costFunctionSystem(params,model,obs,obsloc,sample):
        delta, Ns = model.getGridStepSize()
        coords=model.getGridCoord(obsloc)
        samp = SquaredErrorSamplingCost.generateQSamples(obs,sample,model)
        model.assignParameters(params)
        #model.computeModelRegressors()
        
        S = len(sample)
        M=len(obs) # number of observations
        c=0
        for i,samples in enumerate(samp):
            source=model.computeSourceFromPhi(samples)
            conc=model.computeResponse(source)
            c1=np.zeros(model.resolution) # initialise cost
            c1[tuple([*coords.T])] = SquaredErrorSamplingCost.cost(conc,coords,obs,M,S) # cost approximated with hill function

            c += np.sum(c1)*np.prod(delta)

        return c

    def costResponseDerivativeSystem(params,model,obs,obsloc,sample):
        samp = SquaredErrorSamplingCost.generateQSamples(obs,sample,model)
        model.assignParameters(params)
        #model.computeModelRegressors()
        delta,Ns = model.getGridStepSize()
        coords=model.getGridCoord(obsloc)
        
        #conc=model.computeConcentration(source)
        
        
        S = len(sample)
        M=len(obs) # number of observations
        L_m=np.zeros(len(params))
        for i,samples in enumerate(samp):
            #print(q.shape)
            source=model.computeSourceFromPhi(samples)
            conc=model.computeResponse(source)
            dmH=model.getSystemDerivative(conc,source)
            dc=np.zeros(model.resolution) #initialise cost derivative
            
            dc[tuple([*coords.T])] = SquaredErrorSamplingCost.dcost(conc,coords,obs,M,S) # cost derivative approximated with step functions
            #print(np.sum((c/2)**2))
            #integrand = -model.computeGradientAdjoint(dc)*dmH
            #L_m += np.trapz(integrand,dx=delta)/len(samp)
            #L_m=np.sum(integrand)*np.prod(delta)/len(samp)

            for j, dmHi in enumerate(dmH):
                integrand = model.computeAdjoint(-dc)*dmHi
    #L_m = np.trapz(integrand,dx=dt)
                #L_m[j] += np.sum(integrand)*delta[0]*delta[1]*delta[2]
                L_m[j] += np.sum(integrand)*np.prod(delta)


        return L_m
    
    
    def costFunctionLengthscale(lengthscale,model,obs,obsloc,sample):
        delta, Ns = model.getGridStepSize()
        model.kernel.l2=lengthscale
        coords=model.getGridCoord(obsloc)
        model.computeModelRegressors() # Compute regressor matrix 
        samp = SquaredErrorSamplingCost.generateQSamples(obs,sample,model)
        S = len(sample)
        M=len(obs) # number of observations
        c=0
        for i,samples in enumerate(samp):
            source=model.computeSourceFromPhi(samples)
            conc=model.computeResponse(source)
            c1=np.zeros(model.resolution) # initialise cost
            c1[tuple([*coords.T])] = SquaredErrorSamplingCost.cost(conc,coords,obs,M,S) # cost approximated with hill function

            c += np.sum(c1)*np.prod(delta)

        return c

    def costResponseDerivativeLengthscale(lengthscale,model,obs,obsloc,sample):
        model.kernel.l2=lengthscale
        delta,Ns = model.getGridStepSize()
        coords=model.getGridCoord(obsloc)
        model.computeModelRegressors() # Compute regressor matrix 
        samp = SquaredErrorSamplingCost.generateQSamples(obs,sample,model)
        M=len(obs) # number of observations
        S = len(sample)
        #conc=model.computeConcentration(source)
        L_m=0
        c=0
        for i,samples in enumerate(samp):
            #print(q.shape)
            source=model.computeSourceFromPhi(samples)
            conc=model.computeResponse(source)
            dmH=model.getSourceLengthscaleDerivative(samples,obs,sample[i])
            dc=np.zeros(model.resolution) #initialise cost derivative
            dc[tuple([*coords.T])] = SquaredErrorSamplingCost.dcost(conc,coords,obs,M,S) # cost derivative approximated with step function
            integrand = model.computeAdjoint(-dc)*dmH
            L_m += np.sum(integrand)*np.prod(delta)
        
        return L_m