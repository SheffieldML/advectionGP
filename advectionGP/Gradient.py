import numpy as np


class SquaredErrorSamplingCost():

    def generateQSamples(mean,cov,N):
    
        samples = np.random.multivariate_normal(mean, cov,N)
        
        return samples
        
    def costFunction(params,model,obs,tloc,samp):
        delta, Ns = model.getGridStepSize()
        model.k_0=params[2]
        model.u=params[0]
        model.eta=params[1]
        c=0
        for i,sample in enumerate(samp):
            source=model.computeSourceFromPhi(sample)
            conc=model.computeResponse(source)

            c1=np.zeros(model.resolution) # initialise cost
            M=len(obs) # number of observations
            #c1[model.getGridCoord(tloc)] = ((conc[model.getGridCoord(tloc)]-obs)**2)*(1/M) # cost approximated with hill function
            c1[tuple(map(tuple,model.getGridCoord(tloc).T))] = ((conc[tuple(map(tuple,model.getGridCoord(tloc).T))]-obs)**2)*(1/M)

            c += np.sum(c1)*delta/len(samp)

        return c

    def costResponseDerivative(params,model,obs,tloc,samp):
        model.k_0=params[2]
        model.u=params[0]
        model.eta=params[1]
        delta,Ns = model.getGridStepSize()
        #conc=model.computeConcentration(source)
        L_m=0
        for i,sample in enumerate(samp):
            #print(q.shape)
            source=model.computeSourceFromPhi(sample)
            conc=model.computeResponse(source)
            dmH=model.getSystemDerivative(conc,source)
            dc=np.zeros(model.resolution) #initialise cost derivative
            M=len(obs) # number of observations
            dc[model.getGridCoord(tloc)] = 2*(conc[model.getGridCoord(tloc)]-obs)*(1/M) # cost derivative approximated with step functions
            #print(np.sum((c/2)**2))
            integrand = -model.computeGradientAdjoint(dc)*dmH
            L_m += np.trapz(integrand,dx=delta)/len(samp)
        #L_m = np.sum(integrand)*dt
        return L_m