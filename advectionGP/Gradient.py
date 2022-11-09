import numpy as np
from advectionGP.wind import WindSimple

class SquaredErrorSamplingCost():
    def generateQSamples(mean,cov,N):
    
        samples = np.random.multivariate_normal(mean, cov,N)
        
        return samples
    
    def cost(conc,coords,obs,M,samp):
        cost = ((conc[tuple([*coords.T])]-obs)**2)*(1/M)*(1/len(samp))
        return cost
    
    def dcost(conc,coords,obs,M,samp):
        dcost=2*(conc[tuple([*coords.T])]-obs)*(1/M)*(1/len(samp))
        return dcost
        
    def costFunctionSystem(params,model,obs,obsloc,samp):
        delta, Ns = model.getGridStepSize()
        model.assignParameters(params)
        coords=model.getGridCoord(obsloc)
        c=0
        for i,sample in enumerate(samp):
            source=model.computeSourceFromPhi(sample)
            conc=model.computeConcentration(source)
            c1=np.zeros(model.resolution) # initialise cost
            M=len(obs) # number of observations
            c1[tuple([*coords.T])] = SquaredErrorSamplingCost.cost(conc,coords,obs,M,samp) # cost approximated with hill function

            c += np.sum(c1)*np.prod(delta)

        return c

    def costResponseDerivativeSystem(params,model,obs,obsloc,samp):
        model.assignParameters(params)
        delta,Ns = model.getGridStepSize()
        coords=model.getGridCoord(obsloc)
        #conc=model.computeConcentration(source)
        L_m=np.zeros(len(params))
        for i,sample in enumerate(samp):
            #print(q.shape)
            source=model.computeSourceFromPhi(sample)
            conc=model.computeConcentration(source)
            dmH=model.getSystemDerivative(conc,source)
            dc=np.zeros(model.resolution) #initialise cost derivative
            M=len(obs) # number of observations
            dc[tuple([*coords.T])] = SquaredErrorSamplingCost.dcost(conc,coords,obs,M,samp) # cost derivative approximated with step functions
            #print(np.sum((c/2)**2))
            #integrand = -model.computeGradientAdjoint(dc)*dmH
            #L_m += np.trapz(integrand,dx=delta)/len(samp)
            #L_m=np.sum(integrand)*np.prod(delta)/len(samp)

            for j, dmHi in enumerate(dmH):
                integrand = model.computeGradientAdjoint(dc)*dmHi
    #L_m = np.trapz(integrand,dx=dt)
                #L_m[j] += np.sum(integrand)*delta[0]*delta[1]*delta[2]
                L_m[j] += np.sum(integrand)*np.prod(delta)


        return L_m