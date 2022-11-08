import numpy as np
from advectionGP.wind import WindSimple

class SquaredErrorSamplingCost():

    def generateQSamples(mean,cov,N):
    
        samples = np.random.multivariate_normal(mean, cov,N)
        
        return samples
        
    def costFunction(params,model,obs,obsloc,samp,mType):
        delta, Ns = model.getGridStepSize()
        if mType=="Oscillator":
            model.k_0=params[2]
            model.u=params[0]
            model.eta=params[1]
        if mType == "ADR":
            model.windmodel=WindSimple(params[0],params[1])
            model.u=model.windmodel.getu(model)
            model.k_0=params[2]
            model.R=params[3]
        else:
            print("No valid model type specified")
            pass
        coords=model.getGridCoord(obsloc)
        c=0
        for i,sample in enumerate(samp):
            source=model.computeSourceFromPhi(sample)
            if mType=="Oscillator":
                conc=model.computeResponse(source)
            if mType=="ADR":
                conc=model.computeConcentration(source)

            c1=np.zeros(model.resolution) # initialise cost
            M=len(obs) # number of observations
            c1[tuple([*coords.T])] = ((conc[tuple([*coords.T])]-obs)**2)*(1/M)*(1/len(samp)) # cost approximated with hill function

            c += np.sum(c1)*np.prod(delta)

        return c

    def costResponseDerivative(params,model,obs,obsloc,samp,mType):
        if mType=="Oscillator":
            model.k_0=params[2]
            model.u=params[0]
            model.eta=params[1]
        if mType == "ADR":
            model.windmodel=WindSimple(params[0],params[1])
            model.u=model.windmodel.getu(model)
            model.k_0=params[2]
            model.R=params[3]
        else:
            print("No valid model type specified")
            pass
        delta,Ns = model.getGridStepSize()
        coords=model.getGridCoord(obsloc)
        #conc=model.computeConcentration(source)
        L_m=np.zeros(len(params))
        for i,sample in enumerate(samp):
            #print(q.shape)
            source=model.computeSourceFromPhi(sample)
            if mType=="Oscillator":
                conc=model.computeResponse(source)
            if mType=="ADR":
                conc=model.computeConcentration(source)
            dmH=model.getSystemDerivative(conc,source)
            dc=np.zeros(model.resolution) #initialise cost derivative
            M=len(obs) # number of observations
            dc[tuple([*coords.T])] = 2*(conc[tuple([*coords.T])]-obs)*(1/M)*(1/len(samp)) # cost derivative approximated with step functions
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