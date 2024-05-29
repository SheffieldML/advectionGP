import numpy as np
from scipy.interpolate import interp2d

class Wind():
    def __init__(self):
        raise NotImplementedError
    
    def getwind(self,coords):
        """
        Returns the wind at given times and places, pass it Nx3 array [time,x,y].
        """
        raise NotImplementedError
        
    def getu(self,model):
        """
        Return u: a list of two matrices (one for x and one for y).
        u needs to be of the the shape: model.resolution.
        
        Requires the model object, as this can tell the method
        about the grid shape, location and resolution etc, that
        it needs to build u over.
        """
        raise NotImplementedError
    
class WindSimple(Wind):
    def __init__(self,speedx,speedy,speedz=None):
        """
        Same wind direction/speed for all time steps.
        
        speedx = Wind in North direction ("up")
        speedy = Wind in East direction ("right")
        
        This is the direction the wind is going to.
        """
        self.speedx = speedx
        self.speedy = speedy
        self.speedz = speedz

    def getwind(self,coords):
        """
        Returns the wind at given times and places, pass it [something]x3 array [time,x,y].
        
        Added hack to add 3rd axis to space: If speedz is set in constructor then it returns [time,x,y,z]
        """
        #return np.repeat(np.array([self.speedx,self.speedy])[None,:],len(coords),0)
        if self.speedz is None:
            return np.repeat(np.array([self.speedx,self.speedy])[None,:],np.prod(coords.shape[:-1]),axis=0).reshape(coords.shape)
        else:
            return np.repeat(np.array([self.speedx,self.speedy,self.speedz])[None,:],np.prod(coords.shape[:-1]),axis=0).reshape(coords.shape)

    def getu(self,model):
        u = []
        u.append(np.full(model.resolution,self.speedx)) #x direction wind
        u.append(np.full(model.resolution,self.speedy)) #y direction wind
        return u
        
class WindSimple1d(Wind):
    def __init__(self,speedx):
        """
        Same wind direction/speed for all time steps.
        
        speedx = Wind speed
        
        This is the direction the wind is going to.
        """
        self.speedx = speedx

    def getwind(self,coords):
        """
        Returns the wind at given times and places, pass it [something]x2 array [time,x].
        """
        return np.repeat(np.array([self.speedx])[None,:],np.prod(coords.shape[:-1]),axis=0).reshape(coords.shape)
       

    def getu(self,model):
        u = []
        u.append(np.full(model.resolution,self.speedx)) #x direction wind        
        return u        
        
class WindFixU(Wind):
    def __init__(self,u):
        """Class for if you need to set the exact matrices of wind.
        
        u is a list of two matrices (one for x and one for y).
        u needs to be of the the shape: model.resolution."""
        self.u = u

    def getwind(self,coords):
        """
        Returns the wind at given times and places, pass it Nx3 array [time,x,y].
        """
        raise NotImplementedError        

    def getu(self,model):
        #TODO Add exceptions/checks for shape of u.
        return self.u
        
class WindFromStations():
    def __init__(self,stationdata,time_avg):
        """
        Interpolates between locations and times of stations.
        
        stationdata is of the form:
          time, x, y, wind speed, wind direction
          
        - where x and y and time are in the units used for your model
           - ...
        - wind speed should be in the same units too (e.g. km/hour)
        - wind direction should in degrees (north = 0) [direction wind is going to]
          (angle is positive from N->E, i.e. East is +90)
        
        time_avg - how long each sample in our data averages over (e.g. 1 hour)
        """
        self.stationdata = stationdata
        self.time_avg = time_avg
        
    def getwind(self,coords):
        """
        Returns the wind at given times and places, pass it Nx3 array [time,x,y].
        """
        raise NotImplementedError 
        
        
    def getu(self,model):
        ux = []
        uy = []
        for tt in np.linspace(model.boundary[0][0],model.boundary[1][0],model.resolution[0]):
            sliceofstationdata = self.stationdata[(self.stationdata[:,0]>tt-self.time_avg) & (self.stationdata[:,0]<=tt),:]
            xvel = np.cos(np.deg2rad(sliceofstationdata[:,4]))*sliceofstationdata[:,3]
            yvel = np.sin(np.deg2rad(sliceofstationdata[:,4]))*sliceofstationdata[:,3]

            #coords = model.coords.reshape(3,np.prod(model.resolution)).T
            xx=np.linspace(model.boundary[0][1],model.boundary[1][1],model.resolution[1])
            yy=np.linspace(model.boundary[0][2],model.boundary[1][2],model.resolution[2])
            fx = interp2d(sliceofstationdata[:,2],sliceofstationdata[:,1],xvel)
            fy = interp2d(sliceofstationdata[:,2],sliceofstationdata[:,1],yvel)            
            ux.append(fx(xx,yy))
            uy.append(fy(xx,yy))
        return [np.array(ux),np.array(uy)]
