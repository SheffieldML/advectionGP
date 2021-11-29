import numpy as np

class SensorModel():
    def __init__(self):
        """Builds H"""
        assert False, "Not implemented" #TODO Turn into an exception

        
    def getHs(self):
        assert False, "Not implemented" #TODO Turn into an exception
    
class FixedSensorModel(SensorModel):
    def __init__(self,obsLocations,spatialAveraging):
        """Return a self.resolution array describing how the concentration is added up for an observation in x.
        Uses self.spatial_averaging to extend the part of the domain that is being observed.
        
        Parameters:
            x == a 4 element vector, time_start, time_end, x, y
            
        The getHs method returns a model.resolution sized numpy array
        """
        self.obsLocs = obsLocations
        self.spatialAveraging = spatialAveraging
        #TO DO
       
    def getHs(self,model):
        """
        Returns an interator providing indicator matrices for each observation.
        Should integrate to one over the `actual' space (but not necessarily over the grid).
        Arguments:
            model == is a model object (provides grid resolution etc)
            
        """
        halfGridTile = np.array([0,self.spatialAveraging/2,self.spatialAveraging/2])
        print(self.obsLocs[:,[0,2,3]]-halfGridTile)
        startOfHs = model.getGridCoord(self.obsLocs[:,[0,2,3]]-halfGridTile)
        endOfHs = model.getGridCoord(self.obsLocs[:,[1,2,3]]+halfGridTile)
        
        endOfHs[endOfHs==startOfHs]+=1 #TODO Improve this to ensure we enclose the sensor volume better with our grid.
        
        assert (np.all(startOfHs>=0)) & (np.all(startOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert (np.all(endOfHs>=0)) & (np.all(endOfHs<=model.resolution)), "Observation cell isn't inside the grid."
        assert np.all(endOfHs>startOfHs), "Observation cell has zero volume: at least one axis has no length. startOfHs:"+str(startOfHs)+" endOfHs:"+str(endOfHs)
                
        for start,end,tlength in zip(startOfHs,endOfHs,self.obsLocs[:,1]-self.obsLocs[:,0]):
            h = np.zeros(model.resolution)
            h[start[0]:end[0],start[1]:end[1],start[2]:end[2]] = 1/(self.spatialAveraging**2 * tlength)
            #h /= np.sum(h)
            print(start[0],end[0],start[1],end[1],start[2],end[2])
            yield h
        
#X = an N by 4 matrix of sensor times and locations [time_start, time_end, x, y]
#y = an N long vector of the measurements associated with the sensors.
