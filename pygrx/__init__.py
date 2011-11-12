import version
from cfuncs import *

__version__ = version.get_version()


import numpy as np
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap

def get_map(resolution='c', projection='lcc', llcrnrlon=-110.5, llcrnrlat=23,
            urcrnrlon=-62.5, urcrnrlat=47.5, lat_1=30, lat_2=60, lon_0=-92.5,
            area_thresh=2000):
    """ Creates a basemap instance used for contouring data """
    m = Basemap(resolution=resolution, projection=projection,
                llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
                urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat,
                lat_1=lat_1, lat_2=lat_2, lon_0=lon_0,
                area_thresh=area_thresh)
    return m


def get_vertices(m, x, y, field, levels):
    """ Determines the vertices of a contour field

        m : basemap instance
            Used for the map projection

        x : list or numpy array
            The longitudes of the data grid points

        y : list or numpy array
            The latitudes of the data grid points

        field : list or numpy array
            The values of the field to contour

        min : int or float
            Minimum contour value

        max : int or float
            Maximum contour value (included)

        cint : int or float
            Contour interval to use

        Returns : tuple
            The tuple returns has a length determined by the amount of data
            and interval that was contoured.

            Element 0 is the value of what was contoured
            Element 1 are the longitude, latitude pairs of what was contoured
    """
    data_obj = []
    x, y = m(x, y)
    col = m.contour(x, y, field, levels).collections
    for i in range(len(levels)):
        for coord in col[i].get_paths():    # GET THE PATHS FOR THE FIRST CONTOUR
            points = []
            for xy in coord.vertices:       # ITERATE OVER THE PATH OBJECTS
                xtmp, ytmp = m(xy[0], xy[1], inverse=True)
                points.append((xtmp, ytmp))
            data_obj.append((levels[i], points))

    return data_obj
