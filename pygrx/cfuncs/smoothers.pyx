cimport cython
import numpy as np
cimport numpy as np

DTYPE64 = np.float64
ctypedef np.float64_t DTYPE64_t


@cython.boundscheck(False)
@cython.cdivision(True)
def ninepoint(np.ndarray[DTYPE64_t, ndim=2] field, DTYPE64_t p,
              DTYPE64_t q, DTYPE64_t mask, int wrap=0):
    """
    ***********************************************************************
     modified j. olson routine
    ***********************************************************************

     this routine does 9-point smoothing using the equation:

       f0 = f0 + (p/4)*(f2+f4+f6+f8-4*f0) + (q/4)*(f1+f3+f5+f7-4*f0)

      where the grid is:

          1-------------8-------------7
          |             |             |
          |             |             |
          |             |             |
          |             |             |
          2-------------0-------------6
          |             |             |
          |             |             |
          |             |             |
          |             |             |
          3-------------4-------------5

        arguments (inputs):
     .
     .   field    - 2-d input array
     .   p        - first  weight (suggested value of  0.50)
     .   q        - second weight (suggested value of  0.25)
     .   mask     - value of missing points
     .   wrap     - logical flag to include wraparound points in smoothing
     .              if wrap = 1, smooth left and right endpoints
     .              if wrap = 0, no smoothing
     .
        arguments (output):
     .   tmp      - 2-d output array
     .
      notes:

           1)  if a point or any of its neighbors is missing, the point is
               not smoothed
           2)  this routine does not smooth the edges, just the interior
               with the exception that the left and right edges are smoothed
               when "lwrap" is true.
           3)  array has to have at least 3 points in each direction
           4)  smoothed results are returned via the original input array

    ***********************************************************************
    """

    cdef Py_ssize_t i, j
    cdef Py_ssize_t jm1, jp1, im1, ip1
    cdef int nx, ny, nxb, nxe, nyb, nye
    cdef float po4, qo4
    cdef float term1, term2
    cdef np.ndarray[DTYPE64_t, ndim=2] tmp

    po4 = p/4.
    qo4 = q/4.
    nx = field.shape[0]
    ny = field.shape[1]
    tmp = np.ones((nx, ny)) * mask

    if wrap == 1:
        nxb = 0
        nxe = nx-1
    else:
        nxb = 1
        nxe = nx-2
    nyb = 2
    nye = ny - 1

    for j in range(nyb, nye):
        for i in range(nxb, nxe):
            jm1 = j-1
            jp1 = j+1
            im1 = i-1
            ip1 = i+1

            if im1 < 0:
                im1 = nx
            if ip1 > nx:
                ip1 = 1

            if (field[i,j] == mask or field[im1,jp1] == mask or
                field[im1,j] == mask or field[im1,jm1] == mask or
                field[i,jm1] == mask or field[ip1,jm1] == mask or
                field[ip1,j] == mask or field[ip1,jp1] == mask or
                field[i,jp1] == mask):
                    tmp[i,j] = field[i,j]
            else:
                term1 = po4 * (field[im1,j]+field[i,jm1]+
                               field[ip1,j] + field[i,jp1] - (4. * field[i,j]))
                term2 = qo4 * (field[im1,jp1]+field[im1,jm1]+
                               field[ip1,jm1]+field[ip1,jp1] - (4 * field[i,j]))
                tmp[i,j] = field[i,j] + term1 + term2

    return tmp