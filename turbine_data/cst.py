# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

"""
Class Shape Transform (CST) utilities
"""

import numpy as np
import scipy.optimize as sopt
from scipy.special import comb
from scipy.interpolate import PchipInterpolator


class CST2D:
    """CST representation for 2-D shapes"""

    #: Mapping of the ``N1`` and ``N2`` coefficients for various shapes
    shape_class = {
        'airfoil' : (0.5, 1.0),
        'ellipse' : (0.5, 0.5),
        'biconvex': (1.0, 1.0),
        'sears_hack': (0.75, 0.75),
        'projectile': (0.75, 0.25),
        'cone' : (1.0, 0.001),
        'rectangle' : (0.001, 0.001)
    }

    def __init__(self, order=8):
        """
        Args:
            order (int): Bernstein polynomial order
        """
        #: Bernstein polynomial order of the CST parametrization
        self.order = order
        #: Polynomial coefficients
        self.kvals = comb(order, range(order + 1))
        # Assume default is airfoil
        self.n1, self.n2 = self.shape_class["airfoil"]

    def cls_fcn(self, xco):
        """Class function for a given psi = (x/c)

        Return:
            np.ndarray: Array containing class functions with length of xinp
        """
        return np.power(xco, self.n1) * np.power((1.0 - xco), self.n2)

    def shape_fcn(self, xinp):
        """Shape functions for a given psi = (x/c)

        Return:
            np.ndarray: ``[BP+1, len(xinp)]`` array of shape function
        """
        K = self.kvals
        N = self.order
        xco = np.asarray(xinp)

        stmp = np.empty((N+1, xco.shape[0]))
        for i in range(N + 1):
            stmp[i, :] = K[i] * np.power(xco, i) * np.power((1.0 - xco), (N-i))
        return stmp

    def cst_matrix(self, xinp):
        """Return product of C(psi) * S_i(psi)"""
        return (self.cls_fcn(xinp) * self.shape_fcn(xinp)).T

    def solve(self, xco, yco, yte=0.0):
        """Solve the least squares problem for a given shape

        Args:
            xco (np.ndarray): (x/c) coordinates locations
            yco (np.ndarray): (y/c) coordinate locations
            yte (double): Trailing edge thickness

        Return:
            np.ndarray: ``(BP+1)`` CST parameters
        """
        amat = self.cst_matrix(xco)
        bvec = yco - xco * yte
        out = sopt.lsq_linear(amat, bvec)
        return out



class CSTAirfoil(CST2D):
    """Concrete implementation of CST for airfoils"""

    def __init__(self, airfoil, order=8, shape_class='airfoil'):
        """
        Args:
            airfoil (AirfoilShape): Airfoil geometry information
            order (int): Polynomial order
            shape_class (string): Shape class
        """
        super().__init__(order)
        self.n1, self.n2 = self.shape_class[shape_class]
        #: Instance of :class:`~uaero_ml.aero.airfoil.AirfoilShape`
        self.airfoil = airfoil

    @classmethod
    def from_cst_parameters(cls, airfoil, cst_lower, cst_upper,
                            n1=0.5, n2=1.0):
        """
        Args:
            airfoil (AirfoilShape): Airfoil geometry information
            cst_lower: Lower surface CST coefficients
            cst_upper: Upper surface CST coefficients
        """
        order = np.size(cst_lower)-1
        self = CSTAirfoil(airfoil, order)
        self.n1 = n1
        self.n2 = n2
        self._cst_lower = cst_lower
        self._cst_upper = cst_upper
        self._cst = np.r_[self._cst_lower, self._cst_upper]
        return self

    def _compute_cst(self):
        """Compute CST coefficients for airfoil on demand"""
        af = self.airfoil
        out1 = self.solve(af.xupper, af.yupper, af.te_upper)
        self._cst_upper = out1.x
        out2 = self.solve(af.xlower, af.ylower, af.te_lower)
        self._cst_lower = out2.x
        self._cst = np.r_[self._cst_lower, self._cst_upper]
        return (out1, out2)

    @property
    def cst(self):
        """CST coefficients for the airfoil

        Returns an array of size (2*(BP+1)) containing the coefficients for the
        lower surface followed by the upper surface.
        """
        if not hasattr(self, "_cst"):
            self._compute_cst()
        return self._cst

    @property
    def cst_upper(self):
        """CST coefficients for the upper (suction) side of the airfoil

        Return:
            np.ndarray: Array of length (BP+1)
        """
        if not hasattr(self, "_cst"):
            self._compute_cst()
        return self._cst_upper

    @property
    def cst_lower(self):
        """CST coefficients for the lower (pressure) side of the airfoil

        Return:
            np.ndarray: Array of length (BP+1)
        """
        if not hasattr(self, "_cst"):
            self._compute_cst()
        return self._cst_lower

    def __call__(self, xinp, p_ar=None, te_upper=None, te_lower=None):
        """Compute coordinates for the airfoil

        Args:
            xinp (np.ndarray): Non-dimensional x-coordinate locations
            p_ar (np.ndarray): Non-dimensional perturbation of cst coefficients
            te_upper (double): Trailing edge thickness above camber line
            te_lower (double): Trailing edge thickness below camber line

        Return:
            tuple: (ylo, yup) Numpy arrays for the lower, upper y-coordinates
        """
        xco = np.asarray(xinp)
        telo = self.airfoil.te_lower if te_lower is None else te_lower
        teup = self.airfoil.te_upper if te_upper is None else te_upper
        amat = self.cst_matrix(xinp)
        cst_lower = self.cst_lower
        cst_upper = self.cst_upper
        if p_ar is not None:
            cst_lower = self.cst_lower * (1.0 + p_ar[:(1+self.order)])
            cst_upper = self.cst_upper * (1.0 + p_ar[(1+self.order):])
        ylo = np.dot(amat, cst_lower) + telo * xco
        yup = np.dot(amat, cst_upper) + teup * xco
        return (ylo, yup)

class AirfoilShape:
    """Representation of airfoil point data"""

    def __init__(self, xco, yco, shape_class='airfoil'):
        """
        Args:
            xco (np.ndarray): Array of x-coordinates
            yco (np.ndarray): Array of y-coordinates
            shape_class (string): Airfoil Shape type
        """
        xlo = np.min(xco)
        xhi = np.max(xco)

        #: Chord length based on input data
        self.chord = (xhi - xlo)
        #: Normalized x-coordinate array
        self.xco = (xco - xlo) / self.chord
        #: Normalized y-coordinate array
        self.yco = yco / self.chord

        # Leading edge index
        le_idx = np.argmin(self.xco)
        # Determine orientation of the airfoil shape
        y1avg = np.average(self.yco[:le_idx])
        # Flip such that the pressure side is always first
        if y1avg > 0.0:
            self.xco = self.xco[::-1]
            self.yco = self.yco[::-1]

        self._le = np.argmin(self.xco)

        self.shape_class = shape_class

    @classmethod
    def from_cst_parameters(cls, cst_lower, te_lower, cst_upper, te_upper,
                            n1=0.5, n2=1.0):
        """Create airfoil from CST parameters
        Args:
            cst_lower (np.ndarray): Array of lower surface CST parameters
            cst_upper (np.ndarray): Array of upper surface CST parameters
            te_lower (double): Lower surface trailing edge y coordinate
            te_upper (double): Upper surface trailing edge y coordinate
            n1 (double): N1 parameter for CST
            n2 (double): N2 parameter for CST
        """
        ccst = CSTAirfoil.from_cst_parameters(cls,cst_lower,cst_upper,n1,n2)
        x_c = -np.cos(np.arange(0,np.pi+0.005,np.pi*0.005))*0.5+0.5
        yl,yu = ccst(x_c, te_upper=te_upper, te_lower=te_lower)
        xco = np.append(x_c[::-1],x_c[1:])
        yco = np.append(yl[::-1],yu[1:])
        self = AirfoilShape(xco,yco)
        self._cst = ccst
        return self

    @classmethod
    def from_txt_file(cls, coords_file):
        """Load airfoil from a text file"""
        fpath = Path(coords_file).resolve()
        assert fpath.exists()
        xco, yco = np.loadtxt(fpath, unpack=True)
        self = AirfoilShape(xco, yco)
        return self


    @property
    def xupper(self):
        """Coordinates of the suction side"""
        return self.xco[self._le:]

    @property
    def yupper(self):
        """Coordinates of the suction side"""
        return self.yco[self._le:]

    @property
    def xlower(self):
        """Coordinates of the pressure side"""
        return self.xco[:self._le+1]

    @property
    def ylower(self):
        """Coordinates of the pressure side"""
        return self.yco[:self._le+1]

    @property
    def te_upper(self):
        """Trailing edge thickness on suction side"""
        return self.yco[-1]

    @property
    def te_lower(self):
        """Trailing edge thickness on pressure side"""
        return self.yco[0]

    def cst(self, order=8):
        """Return CST representation of the airfoil"""
        if not hasattr(self, "_cst"):
            self._cst = CSTAirfoil(self, order, self.shape_class)
        return self._cst

    def n1(self):
        """Return n1 parameter for airfoil"""
        if not hasattr(self, "_cst"):
            self._cst = CSTAirfoil(self, order, self.shape_class)
        return self._cst.n1

    def n2(self):
        """Return n2 parameter for airfoil"""
        if not hasattr(self, "_cst"):
            self._cst = CSTAirfoil(self, order, self.shape_class)
        return self._cst.n2


    def __call__(self, xinp):
        """Return interpolated y-coordinates for an airfoil

        Args:
            xinp (np.ndarray): Non-dimensional x-coordinate locations

        Return:
            tuple: (xco, ylo, yup) Dimensional (lower, upper) y-coordinates
        """
        afcst = self.cst()
        (ylo, yup) = afcst(xinp, None, te_upper=self.te_lower, te_lower=self.te_lower)
        return (xinp * self.chord, ylo * self.chord, yup * self.chord)

    def perturb(self, xinp, p_ar):
        """ Return perturbed y-coordinates for an airfoil by perturbing
        the cst coefficients

        Args:
            xinp (np.ndarray): Non-dimensional x-coodinate locations
            p_ar (np.ndarray): Non-dimensional perturbation

        Return:
            tuple: (xco, ylo, yup) Dimensional (lower, upper) y-coordinates
        """

        afcst = self.cst()
        (ylo, yup)= afcst(xinp, p_ar)
        return (xinp * self.chord, ylo * self.chord, yup * self.chord)



class CSTBlade():
    """Concrete implementation of CST for blades"""

    def __init__(self, blade, order=8):
        """
        Args:
            blade (BladeShape): Blade geometry information
            order (int): Bernstein polynomial order
        """
        self.blade = blade
        #: Bernstein polynomial order of the CST parametrization
        self.order = order
        #: Polynomial coefficients
        self.kvals = comb(order, range(order + 1))

    def shape_fcn(self, etainp):
        """Shape functions for a given eta = (y/L)

        Return:
            np.ndarray: ``[BP+1, len(xinp)]`` array of shape function
        """
        K = self.kvals
        N = self.order
        eta = np.asarray(etainp)
        stmp = np.empty((N+1, eta.size))
        for i in range(N + 1):
            stmp[i, :] = K[i] * np.power(eta, i) * np.power((1.0 - eta), (N-i))
        return stmp


    def cst_matrix(self, xinp):
        """Return S_i(psi)"""
        return (self.shape_fcn(xinp)).T

    def solve(self, eta, ai):
        """Solve the least squares problem for a given set of cst coefficients
        corresponding to the shape function along the span

        Args:
            eta (np.ndarray): (\eta/L) coordinates locations along the span
            ai (np.ndarray): CST coefficients for airfoils along the span

        Return:
            np.ndarray: ``(BP+1)`` CST parameters

        """
        amat = self.cst_matrix(eta)
        out = sopt.lsq_linear(amat, ai)
        return out

    def _compute_cst3d(self):
        """ Compute the CST coefficients along the span

        Return:

        """
        blade = self.blade
        self._cst3d = np.zeros(( (self.order+1), 2*(blade._cst2d_order+1)  ))
        amat = self.cst_matrix(blade._eta)
        for i in range( 2*(blade._cst2d_order+1) ):
            self._cst3d[:,i] = self.solve(blade._eta, blade._cst2d[:, i]).x

    @property
    def cst3d(self):
        """CST coefficients for the blade

        Returns an array of size (BP+1) containing the coefficients
        """
        if not hasattr(self, "_cst3d"):
            self._compute_cst3d()
        return self._cst3d

    def __call__(self, etainp, p_ar=None):
        """Compute CST coefficients representing cross section at etainp

        Args:
            etainp (np.ndarray): Non-dimensional spanwise coordinate locations
            p_ar (np.ndarray): Non-dimensional perturbation of cst coefficients

        Return:
           cs2d (np.ndarray):  Numpy array of CST coefficients representing
                               cross section at eta
        """
        eta = np.asarray(etainp)
        amat = self.cst_matrix(eta)
        cst3d = self.cst3d
        cst2d = np.zeros(2*(self.blade._cst2d_order+1))
        for i in range(2*(self.blade._cst2d_order+1)):
            cst2d[i] = np.dot(amat, cst3d[:,i] )
        return cst2d

class BladeShape():

    def __init__(self, eta, cst_2d, order=8, eta_cmax=0.3):
        """Initialize the blade shape in the spanwise direction using CST
        coefficients at different cross sections

        Args:
            eta (np.ndarray): non-dim coordinate along the span
            cst_2d (np.ndarray): Array of CST coefficients for 2D cross sections

        """
        self._cst3d_order = order

        #Determine order of Bernstein polynomials for CST
        #representation of cross-sections
        self._cst2d_order = int((cst_2d.shape[1]-6) / 2)
        self._cst2d = cst_2d
        self._eta = eta

        print("Removing CST for blade shape definition. Continue to use CST for airfoil definition and linear interpolation for the blade shape.")

    def cst3d(self, order=8):
        """Return CST representation of the blade"""
        if not hasattr(self, "_cst3d"):
            self._cst3d = CSTBlade(self, order)
        return self._cst3d


    def interp_tanh(self, xinp, x_d, y_d):
        """Interpolate to x using a tanh function
        Args:

            xinp (np.ndarray): Array or individual location at which the
                            interpolated y value is desired
            x_d (np.ndarray): Array of input x values for interpolation
            y_d (np.ndarray): Array of input y values for interpolation

        Return:
            yout (np.ndarray): Array of output interpolated values at xinp

        """
        ixloc = np.searchsorted(x_d, xinp)
        x_d = np.asarray(x_d)
        y_d = np.asarray(y_d)
        if (ixloc == 0):
            yout = y_d[0]
        elif(ixloc == x_d.size):
            yout = y_d[-1]
        else:
            x1 = x_d[ixloc - 1]
            x2 = x_d[ixloc]
            y1 = y_d[ixloc - 1]
            y2 = y_d[ixloc]
            interp_fac = (np.tanh( 5.0* ((xinp-x1)/(x2-x1) * 2.0 - 1.0) ) + 1.0) * 0.5
            yout = y1 + interp_fac * (y2 - y1)
        return yout


    def __call__(self, eta, p_ar=None):
        """Compute 2D shape at a given spanwise location
        """

        # Bound eta in [0,1]
        eta = np.minimum( np.maximum(0.0, eta), 1.0)

        BPpsi = self._cst2d_order
        # Use CST representation to interpolate the shape function coefficients
        cst3d = self.cst3d()
        cst2d = np.zeros( 2*(BPpsi + 2) + 2)
        #cst2d[:2*(BPpsi + 1)] = cst3d(eta)
        for i in range(2*(BPpsi + 1)):
            cst2d[i] = np.interp(eta, self._eta, self._cst2d[:,i])

        # Manually interpolate the rest
        # Trailing edge thickness using  linear interpolation
        for i in range(2*(BPpsi + 1), 2*(BPpsi + 2)):
            cst2d[i] = np.interp(eta, self._eta, self._cst2d[:,i])
        #Interpolate class function exponents N1 and N2 using tanh interpolation
        for i in range(2*(BPpsi + 2), 2*(BPpsi + 3)):
            cst2d[i] = self.interp_tanh(eta, self._eta, self._cst2d[:,i])

        ccst = AirfoilShape.from_cst_parameters(cst2d[0:BPpsi+1], cst2d[2*(BPpsi+1)], cst2d[BPpsi+1:2*(BPpsi+1)],  cst2d[2*(BPpsi+1)+1], cst2d[-2], cst2d[-1])
        return [ccst.xco, ccst.yco]


# class GrassmanBladeShape():
#
#     def __init__(self, eta, xy):
#         self.eta_inp = eta
#         self.xy = xy
#
#     def __call__(self, eta):
#         """Compute 2D shape at a given spanwise location """
#
#         # Bound eta in [0,1]
#         eta = np.minimum( np.maximum(0.0, eta), 1.0)
#
#
#         # Do simple linear interpolation of shapes for now
#         new_x = self.xy[:,0,0] #It's the same x at all eta locations
#         new_y = np.array([ np.interp(eta, self.eta_inp, self.xy[i,1,:])
#                            for i in range(401) ])
#         return [new_x, new_y]

class GrassmanBladeShape():

    def __init__(self, eta_nominal, xy):
        """
        :param eta_nominal: list of nominal eta from 0 to 1
        :param xy: (n, 2, n_shapes) arrays with shape coordinates in physical space
        """
        xy = np.rollaxis(xy, 2, 0)      # convert to ndarray with shape (n_shapes, n, 2)
        # sort based on increasing span
        self.xy_nominal = xy[np.argsort(eta_nominal)]
        self.eta_nominal = np.sort(eta_nominal)
        self.n_shapes, self.n_landmarks, _ = self.xy_nominal.shape

        # preallocate affine transformations
        self.xy_grassmann = np.empty_like(self.xy_nominal)
        self.M = np.empty((self.n_shapes, 2, 2))
        self.b = np.empty((self.n_shapes, 2))
        # LA transformation (calculate self.xy_grassmann, self.scale, self.scale_inv, self.shift)
        self.landmark_affine_transform(self.xy_nominal)

        self.dist_grassmann = self.calc_grassmann_distance(self.xy_grassmann)
        self.log_map = self.calc_log_mapping(self.xy_grassmann)

        self.t_nominal = np.cumsum(self.dist_grassmann)
        # PCHIP 1-D monotonic cubic interpolation
        self.interpolator = PchipInterpolator(self.eta_nominal, self.t_nominal)


    def landmark_affine_transform(self, xy_list):
        """
        Shift and scale all shapes using Landmark-Affine standardization (Bryner, 2D affine and projective spaces)
        :param xy_list: list of (n, 2) arrays of physical coordinates of shapes
        :return:
        """
        for i, xy in enumerate(xy_list):
            center_mass = np.mean(xy, axis=0)
            U, D, _ = np.linalg.svd(1 / np.sqrt(self.n_landmarks - 1) * (xy - center_mass).T)
            self.M[i] = np.diag(1. / D) @ U.T
            self.b[i] = -self.M[i] @ center_mass
            xy_transformed = xy @ self.M[i].T + self.b[i]
            self.xy_grassmann[i] = 1 / np.sqrt(self.n_landmarks - 1) * xy_transformed
        return

    def calc_grassmann_distance(self, xy_grassmann):
        """
        Calculate distance on grassmannian
        :param xy_grassmann: list of (n, 2) arrays
        :return: list of distances on grassmannian
        """
        dist = np.zeros(len(xy_grassmann))
        for i in range(1, len(xy_grassmann)):
            dist[i] = self.distance(xy_grassmann[i], xy_grassmann[i - 1])
        return dist

    def calc_log_mapping(self, xy_grassmann):
        """
        Calculate log mapping (the tangent direction from X1 to X2 )
        :return: array of directions
        """
        n, m, d = xy_grassmann.shape
        log_map = np.empty((n - 1, m, d))
        for i in range(n-1):
            log_map[i] = self.log(xy_grassmann[i], xy_grassmann[i+1])  # compute direction
        return log_map

    def sample_eta(self, n_samples):
        """
        Sample eta. so corresponding t is uniformly distributed
        :param n_samples: number of sample to return
        :return: array of eta samples
        """
        t_span = np.linspace(0, self.t_nominal[-1], n_samples)
        interpolator_inv = PchipInterpolator(self.t_nominal, self.eta_nominal)
        eta_span = interpolator_inv(t_span)
        return eta_span

    @staticmethod
    def exp(t, X, log_map):
        """
        Exponential mapping (Grassmannian geodesic)
        :param t: parameter that controls the location on geodesic (scalar from [0, 1])
        :param X: starting point of geodesic in Grassmannian
        :param log_map: direction (tangent vector \Delta)
        :return:
        """
        U, S, V = np.linalg.svd(log_map, full_matrices=False)
        exp_map = np.hstack((X @ V, U)) @ np.vstack((np.diag(np.cos(t * S)), np.diag(np.sin(t * S)))) @ V.T
        return exp_map

    @staticmethod
    def log(X, Y):
        """
        Calculate logarithmic map log_X(Y) (inverse mapping of exponential map).
        Calculates direction(tangent vector \Delta) from X to Y in tangent subspace.
        :param X: starting point of geodesic on Gr
        :param Y: end point of geodesic on Gr
        :return: direction (tangent vector \Delta)
        """
        ortho_projection = np.eye(len(X)) - X @ X.T
        Delta = ortho_projection @ Y @ np.linalg.inv(X.T @ Y)
        U, S, V = np.linalg.svd(Delta, full_matrices=False)
        log_map = U @ np.diag(np.arctan(S)) @ V.T
        return log_map

    @staticmethod
    def distance(X, Y):
        """
        Geodesic distance on Grassmannian is defined by the principal angles
        between the subspaces spanned by the columns of X1 and X2, denoted by
        span(X) and span(Y). The cosines of the principal angles theta1 and
        theta2 between span(X) and snap(Y) are the singular values of X.T@Y.
        That is, X.T@Y = U D V.T,  where D = diag(cos(theta1), cos(theta2)).
        The distance between two shapes is then defined as dist = sqrt(theta1**2+theta2**2)
        :param X: first shape
        :param Y: second shape
        :return: distance between two shapes on Grassmannian
        """
        dim = X.shape[1]  # get dimensions(must be identical for Y)
        if X.shape != Y.shape:
            raise ValueError('Input matrices must have the same number of columns')
        if not np.allclose(X.T @ X, np.eye(dim)):
            raise ValueError('First input does not constitute an element of the Grassmannian')
        if not np.allclose(Y.T @ Y, np.eye(dim)):
            raise ValueError('Second input does not constitute an element of the Grassmannian')

        D = np.linalg.svd(X.T @ Y, compute_uv=False)  # compute sungular values

        ind_of_ones = np.array(
            [i for i in range(dim) if np.isclose(D[i], 1.0)])  # to avoid nan if value is close to 1.0
        D[ind_of_ones] = 1.0

        theta = np.arccos(D)  # compute principal angles
        dist = np.sqrt(np.sum(np.real(theta) ** 2))

        return dist

    def __call__(self, eta):
        """
        Compute 2D shape at a given spanwise location
        :param eta: physical spanwise location (assume eta is in [0,1])
        :return: (n, 2) array of xy coordinates of 2D shape
        """
        # Bound eta in [0,1]
        eta = np.minimum(np.maximum(0.0, eta), 1.0)

        # locate given eta according the nominal etas
        ind = np.where(self.eta_nominal <= eta)[-1][-1]
        t_start = self.t_nominal[ind]
        t_end = self.t_nominal[min(self.n_shapes-1, ind + 1)]
        if t_start == t_end:
            shape_phys = self.xy_nominal[ind]
        else:
            t = self.interpolator(eta)

            print(f'ind={ind}, ind+1 ={ind+1}, {self.t_nominal[ind]}, {self.t_nominal[ind + 1]}, {(t_end-t_start)}')
            t_norm = (t-t_start)/(t_end-t_start)

            # shape in Grassmannian
            geodesic_grassmann = self.exp(t_norm, self.xy_grassmann[ind], self.log_map[ind])

            # inverse LA transformation
            M_lin = np.linalg.inv((1 - t_norm) * self.M[ind] + t_norm * self.M[ind + 1])
            b_lin = (1 - t_norm) * self.b[ind] + t_norm * self.b[ind + 1]
            shape_phys = (np.sqrt(self.n_landmarks - 1) * geodesic_grassmann - b_lin) @ M_lin.T

        return [shape_phys[:, 0], shape_phys[:, 1]]
