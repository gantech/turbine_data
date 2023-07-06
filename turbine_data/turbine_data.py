import yaml
import numpy as np

from . import cst
from .airfoil import AirfoilTable, AirfoilTableInterp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .transforms import Quaternion

class TurbineData:
    """A class for aerodynamic data of a wind turbine"""

    def __init__(self, yaml_file):
        self.turbine = None
        self.read_yaml_file(yaml_file)

    def read_yaml_file(self, yaml_file):
        """Read and process the yaml style wind turbine ontology as defined at
        https://github.com/IEAWindTask37/IEA-15-240-RWT/tree/master/WT_Ontology

        Args:
            yaml_file (string): YAML file containing wind turbine ontology

        Return:
            None
        """
        self.yaml_file_ = yaml_file
        with open(yaml_file,'r') as f:
            self.turbine = yaml.load(f.read(), Loader=yaml.Loader)

        self.chord_values = self.turbine['components']['blade']['outer_shape_bem']['chord']['values']
        self.chord_grid = self.turbine['components']['blade']['outer_shape_bem']['chord']['grid']
        self.eta_cmax = self.chord_grid[np.argmax(self.chord_values)]
        self.twist_values = self.turbine['components']['blade']['outer_shape_bem']['twist']['values']
        self.twist_grid = self.turbine['components']['blade']['outer_shape_bem']['twist']['grid']
        self.r_axis = {}
        self.r_axis['x_values'] = np.array(self.turbine['components']['blade']['outer_shape_bem']['reference_axis']['x']['values'])
        self.r_axis['x_grid'] = np.array(self.turbine['components']['blade']['outer_shape_bem']['reference_axis']['x']['grid'])
        self.r_axis['y_values'] = np.array(self.turbine['components']['blade']['outer_shape_bem']['reference_axis']['y']['values'])
        self.r_axis['y_grid'] = np.array(self.turbine['components']['blade']['outer_shape_bem']['reference_axis']['y']['grid'])
        self.r_axis['z_values'] = np.array(self.turbine['components']['blade']['outer_shape_bem']['reference_axis']['z']['values'])
        self.r_axis['z_grid'] = np.array(self.turbine['components']['blade']['outer_shape_bem']['reference_axis']['z']['grid'])
        self.pitch_axis_values = np.array(self.turbine['components']['blade']['outer_shape_bem']['pitch_axis']['values'])
        self.pitch_axis_grid = np.array(self.turbine['components']['blade']['outer_shape_bem']['pitch_axis']['grid'])

        #Deal with normals of reference axis
        ref_grid = np.array(self.r_axis['z_grid'])
        n_ref = np.size(ref_grid)
        x_values = np.interp(self.r_axis['z_grid'], self.r_axis['x_grid'],
                             self.r_axis['x_values'])
        y_values = np.interp(self.r_axis['z_grid'], self.r_axis['y_grid'],
                            self.r_axis['y_values'])
        ref_axis = np.transpose(np.array( [x_values, y_values,
                                           self.r_axis['z_values']] ))
        self.normals_grid = np.zeros(n_ref+1)
        self.normals_grid[0] = ref_grid[0]
        self.normals_grid[-1] = ref_grid[-1]
        self.normals_grid[1:-1] = 0.5 * (ref_grid[1:]+ref_grid[:-1])
        self.normals = np.zeros((n_ref+1,3))
        self.normals[0] = ref_axis[1] - ref_axis[0]
        self.normals[-1] = ref_axis[-1] - ref_axis[-2]
        self.normals[1:-1] = ref_axis[1:] - ref_axis[:-1]
        for i in range(n_ref+1):
            self.normals[i] = self.normals[i]/np.linalg.norm(self.normals[i])

        self.airfoil_labels = self.turbine['components']['blade']['outer_shape_bem']['airfoil_position']['labels']
        self.airfoil_grid = self.turbine['components']['blade']['outer_shape_bem']['airfoil_position']['grid']

        self.rotor_diam = self.turbine['assembly']['rotor_diameter']
        self.hub_diam = self.turbine['components']['hub']['diameter']

        self.af_polars = {}

        for j,af in enumerate(self.turbine['airfoils']):
            af_name = af['name']
            for ap_re in af['polars']:
                aoa = np.degrees(np.array(ap_re['c_d']['grid']))
                cd_values = np.array(ap_re['c_d']['values'])
                cm_grid = np.degrees(np.array(ap_re['c_m']['grid']))
                cm_values = np.array(ap_re['c_m']['values'])
                cl_grid = np.degrees(np.array(ap_re['c_l']['grid']))
                cl_values = np.array(ap_re['c_l']['values'])

                self.af_polars[af_name] = {
                    'aoa': aoa.tolist(),
                    'cl': np.interp(aoa, cl_grid, cl_values).tolist(),
                    'cd': cd_values.tolist(),
                    'cm': np.interp(aoa, cm_grid, cm_values).tolist() }

    def dump_yaml(self, filename):
        """Dump the turbine configuration to a yaml file
        Args:
            filename (string): File name to dump the turbine configuration
        Return:
            None
        """
        yaml.dump(self.turbine, open(filename,'w'), default_flow_style=False)


    @property
    def num_blades(self):
        return 3

    @property
    def blade_length(self):
        """Return blade length"""
        return 0.5 * (self.rotor_diam - self.hub_diam)

    @property
    def rotor_diameter(self):
        """Return rotor diameter"""
        return self.rotor_diam

    @property
    def max_chord_loc(self):
        """Return the location of the maximum chord"""
        return self.eta_cmax

    @property
    def hub_diameter(self):
        """Return the hub diameter"""
        return self.hub_diam

    def set_tsr(self, tsr):
        """Set the tip speed ratio
        Args:
            tsr (double): Tip speed ratio
        Return:
            None
        """
        self.turbine['control']['tsr'] = tsr

    def get_tsr(self):
        """Get the tip speed ratio
        Args:
            None
        Return:
            tsr (double): Tip speed ratio
        """
        return float(self.turbine['control']['tsr'])
        
        
    def set_airfoil_labels(self, af_grid, af_labels):
        """Set the airfoil distribution
        Args:
            af_grid (np.array): Non-dimensionalized location along the blade
            af_labels (list): Airfoil labels at the grid locations
        Return:
            None
        """
        self.airfoil_grid = af_grid
        self.airfoil_labels = af_labels
        self.turbine['components']['blade']['outer_shape_bem']['airfoil_position']['grid'] = af_grid.tolist()
        self.turbine['components']['blade']['outer_shape_bem']['airfoil_position']['labels'] = af_labels

    def set_airfoil(self, af_label, af_data, rey, af_cst=None):
        """Set the airfoil polar and shape (optional) at a given station for a
           specified Reynolds number

        Args:
            af_label (string): String describing airfoil
            af_data (pd.DataFrame): Pandas DataFrame containing airfoil polar
            rey (double): Reynolds number
            af_cst (np.array): CST parameters describing airfoil shape
        Return:
            None

        """

        af_names = [af['name'] for af in self.turbine['airfoils'] ]
        if (af_label not in af_names):
            self.turbine['airfoils'].append({'name': af_label})

        for j,af in enumerate(self.turbine['airfoils']):
            af_name = af['name']
            if (af_name == af_label):
                #Assume only one Reynolds number for now
                afp = af['polars'][0]

                if (af_cst is not None):
                    BP = 8
                    ccst = cst.AirfoilShape.from_cst_parameters(
                             af_cst[0:BP+1],
                             af_cst[2*(BP+1)],
                             af_cst[BP+1:2*(BP+1)],
                             af_cst[2*(BP+1)+1])
                    af['coordinates']['x'] = ccst.xco.tolist()
                    af['coordinates']['y'] = ccst.yco.tolist()


                afp['c_l']['grid'] = np.radians(af_data['aoa']).tolist()
                afp['c_d']['grid'] = np.radians(af_data['aoa']).tolist()
                afp['c_l']['values'] = af_data['cl'].tolist()
                afp['c_d']['values'] = af_data['cd'].tolist()

                if ('c_m' not in af_data.columns):
                    afp['c_m']['grid'] = [ float(np.radians(af_data['aoa'].iloc[0])),
                                           float(np.radians(af_data['aoa'].iloc[-1]))]
                    afp['c_m']['values'] = [0.0, 0.0]
                else:
                    afp['c_m']['grid'] = np.radians(af_data['aoa']).tolist()
                    afp['c_m']['values'] = af_data['cm'].tolist()


    def set_chord(self, c_grid, c_values):
        """Set the chord distribution
        Args:
            c_grid (np.array): Non-dimensionalized location along the blade
            c_values (np.array): Chord values at the grid locations
        Return:
            None
        """
        self.chord_grid = c_grid
        self.chord_values = c_values
        self.turbine['components']['blade']['outer_shape_bem']['chord']['grid'] = c_grid.tolist()
        self.turbine['components']['blade']['outer_shape_bem']['chord']['values'] = c_values.tolist()

    def set_twist(self, t_grid, t_values):
        """Set the chord distribution
        Args:
            t_grid (np.array): Non-dimensionalized location along the blade
            t_values (np.array): Twist values at the grid locations (degrees)
        Return:
            None
        """
        self.twist_grid = t_grid
        self.twist_values = t_values
        self.turbine['components']['blade']['outer_shape_bem']['twist']['grid'] = t_grid.tolist()
        self.turbine['components']['blade']['outer_shape_bem']['twist']['values'] = np.radians(t_values).tolist()

    def twist(self, eta):
        """User-defined function describing the variation of twist as a function
        of the distance along pitch axis.
        Args:
            eta (double): Non-dimensionalized location along the blade
        Return:
            twist (double): Interpolated twist along the blade
        """
        return np.degrees(np.interp(eta, self.twist_grid, self.twist_values))

    def chord(self, eta):
        """User-defined function describing the variation of chord as a function of
        the distance along pitch axis
        Args:
            eta (double): Non-dimensionalized location along the blade
        Return:
            chord (double): Interpolated chord along the blade
        """
        return np.interp(eta, self.chord_grid, self.chord_values,
                         self.chord_values[0], self.chord_values[-1])

    def pitch_axis(self, eta):
        """User-defined function describing the variation of chordwise pitch
        axis location as a function of the distance along pitch axis.

        Args:
            eta (double): Non-dimensionalized location along the blade
        Return:
            pitch_axis (double): Interpolated pitch axis along the blade
        """
        return np.interp(eta, self.pitch_axis_grid, self.pitch_axis_values,
                         self.pitch_axis_values[0], self.pitch_axis_values[1])


    def blade_coordinates(self, r):
        """Compute non-dimensional coordinate along the blade given the radial
        distance from the hub

        Args:
            r (double): Dimensional radial coordinate from the hub
        Return:
            eta (double): Non-dimensional coordinate along the blade
        """

        return np.interp(r, (0.5 * self.hub_diam +
                             self.r_axis['z_values'])/self.rotor_diameter*2.0,
                         self.r_axis['z_grid'])

    def ref_axis_normal(self, eta):
        """Calculate the reference axis direction as a function of
        non-dimensional distance along reference axis

        Args:
            eta (real) : Non-dimensional coordinate along blade in [0,1]
        Return:
            c_norm (np.ndarray(3)): Numpy array containing local
            normal vector to reference axis
        """
        c_norm = np.array([
            np.interp(eta, self.normals_grid, self.normals[:,0]),
            0.0, #Ignore this direction because airfoil is only
                 #translated along the sweep
            np.interp(eta, self.normals_grid, self.normals[:,2])])
        c_norm = c_norm/np.linalg.norm(c_norm)
        return c_norm

    def ref_axis_quaternion(self, eta):
        """Get the quaternion that corresponds to rotation of +z axis [0,0,1]
        to the local normal vector at blade non-dimensional coordinate
        eta

        Args:
            eta (real) : Non-dimensional coordinate along blade in [0,1]
        Return:
            q (np.ndarray(4)): Numpy array containing quaternion
        """
        c_norm = self.r_axis_normal(eta)
        rot_axis = np.cross([0,0,1],c_norm)
        rot_axis = rot_axis/(np.linalg.norm(rot_axis) + 1e-16)
        cosTby2 = np.sqrt( 0.5*(1+c_norm[2]) )
        sinTby2 = np.sqrt( 0.5*(1-c_norm[2]) )
        q = [cosTby2,
             sinTby2*rot_axis[0],
             sinTby2*rot_axis[1],
             sinTby2*rot_axis[2]]
        return q

    def quaternion_rotate(self, q, coords):
        """Rotate a given set of coordinates using the quaternion q
        Args:
            q (np.ndarray(4)): Numpy array containing quaternion
            coords (np.ndarray(3)): Numpy array of coordinates
        Return:
            rot_coords(np.ndarray(3)): Numpy array of rotated coordinates
        """
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        rot_coords = np.copy(coords)
        for i,c in enumerate(coords):
            rot_coords[i,0] = (q0**2 + q1**2 - q2**2 - q3**2)*c[0]
            + 2*(q1*q2 - q0*q3)*c[1] + 2*(q0*q2 + q1*q3)*c[2]
            rot_coords[1,1] = 2*(q1*q2 + q0*q3)*c[0]
            + (q0**2 - q1**2 + q2**2 - q3**2)*c[1] + 2*(q2*q3 + q0*q1)*c[2]
            rot_coords[i,2] = 2*(q1*q3 - q0*q2)*c[0] + (q0*q1 + q2*q3)*c[1]
            + (q0**2 - q1**2 - q2**2 + q3**2)*c[2]
        return rot_coords

    def get_airfoil(self, af_name):
        """Get coordinates of airfoil by name
        Args:
            af_name (string): Name of airfoil
        Return:
            [x,y] (np.array): x,y coordinates of the airfoil
        """

        af_index = None
        for i,af in enumerate(self.turbine['airfoils']):
            if (af['name'] == af_name):
                af_index = i
        x_true = np.array(self.turbine['airfoils'][af_index]['coordinates']['x'])
        y_true = np.array(self.turbine['airfoils'][af_index]['coordinates']['y'])
        return [x_true, y_true]
        
        BP = 8
        af_cst = self.create_af_cst(af_name)
        ccst = cst.AirfoilShape.from_cst_parameters(
                    af_cst[0:BP+1],
                    af_cst[2*(BP+1)],
                    af_cst[BP+1:2*(BP+1)],
                    af_cst[2*(BP+1)+1],
                    af_cst[-2], af_cst[-1])

        return [ccst.xco, ccst.yco]

    def create_af_cst(self, af_name, BP=8):
        """ Create the CST representation of a given airfoil

        Args:
            af_name (string): Name of airfoil
            BP (integer): Order of Bernstein polynomials for CST
        Return:
            af_cst (np.array): CST representation of airfoil. Size 2 * (BP + 3)
                               [BP+1 (cst_lower), BP+1 (cst_upper),
                                te_lower, te_upper,
                                N1, N2]
        """

        af_index = None
        for i,af in enumerate(self.turbine['airfoils']):
            if (af['name'] == af_name):
                af_index = i
        x_true = np.array(self.turbine['airfoils'][af_index]['coordinates']['x'])
        y_true = np.array(self.turbine['airfoils'][af_index]['coordinates']['y'])

        if ('Cylinder' in af_name):
            af_cst = cst.AirfoilShape(x_true, y_true,shape_class='ellipse')
        else:
            af_cst = cst.AirfoilShape(x_true, y_true)
        te_lower = np.minimum(af_cst.te_lower,-0.001)
        te_upper = np.maximum(af_cst.te_upper,0.001)
        return np.r_[ af_cst.cst().cst_lower, af_cst.cst().cst_upper,
                      np.array([
                          te_lower, te_upper, af_cst.n1(), af_cst.n2()]) ]

    def create_blade_shape(self, interp_method='linear_cst', BP=8):
        """ Create the CST representation of a given airfoil

        Args:
            interp_method (string): Interpolation method
                                    (linear_cst or grassman)
            BP (integer): Order of Bernstein polynomials for CST
        Return:
            None
        """

        af_list = self.airfoil_labels
        eta_true = self.airfoil_grid

        af_cst = np.array([ self.create_af_cst(af,BP) for af in af_list ])
        self.bld_shape = cst.BladeShape(eta_true, af_cst[:,:])

    def calc_local_af_cst(self, eta, BP=8):
        """Calculate local airfoil CST coefficients at a specified
        non-dimensional location along the reference axis

        Args:
            eta (double) : Non-dimensional coordinate along blade in [0,1]
            BP (integer): Order of Bernstein polynomials for CST

        Return:
            af_cst (np.array):
        """
        inp_eta = eta
        if (eta < 0.0):
            eta = 0.0

        x,y = self.bld_shape(eta)
        af_cst = cst.AirfoilShape(x, y)
        return np.r_[ af_cst.cst().cst_lower, af_cst.cst().cst_upper,
                       np.array([
                           af_cst.te_lower, af_cst.te_upper,
                           af_cst.n1(), af_cst.n2()]) ]

    def apply_local_transformation(self, xinp, yinp, Epsilon):
        """Apply local transformations to create the airfoil shape"""

        inp_epsilon = Epsilon
        if (Epsilon < 0.0):
            Epsilon = 0.0

        chord = self.chord(Epsilon)
        norm_vec = self.ref_axis_normal(Epsilon)

        y = (xinp - self.pitch_axis(Epsilon)) * chord
        x = yinp * chord

        q1 = Quaternion.from_axis_angle([0,0,1], -self.twist(Epsilon))
        q2 = Quaternion.from_two_vectors([0,0,1], norm_vec)
        q = q2 * q1
        rot_coords = np.array([ q(xyz) for xyz in
                                np.column_stack ( [x,y,np.zeros_like(x)] )])

        trans_vec = self.ref_axis(Epsilon)
        # if (inp_epsilon < 0.0):
        #     z_coord = self.hub_diam * 0.5 * 0.5
        # else:
        #     interp_z = np.interp(Epsilon,
        #                          self.ref_axis['z_grid'],
        #                          self.ref_axis['z_values'],
        #                          self.ref_axis['z_values'][0],
        #                          self.ref_axis['z_values'][-1])
        #     z_coord  = interp_z +  self.hub_diam * 0.5

        # trans_vec = np.array([
        #     np.interp(Epsilon, self.ref_axis['x_grid'], self.ref_axis['x_values']),
        #     np.interp(Epsilon, self.ref_axis['y_grid'], self.ref_axis['y_values']),
        #     z_coord
        # ])

        pnts = rot_coords + trans_vec

        return pnts    

    def get_tbycmax(self, af_cst):
        """Calculate local airfoil thickness

        Args:
            af_cst (np.array) : CST array

        Return:
            tbyc (double): Thickness / chord ratio
        """
        
        order = int(np.size(af_cst)/2) - 2
        cst_lower = af_cst[:order+1]
        cst_upper = af_cst[order+1:2*(order+1)]
        te_lower = af_cst[-2]
        te_upper = af_cst[-1]
        af_shape = cst.AirfoilShape.from_cst_parameters(cst_lower=cst_lower,te_lower=te_lower,cst_upper=cst_upper,te_upper=te_upper)
        return np.max(af_shape.yupper-af_shape.ylower[::-1])
    

    def dump_all_af_cst(self, yaml_filename="airfoil_cst_db.yaml"):
        """Dumps the CST coefficients corresponding to all the airfoils into a
        yaml file

        Args:
          yaml_filename (string): File name to dump the CST coefficients
        """
        af_list = self.airfoil_labels
        af_cst_yaml = {
            'airfoil_cst_db': {af: np.ndarray.tolist( self.create_af_cst(af) )
                               for i,af in enumerate(af_list)}
        }
        yaml.dump(af_cst_yaml, open(yaml_filename,'w'),default_flow_style=False)


    def dump_all_af_polar_cst(self, yaml_filename="airfoil_polar_db.yaml"):
        """Dumps the polars and CST coefficients corresponding to all the
           airfoils into a yaml file

        Args:
          yaml_filename (string): File name to dump the CST coeffs and polars
        """
        af_list = self.airfoil_labels
        af_polar_cst_yaml = {
          af: { 'cst': np.ndarray.tolist( self.create_af_cst(af) ),
                'aoa': self.af_polars[af]['aoa'],
                'cl': self.af_polars[af]['cl'],
                'cd': self.af_polars[af]['cd'],
                'cm': self.af_polars[af]['cm']
               } for af in af_list }
        yaml.dump(af_polar_cst_yaml, open(yaml_filename,'w'),
                  default_flow_style=False)


    def ref_axis(self, eta):
        """Get the reference axis points interpolated to the non-dimensional
        locations alone the blade

        Args:
            eta (double): Non-dimensionalized location along the blade
        Return:
            ref_axis (double): Interpolated reference axis along the blade
        """

        eta = np.asarray(eta)
        ref_axis = np.zeros((np.size(eta),3))
        ref_axis[:,0] = np.interp(eta,
                                  self.r_axis['x_grid'],
                                  self.r_axis['x_values'])
        ref_axis[:,1] = np.interp(eta,
                                  self.r_axis['y_grid'],
                                  self.r_axis['y_values'])
        ref_axis[:,2] = np.interp(eta,
                                  self.r_axis['z_grid'],
                                  self.r_axis['z_values'],
                                  left=-0.25*self.hub_diam) + self.hub_diam * 0.5
        return ref_axis

    def af_polar(self, af_name):
        """Get an AirfoilTable object for a given airfoil

        Args:
            af_name (string): Airfoil name
        Return:
            af_table (AirfoilTable): Airfoil Polar
        """
        return AirfoilTable(aoa = np.array(self.af_polars[af_name]['aoa']),
                           cl = np.array(self.af_polars[af_name]['cl']),
                           cd = np.array(self.af_polars[af_name]['cd']),
                           cm = np.array(self.af_polars[af_name]['cm']) )

    def af_table(self, eta):
        """Get an AirfoilTable object with polars interpolated to a
        radial location on the blade

        Args:
            eta (double): Non-dimensionalized location along the blade
        Return:
            af_table (AirfoilTableInterp): Interpolated AirfoilTable
        """

        af_indices = list(range(len(self.airfoil_labels)))
        loc = np.interp(eta, self.airfoil_grid, af_indices)
        interp_wt = loc - int(loc)

        if ( (loc == 8) ):
            loc -= 1
            interp_wt += 1.0

        af_left = self.airfoil_labels[ int(loc) ]
        af_right = self.airfoil_labels[ int(loc) + 1 ]
        afp_left = AirfoilTable(aoa = np.array(self.af_polars[af_left]['aoa']),
                                 cl = np.array(self.af_polars[af_left]['cl']),
                                 cd = np.array(self.af_polars[af_left]['cd']),
                                 cm = np.array(self.af_polars[af_left]['cm']) )
        afp_right = AirfoilTable(aoa = np.array(self.af_polars[af_right]['aoa']),
                                 cl = np.array(self.af_polars[af_right]['cl']),
                                 cd = np.array(self.af_polars[af_right]['cd']),
                                 cm = np.array(self.af_polars[af_right]['cm']))
        return AirfoilTableInterp(afp_left, afp_right, interp_wt)
