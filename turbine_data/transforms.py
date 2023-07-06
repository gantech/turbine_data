import numpy as np

class Quaternion:

    def __init__(self, q0, q1, q2, q3):
        self._q = np.array([q0,q1,q2,q3])

    @classmethod
    def from_axis_angle(cls, axis, angle):
        """ Setup quaternion that rotates a specified angle about an axis

        Args:
            axis (np.ndarray): Setup rotation about this axis
            angle (float): Rotation angle in degrees
        Return:
            q (Quaternion): Quaternion object
        """
        axis = np.asarray(axis,dtype=float)
        angle = np.radians(angle)
        sinThetaby2  = np.sin(angle*0.5)
        cosThetaby2 = np.cos(angle*0.5)
        axis *= sinThetaby2 / np.linalg.norm(axis)
        return cls(cosThetaby2, *axis)

    @classmethod
    def from_two_vectors(cls, vec1, vec2):
        """Create a quaternion that rotates vec1 to vec2

        Args:
            vec1 (np.ndarray):
            vec2 (np.ndarray):
        Return:
            q (Quaternion): Quaternion object that rotates vec1 to vec2

        """
        v1Cv2 = np.cross(vec1, vec2)
        q = np.r_[ np.dot(vec1, vec2), v1Cv2]
        q /= np.linalg.norm(q)
        q[0] += 1.0 #Change angle from theta to theta/2
        q /= np.linalg.norm(q)
        return cls( *q )

    @property
    def q(self):
        return self._q

    def __call__(self, vec):

        """Apply the transformation to the supplied vector

        Args:
            vec (np.ndarray): Vector to transform
        Return:
            rot_vec (np.ndarray): Rotated vector
        """
        q0,q1,q2,q3 = [*self.q]

        rot_vec = np.zeros(3)
        rot_vec[0] = (q0*q0 + q1*q1 - q2*q2- q3*q3)*vec[0] + 2.0*(q1*q2 - q0*q3)*vec[1] + 2.0*(q1*q3 + q0*q2)*vec[2]
        rot_vec[1] = 2.0*(q1*q2 + q0*q3)*vec[0] + (q0*q0 - q1*q1 + q2*q2 - q3*q3)*vec[1] + 2.0*(q2*q3 - q0*q1)*vec[2]
        rot_vec[2] = 2.0*(q1*q3 - q0*q2)*vec[0] + 2.0*(q2*q3 + q0*q1)*vec[1] + (q0*q0 - q1*q1 - q2*q2 + q3*q3)*vec[2]

        return rot_vec

    def inverted(self):
        """Return inverted Quaternion

        Return:
            q_inv (Quaternion): Inverted Quaternion

        """
        return Quaternion(*(-self.q))

    def inverse_transform(self, vec):

        """Apply inverse transformation to the supplied vector

        Args:
            vec (np.ndarray): Vector to transform
        Return:
            rot_vec (np.ndarray): Rotated vector
        """
        q0,q1,q2,q3 = [*self.q]

        q1 *= -1
        q2 *= -1
        q3 *= -1

        rot_vec = np.zeros(3)
        rot_vec[0] = (q0*q0 + q1*q1 - q2*q2- q3*q3)*vec[0] + 2.0*(q1*q2 - q0*q3)*vec[1] + 2.0*(q1*q3 + q0*q2)*vec[2]
        rot_vec[1] = 2.0*(q1*q2 + q0*q3)*vec[0] + (q0*q0 - q1*q1 + q2*q2 - q3*q3)*vec[1] + 2.0*(q2*q3 - q0*q1)*vec[2]
        rot_vec[2] = 2.0*(q1*q3 - q0*q2)*vec[0] + 2.0*(q2*q3 + q0*q1)*vec[1] + (q0*q0 - q1*q1 - q2*q2 + q3*q3)*vec[2]

        return rot_vec

    def __mul__(self, q_2):

        """Create a composition of this quaternion with q2

        Args:
            q_2 (Quaternion): Quaternion

        Return:
            q3 (Quaternion): Composition of self * q2
        """

        q2 = q_2._q
        q1 = self._q
        q3 = np.r_[ q1[0] * q2[0] - np.dot(q1[1:], q2[1:]),
                    q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:]) ]
        return Quaternion(*q3)
    
