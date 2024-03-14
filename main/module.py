""" docstring module """
import numpy as np
import quaternion 
np.seterr(divide='ignore', invalid='ignore')

class State(np.quaternion):

    def __init__(self, alpha = 0, chi = 0):
        self.alpha = np.deg2rad(alpha)
        self.chi = np.deg2rad(chi)
        # Call the superclass "np.quaternion" to return the State as a quaternion
        super().__init__(1,
                        np.cos(2*self.alpha) * np.cos(2*self.chi), 
                        np.sin(2*self.alpha) * np.cos(2*self.chi), 
                        np.sin(2*self.chi)
                        )
        
    def ellipse(self):
        pass

    def Poincare_sphere(self):
        pass

    def operate(self, operator):
        if np.size(operator) > 1:
            transformed_states = []
            for element in operator:
                transformed_states.append(element*self*np.quaternion.conjugate(element))
            return transformed_states
        else:
            transformed_state = operator*self*np.quaternion.conjugate(operator)
            return transformed_state     
            
class Waveplate(np.quaternion):

    def __init__(self, phase_shift, axis_angle = 0, eigen_state = (0,0)):
        self.phase_shift = np.deg2rad(phase_shift)
        self.axis_angle = np.deg2rad(axis_angle)
        self.alpha, self.chi = np.deg2rad(eigen_state)
        self.eigenstate = State(self.alpha + self.axis_angle/2, self.chi)
         # Call the superclass "np.quaternion" to return the Retarder as a quaternion
        super().__init__(np.cos(self.phase_shift / 2),
                         np.cos(2*self.alpha + self.axis_angle) * np.cos(2*self.chi) * np.sin(self.phase_shift / 2),
                         np.sin(2*self.alpha + self.axis_angle) * np.cos(2*self.chi) * np.sin(self.phase_shift / 2),
                         np.sin(2*self.chi) * np.sin(self.phase_shift / 2))
        
    def operate(self, state):
        if np.size(state) > 1:
            transformed_states = []
            for element in state:
                transformed_states.append(self*element*np.quaternion.conjugate(self))
            return transformed_states
        else:
            transformed_state = self*state*np.quaternion.conjugate(self)
            return transformed_state

class Qwp(Waveplate):
    
    def __init__(self, axis_angle = 0, eigenstate = (0,0)):
        self._axis_angle = axis_angle
        self._eigenstate = eigenstate
        super().__init__(90, self._axis_angle, self._eigenstate)

class Hwp(Waveplate):
    
    def __init__(self, axis_angle = 0, eigenstate = (0,0)):
        self._axis_angle = axis_angle
        self._eigenstate = eigenstate
        super().__init__(180, self._axis_angle, self._eigenstate)

class Rotation(np.quaternion):
    
    def __init__(self, angle):
        self.angle = angle
        # Call the superclass "np.quaternion" to return the Rotation as a quaternion
        super().__init__(np.cos(self.angle / 2),
                         0,
                         0,
                         np.sin(self.angle / 2))

class Composite_waveplate(Waveplate):

    def product(quaternions):
        return np.prod(quaternions)
    
    def equivalent(quaternions):
        q = Composite_waveplate.product(quaternions)
        
        #Esta definición da el angulo negativo, posiblemente por la funcion arccos
        gamma =  2*np.arccos(q.w)
        #Se uso esta otra identidad y da correcto con el arcsin
        #gamma = 2*np.arcsin(np.sqrt(1-q.w**2))
    
        if q.x == 0:
            alpha = np.pi/4
        else:
            alpha = np.arctan2(q.y , q.x) / 2
        
        if 1. - q.w**2 == 0:
            chi = np.pi/4
        else:
            chi = np.arccos( np.sqrt((q.x**2 + q.y**2) / (1. - q.w**2))) / 2
        
        dict = {'gamma': np.rad2deg(gamma), 
                'alpha': np.rad2deg(alpha), 
                'chi': np.rad2deg(chi)}
        
        return dict

    def jones_caracterization(quaternions):
        q = Composite_waveplate.product(quaternions)
        
        phi =  2 * ( np.pi/2 - np.arctan2( q.w , q.z ))
        
        if q.x == 0:
            alpha = (np.pi/2 + phi/2) / 2
        else:
            alpha = (np.arctan2(q.y, q.x) + phi/2) / 2
        
        delta = 2 * np.arcsin( np.sqrt( q.x**2 + q.y**2 ) )
        
        #delta =  2 * np.arccos( np.sqrt( q.w**2 + q.z**2 ) )
        
        dict = {'phi': np.rad2deg(phi),
                 'alpha_p':np.rad2deg(alpha),
                   'delta': np.rad2deg(delta)}
        
        return dict
    
    def __init__(self, waveplates):
        self.waveplates = list(waveplates)
        self.phase_shifts = np.array([wp.phase_shift for wp in self.waveplates])
        self.main_angles = np.array([wp.angle for wp in self.waveplates])
        self.eigenstates = np.array([wp.eigenstate for wp in self.waveplates])
        self.equiv_waveplate = Composite_waveplate.product(self.waveplates)
        super().__init__(self.caracterization()['General_caracterization']['gamma'],
                       0, 
                       (self.caracterization()['General_caracterization']['alpha'], 
                        self.caracterization()['General_caracterization']['chi']))
        
    def caracterization(self):
        caracterization = {'Jones_Theorem': Composite_waveplate.jones_caracterization(self.waveplates),
            'General_caracterization': Composite_waveplate.equivalent(self.waveplates)}
        return caracterization

    def add_wp(self, waveplate):
        self.waveplates.append(waveplate)
    
class CW_as_BC(Composite_waveplate):
    """
        This subclass generates a particular composite_waveplate
        composed of a Hwp(theta), Hwp(). This is a phase variable Circular
        Waveplate
        
        we can select the angle "theta" of the first HWP and then,
        use the methods to obtain the equivalent representations and
        the transformation of a set of initial states
        
    """
    def __init__(self, theta):
        self.theta = theta
        self._waveplates = (Hwp(self.theta),
                            Hwp()
                            )
        super().__init__(self._waveplates)

class Tuneable_BL(Composite_waveplate):
    """
        This subclass generates a particular composite_waveplate
        composed of a Qwp(theta), Hwp(theta_p), Hwp(), Qwp(theta + pi/2).
        This is a full tuneable Linear Waveplate, i.e. you can change its phase
        and linear eigenstate varying "theta" and "theta_p"
    """
    def __init__(self, theta, theta_1):
        self.theta = theta
        self.theta_1 = theta_1
        self._waveplates = (Qwp(self.theta),
                            Hwp(self.theta_1),
                            Hwp(),
                            Qwp(self.theta + 90)
                            )
        super().__init__(self._waveplates)

class FTB(Composite_waveplate):
    """
        This subclass generates a particular composite_waveplate
        composed of a Hwp(theta_2), Hwp(), Qwp(theta), Hwp(theta_1), Hwp(), Qwp(theta + 90°).
        This is a full tuneable Waveplate, i.e. you can change its phase
        and eigenstate varying the values of "theta", "theta_1" and "theta_2"
    """
    def __init__(self, theta, theta_1, theta_2):
        self.theta = theta
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self._waveplates = (Hwp(self.theta_2),
                           Hwp(),
                           Qwp(self.theta),
                           Hwp(theta_1),
                           Hwp(),
                           Qwp(self.theta + 90)
        )
        super().__init__(self._waveplates)

class Pancharaman(Composite_waveplate):
    """
        This subclass generates a particular composite waveplate
        composed of a Qwp(theta_1), Hwp(theta_2), Qwp(theta_1). This is a phase
        variable linear Waveplate 
    """
    def __init__(self, theta_1, theta_2):
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self._waveplates = (Qwp(self.theta_1),
                            Hwp(self.theta_2),
                            Qwp(self.theta_1)
                            )
        super().__init__(self._waveplates)

class Messiadi(Composite_waveplate):
    """
        This subclass generates a particular composite waveplate
        composed of a Qwp(), Hwp(theta), Hwp(), QWP(90°). It is a phase
        variable Linear Waveplate
    """
    def __init__(self, theta ):
        self.theta = theta
        self._waveplates = (Qwp(),
                            Hwp(self.theta),
                            Hwp(),
                            Qwp(90)
                            )
        super().__init__(self._waveplates)