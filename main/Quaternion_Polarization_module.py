""" docstring module """
import numpy as np
import quaternion 
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Ellipse
from matplotlib.transforms import Affine2D
np.seterr(divide='ignore', invalid='ignore')

class State(np.quaternion):

    def __init__(self, alpha = 0, chi = 0):
        '''
        This class defines a polarization state in the representation of quaternions.
        To construct a polarization state you need to specify the oritentation angle "alpha" 
        and the ellipticity angle "chi" of the associated polarization ellipse you want to
        describr through quaternions.

        By defect, the angles ()"alpha" , "chi") must be defined in DEGREES!
        '''
        self.alpha = np.deg2rad(alpha)
        self.chi = np.deg2rad(chi)
        # Call the superclass "np.quaternion" to return the State as a quaternion
        super().__init__(1,
                        np.cos(2*self.alpha) * np.cos(2*self.chi), 
                        np.sin(2*self.alpha) * np.cos(2*self.chi), 
                        np.sin(2*self.chi)
                        )
        
    def operate(self, operator):
        '''
        This function operates, through the quaternion product, the polarization state
        by an operator or a list of operators (another quaternions as Waveplates, Rotators
        or Composite Waveplates) 
        '''
        if np.size(operator) > 1:
            transformed_states = []
            for element in operator:
                transformed_states.append(element*self*np.quaternion.conjugate(element))
            return transformed_states
        else:
            transformed_state = operator*self*np.quaternion.conjugate(operator)
            return transformed_state     
    
    def ellipse(self):
        '''
        This function returns the graphic of the polarization ellipse asociated with the Polarization State
        '''
        fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})

        horizontal_axis = 1 / np.sqrt(np.tan(self.chi)**2 + 1)
        vertical_axis = np.tan(self.chi) / np.sqrt(np.tan(self.chi)**2 + 1)

        ellipse = Ellipse(
            xy = (0, 0),
            width = horizontal_axis,
            height = vertical_axis,
            angle = np.rad2deg(self.alpha),
            facecolor="none",
            edgecolor="b"
            )
        
        ax.add_patch(ellipse)

        # Plot an arrow marker at the end point of minor axis
        vertices = ellipse.get_co_vertices()
        t = Affine2D().rotate_deg(ellipse.angle)

        ax.plot(
            vertices[0][0],
            vertices[0][1],
            color="b",
            marker=MarkerStyle(">", "full", t),
            markersize=10
                )
        #Note: To reverse the orientation arrow, switch the marker type from > to <.
        plt.xlim([-0.5,0.5])
        plt.ylim([-0.5,0.5])

        plt.show()

    def Poincare_sphere(self):
        Graphic(self)
     
class Waveplate(np.quaternion):

    def __init__(self, phase_shift, axis_angle = 0, eigen_state = (0,0)):
        '''
        This class defines a waveplate operator in the representation of quaternions.
        To construct a Waveplate you need to specify its phase_shift, the angle of orientation
        "axis_angle" of the main axis with respect the x-axis, and the eigenstate.

        By defect, the angles "phase_shift" and "axis_angle" must be defined in DEGREES!
        '''
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
        '''
        This function operates, through the quaternion product, the operator of the waveplate with respect
        to an initial State or a list of initial States.
        '''
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
    def __init__(self, theta):
        self.theta = theta
        self._waveplates = (Qwp(),
                            Hwp(self.theta),
                            Hwp(),
                            Qwp(90)
                            )
        super().__init__(self._waveplates)

def Graphic(State):
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
    r = 1
    x = r* np.cos(u)*np.sin(v)
    y =  r* np.sin(u)*np.sin(v)
    z = r* np.cos(v)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar la esfera
    ax.plot_wireframe(x, y, z, rstride=5, cstride=6, color='grey', alpha=0.3,
                      linewidth=1.3)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='grey', alpha=0.2,
                    linewidth=0)
    
    #Graficar los ejes
    ax.plot([-1.1, 1.1], [0, 0], [0, 0], color='black', linewidth=1.5, alpha = 0.7)
    ax.plot([0, 0], [-1.1, 1.1], [0, 0], color='black', linewidth=1.5, alpha = 0.7)
    ax.plot([0, 0], [0, 0], [-1.1, 1.1], color='black', linewidth=1.5, alpha = 0.7)
    
    #Graphic some meridians for aesthetic
    theta = np.linspace(0, 2*np.pi, 100)
 
    xx = r * np.cos(theta)
    yy = r * np.sin(theta)
    zz = np.zeros_like(theta)
    ax.plot(xx, yy, zz, color='gray', linewidth=1.5, alpha = 1)
    ax.plot(yy, zz, xx, color='gray', linewidth=1.5, alpha = 1)

    # Graphic the Data on the Sphere
    if np.size(State) > 1:
        s1 = [state.x for state in State]
        s2 = [state.y for state in State]
        s3 = [state.z for state in State]
    else:
        s1 = State.x
        s2 = State.y
        s3 = State.z


    ax.scatter(s1, s2, s3, color='red', alpha = 1, s=10)

    #Configure the image
    fig.set_size_inches(9, 9)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_aspect("equal")
    ax.patch.set_alpha(0)
    plt.tight_layout()
    plt.axis('off')
    
    # Add the S_1 S_2 and S_3 names for the main axis of Poincaré Sphere
    ax.text(1.15, 0, 0, '$S_1$', fontsize=18)
    ax.text(0, 1.15, 0, '$S_2$', fontsize=18)
    ax.text(0, 0, 1.15, '$S_3$', fontsize=18)
        
    plt.show()
    

