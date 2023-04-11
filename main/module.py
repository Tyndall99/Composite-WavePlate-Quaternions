""" docstring module """
import numpy as np
import quaternion 
np.seterr(divide='ignore', invalid='ignore')

class Polarizacion:
    def __init__(self,**kwargs):
        self.initial_state = kwargs.get("initial_state", [])
        self.materials = []
        self.states = [self.initial_state]
        
    def add_linear_wp(self, theta_values, delta_values):
        material = [
            np.quaternion(
                np.cos(delta/2),
                np.cos(2*theta) * np.sin(delta/2),
                np.sin(2*theta) * np.sin(delta/2),
                0
            ) for theta, delta in zip(theta_values, delta_values)]
        
        self.materials.append(material)
    
    @property
    def matrix_states(self):
        return quaternion.as_float_array(self.states)
    
    def product(self):
        return np.prod(self.materials)
    
    def equiv_prop(self):
        q = self.product()
        gamma = 2 * np.arccos(q.w)
        alpha = np.arctan(q.y / q.x) / 2
        chi = np.arccos( np.sqrt((q.x**2 + q.y**2) / (1 - q.w**2))) / 2
        return [np.rad2deg(gamma), np.rad2deg(alpha),np.rad2deg(chi)]
        
    def equiv_rot(self):
        q = self.product()
        phi =  2 * np.arctan( q.z / q.w )
        alpha = (np.arctan(q.y / q.x) + phi / 2) / 2
        delta = 2 * np.arcsin( np.sqrt( q.x**2 + q.y**2 ) )
        return [np.rad2deg(phi), np.rad2deg(alpha), np.rad2deg(delta)]
    
    def process(self):
        initial_state = self.states[0]
        for m in self.materials[0]:
            emergent_state = m*initial_state*np.quaternion.conjugate(m)
            self.states.append(emergent_state)
            initial_state = emergent_state

n_data = 10
delta_materials = np.array([np.deg2rad(90), np.deg2rad(90)])
theta_materials = np.array([0, -np.pi/6])


#initial_state = [np.quaternion(1,1,1,1) for _ in range(n_data)]

initial_state = [np.quaternion(0,1,0,0)]

obj = Polarizacion(initial_state=initial_state)

obj.add_linear_wp(theta_materials, delta_materials)

#obj.process()
#print(obj.initial_state)
#print(obj.materials)
#print(obj.product())
#print(obj.matrix_states[0].T)
