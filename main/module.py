""" docstring module """
import numpy as np
import quaternion 


class Polarizacion:
    def __init__(self,**kwargs):
        self.initial_state = kwargs.get("initial_state", [])
        self.materials = []
        self.states = [self.initial_state]
        
    def add_linear_wp(self, theta_values, delta_values):
        material = [
            np.quaternion(
                np.cos(delta/2),
                np.cos(theta) * np.sin(delta/2),
                np.sin(theta) * np.sin(delta/2),
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
        gamma = 2 * np.acos(q.w)
        alpha = np.atan(q.y / q.x) / 2
        chi = np.acos( np.sqrt((q.x**2 + q.y**2) / (1 - q.w**2)**2) / 2)
        return [gamma, alpha, chi]
        
    def equiv_rot(self):
        q = self.product()
        phi =  2 * np.atan( q.z / q.w )
        alpha = np.atan( (q.y * q.w + q.x * q.z) / (q.x * q.w - q.y * q.z) ) / 2
        delta = 2 * np.asin( np.sqrt( q.x**2 + q.y**2 ) )
        return [phi, alpha, delta]
    
    def process(self):
        initial_state = self.states[0]
        emergent_state = [m*initial_state*np.quaternion.conjugate(m) for m in self.materials[0]]
        self.states.append(emergent_state)

n_data = 10
delta_materials = np.linspace(0, 90, n_data)
theta_materials = np.linspace(0, 45, n_data)


#initial_state = [np.quaternion(1,1,1,1) for _ in range(n_data)]

initial_state = [np.quaternion(0,1,0,0)]

obj = Polarizacion(initial_state=initial_state)

obj.add_linear_wp(theta_materials, delta_materials)

#obj.process()
print(obj.initial_state)
print(obj.materials)
print(obj.producat())
#print(obj.matrix_states[0].T)
