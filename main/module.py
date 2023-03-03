""" docstring module """
import numpy as np
import quaternion 


class Polarizacion:
    def __init__(self,**kwargs):
        self.initial_state = kwargs.get("initial_state", [])
        self.materials = []
        self.states = [self.initial_state]
    def add_material(self, theta_values, delta):
        material = [
            np.quaternion(
                np.sin(theta),
                np.cos(delta),
                np.sin(theta) * np.cos(delta),
                np.sin(theta) - np.cos(delta)
            ) for theta in theta_values]
        
        self.materials.append(material)
    
    @property
    def matrix_states(self):
        return quaternion.as_float_array(self.states)
    
    def process(self):
        initial_state = self.states[0]
        for material in self.materials:
            emergent_state = [m*s*np.quaternion.conjugate(m) for m, s in zip(material, initial_state)]
            self.states.append(emergent_state)
            initial_state = emergent_state


n_data = 100
delta_material1 = 0.2
theta_material1 = np.linspace(0, 45, n_data)
delta_material2 = -0.2
theta_material2 = np.linspace(0, 45, n_data)

initial_state = [np.quaternion(1,1,1,1) for _ in range(n_data)]

obj = Polarizacion(initial_state=initial_state)

obj.add_material(theta_material1, delta_material1)
obj.add_material(theta_material2, delta_material2)

obj.process()
print(obj.initial_state)
print(obj.materials)
print(obj.matrix_states[0].T)
