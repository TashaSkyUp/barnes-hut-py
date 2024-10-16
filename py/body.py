import numpy as np

class Body:
    def __init__(self, pos, vel, mass, radius):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.acc = np.zeros(2, dtype=float)
        self.mass = mass
        self.radius = radius

    def update(self, dt):
        self.vel += self.acc * dt
        self.pos += self.vel * dt
