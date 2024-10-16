from body import Body
from quadtree import Quad, Quadtree
import numpy as np
import utils

class Simulation:
    def __init__(self):
        self.dt = 0.05
        self.frame = 0
        self.theta = 1.0
        self.epsilon = 1.0
        self.quadtree = Quadtree(self.theta, self.epsilon)
        self.bodies = utils.uniform_disc(500)  # Adjust N to a reasonable value

    def step(self):
        self.iterate()
        self.collide()
        self.attract()
        self.frame += 1

    def attract(self):
        quad = Quad.new_containing(self.bodies)
        self.quadtree.clear(quad)
        for body in self.bodies:
            self.quadtree.insert(body.pos, body.mass)
        self.quadtree.propagate()
        for body in self.bodies:
            body.acc = self.quadtree.acc(body.pos)

    def iterate(self):
        for body in self.bodies:
            body.update(self.dt)

    def collide(self):
        # Simple collision detection for demonstration purposes
        n = len(self.bodies)
        for i in range(n):
            for j in range(i+1, n):
                self.resolve(i, j)

    def resolve(self, i, j):
        b1 = self.bodies[i]
        b2 = self.bodies[j]

        d = b2.pos - b1.pos
        r = b1.radius + b2.radius

        if np.dot(d, d) > r * r:
            return

        v = b2.vel - b1.vel
        d_dot_v = np.dot(d, v)

        m1 = b1.mass
        m2 = b2.mass

        weight1 = m2 / (m1 + m2)
        weight2 = m1 / (m1 + m2)

        if d_dot_v >= 0.0 and not np.allclose(d, np.zeros(2)):
            tmp = d * (r / np.linalg.norm(d) - 1.0)
            b1.pos -= weight1 * tmp
            b2.pos += weight2 * tmp
            return

        v_sq = np.dot(v, v)
        d_sq = np.dot(d, d)
        r_sq = r * r

        discriminant = d_dot_v * d_dot_v - v_sq * (d_sq - r_sq)
        if discriminant < 0:
            return  # No collision

        t = (d_dot_v + np.sqrt(discriminant)) / v_sq

        b1.pos -= b1.vel * t
        b2.pos -= b2.vel * t

        d = b2.pos - b1.pos
        d_dot_v = np.dot(d, v)
        d_sq = np.dot(d, d)

        tmp = d * (1.5 * d_dot_v / d_sq)
        b1.vel += tmp * weight1
        b2.vel -= tmp * weight2
        b1.pos += b1.vel * t
        b2.pos += b2.vel * t
