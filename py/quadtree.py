import numpy as np

class Quad:
    def __init__(self, center, size):
        self.center = np.array(center, dtype=float)
        self.size = float(size)

    @staticmethod
    def new_containing(bodies):
        min_x = min(body.pos[0] for body in bodies)
        min_y = min(body.pos[1] for body in bodies)
        max_x = max(body.pos[0] for body in bodies)
        max_y = max(body.pos[1] for body in bodies)

        center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2], dtype=float)
        size = max(max_x - min_x, max_y - min_y)

        return Quad(center, size)

    def find_quadrant(self, pos):
        quadrant = 0
        if pos[1] > self.center[1]:
            quadrant |= 2
        if pos[0] > self.center[0]:
            quadrant |= 1
        return quadrant

    def into_quadrant(self, quadrant):
        size = self.size / 2
        center = self.center.copy()
        center[0] += ((quadrant & 1) - 0.5) * size
        center[1] += (((quadrant >> 1) & 1) - 0.5) * size
        return Quad(center, size)

    def subdivide(self):
        return [self.into_quadrant(i) for i in range(4)]

class Node:
    def __init__(self, next_index, quad):
        self.children = 0
        self.next = next_index
        self.pos = np.zeros(2, dtype=float)
        self.mass = 0.0
        self.quad = quad

    def is_leaf(self):
        return self.children == 0

    def is_branch(self):
        return self.children != 0

    def is_empty(self):
        return self.mass == 0.0

class Quadtree:
    ROOT = 0

    def __init__(self, theta, epsilon):
        self.t_sq = theta * theta
        self.e_sq = epsilon * epsilon
        self.nodes = []
        self.parents = []

    def clear(self, quad):
        self.nodes.clear()
        self.parents.clear()
        self.nodes.append(Node(0, quad))

    def subdivide(self, node_index):
        self.parents.append(node_index)
        children_index = len(self.nodes)
        node = self.nodes[node_index]
        node.children = children_index

        nexts = [children_index + 1, children_index + 2, children_index + 3, node.next]
        quads = node.quad.subdivide()
        for i in range(4):
            self.nodes.append(Node(nexts[i], quads[i]))

        return children_index

    def insert(self, pos, mass):
        node_index = self.ROOT

        while self.nodes[node_index].is_branch():
            quadrant = self.nodes[node_index].quad.find_quadrant(pos)
            node_index = self.nodes[node_index].children + quadrant

        if self.nodes[node_index].is_empty():
            self.nodes[node_index].pos = pos
            self.nodes[node_index].mass = mass
            return

        p = self.nodes[node_index].pos
        m = self.nodes[node_index].mass
        if np.allclose(pos, p):
            self.nodes[node_index].mass += mass
            return

        while True:
            children_index = self.subdivide(node_index)

            q1 = self.nodes[node_index].quad.find_quadrant(p)
            q2 = self.nodes[node_index].quad.find_quadrant(pos)

            if q1 == q2:
                node_index = children_index + q1
            else:
                n1 = children_index + q1
                n2 = children_index + q2

                self.nodes[n1].pos = p
                self.nodes[n1].mass = m
                self.nodes[n2].pos = pos
                self.nodes[n2].mass = mass
                return

    def propagate(self):
        for node_index in reversed(self.parents):
            node = self.nodes[node_index]
            i = node.children

            mass = (self.nodes[i].mass + self.nodes[i+1].mass + 
                    self.nodes[i+2].mass + self.nodes[i+3].mass)
            pos = (self.nodes[i].pos * self.nodes[i].mass +
                   self.nodes[i+1].pos * self.nodes[i+1].mass +
                   self.nodes[i+2].pos * self.nodes[i+2].mass +
                   self.nodes[i+3].pos * self.nodes[i+3].mass)
            node.pos = pos / mass
            node.mass = mass

    def acc(self, pos):
        acc = np.zeros(2, dtype=float)

        node_index = self.ROOT
        while True:
            n = self.nodes[node_index]

            d = n.pos - pos
            d_sq = np.dot(d, d)

            if n.is_leaf() or n.quad.size * n.quad.size < d_sq * self.t_sq:
                denom = (d_sq + self.e_sq) * np.sqrt(d_sq)
                if denom != 0:
                    acc += d * (n.mass / denom)
                if n.next == 0:
                    break
                node_index = n.next
            else:
                node_index = n.children

        return acc
