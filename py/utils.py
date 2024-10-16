import numpy as np
from body import Body

def uniform_disc(n):
    np.random.seed(0)
    inner_radius = 25.0
    outer_radius = np.sqrt(n) * 5.0

    bodies = []

    m = 1e6
    center = Body([0.0, 0.0], [0.0, 0.0], m, inner_radius)
    bodies.append(center)

    while len(bodies) < n:
        a = np.random.uniform(0.0, 2 * np.pi)
        sin_a = np.sin(a)
        cos_a = np.cos(a)
        t = inner_radius / outer_radius
        r = np.random.uniform(t * t, 1.0)
        pos = np.array([cos_a, sin_a]) * outer_radius * np.sqrt(r)
        vel = np.array([sin_a, -cos_a])
        mass = 1.0
        radius = mass ** (1.0/3.0)
        bodies.append(Body(pos, vel, mass, radius))

    # Sort bodies by distance from center
    bodies.sort(key=lambda b: np.dot(b.pos, b.pos))
    total_mass = 0.0
    for body in bodies:
        total_mass += body.mass
        if np.allclose(body.pos, np.zeros(2)):
            continue
        v = np.sqrt(total_mass / np.linalg.norm(body.pos))
        body.vel *= v

    return bodies
