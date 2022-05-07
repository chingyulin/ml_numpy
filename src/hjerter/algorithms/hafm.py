import numpy as np

p0 = np.array([1, 8])
p1 = np.array([7, 3])

line_a = np.stack([p0, p1])
q = np.array([3, 2])

v = p1 - p0
u = q - p0


def find_angle(v1, v2):
    return np.arccos(np.sum(v1 * v2) / np.sqrt(np.sum(v1 ** 2) * np.sum(v2 ** 2)))


theta_0 = find_angle(u, v)
theta = np.arctan(-v[0] / v[1])
p_v_u = (np.sum(u * v) / np.sum(v ** 2)) * v
t = p0 + p_v_u

d = np.sqrt(np.sum((t - q) ** 2))

w = p1 - q
theta_1 = find_angle(w, v)

print("Theta_0", theta_0)
print("Theta_1", theta_1)
print("Theta", theta)
print("Distance", d)
