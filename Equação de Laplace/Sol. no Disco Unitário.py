import numpy as np
import matplotlib.pyplot as plt

N = 100

r = np.linspace(0, 1, N)
theta = np.linspace(-np.pi, np.pi, N)
phi = np.linspace(-np.pi, np.pi, N)
r_i, theta_i = np.meshgrid(r, theta)

x_i = r_i * np.cos(theta_i)
y_i = r_i * np.sin(theta_i)

phi_3d = phi[np.newaxis, np.newaxis, :]

h = lambda phi: np.cos(phi) + 0.5 * np.sin(3 * phi) + 0.2 * np.cos(7 * phi)
dphi = phi[1] - phi[0]

u = (1/(2*np.pi)) * np.sum([
    h(phi[n]) * ((1 - r_i**2) / (1 + r_i**2 - 2 * r_i * np.cos(theta_i - phi[n]))) * dphi 
    for n in range(len(phi))
], axis=0)

M = np.max(np.abs(h(phi)))
for i in range(len(u[0])):
    for j in range(len(u[0])):
        if np.abs(u[i][j]) > M*1.2:
            u[i][j] = u[i-1][j-1]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.contourf(x_i, y_i, u, N, cmap='jet')
ax.set_title("Equação de Laplace no Disco Unitário")

plt.show()