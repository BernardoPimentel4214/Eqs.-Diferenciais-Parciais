import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Domain and grid
# -----------------------------
a, b = 0.0, 2.0      # x-domain
c, d = 0.0, 1.0      # y-domain

Nx = 80              # grid points in x
Ny = 40              # grid points in y

dx = (b - a) / (Nx - 1)
dy = (d - c) / (Ny - 1)

x = np.linspace(a, b, Nx)
y = np.linspace(c, d, Ny)

# -----------------------------
# Physical parameters
# -----------------------------
epsilon_0 = 8.854187e-12
epsilon_r = 1.0
epsilon = epsilon_0 * epsilon_r

def rho(x, y):
    return 0.0   # no charge (Laplace case)

# -----------------------------
# Boundary conditions
# -----------------------------
def V_left(y):
    return np.sin(np.pi * y / (d - c))

def V_right(y):
    return 0.0

def V_bottom(x):
    return 0.0

def V_top(x):
    return 0.0

# -----------------------------
# Initialize solution
# -----------------------------
V = np.zeros((Ny, Nx))

# Apply boundaries
V[:, 0]  = V_left(y)
V[:, -1] = V_right(y)
V[0, :]  = V_bottom(x)
V[-1, :] = V_top(x)

# -----------------------------
# Gauss-Seidel / SOR solver
# -----------------------------
max_iter = 5000
tol = 1e-6
omega = 1.8   # relaxation (1 = Gauss-Seidel, 1<ω<2 = SOR)

dx2 = dx**2
dy2 = dy**2
denom = 2 * (dx2 + dy2)

for it in range(max_iter):
    V_old = V.copy()

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):

            rhs = -rho(x[i], y[j]) / epsilon

            V_new = (
                (V[j, i+1] + V[j, i-1]) * dy2 +
                (V[j+1, i] + V[j-1, i]) * dx2 -
                rhs * dx2 * dy2
            ) / denom

            # SOR update
            V[j, i] = (1 - omega) * V[j, i] + omega * V_new

    # Convergence check
    error = np.max(np.abs(V - V_old))
    if error < tol:
        print(f"Converged in {it} iterations")
        break
else:
    print("Did not converge")

# -----------------------------
# Plot
# -----------------------------
X, Y = np.meshgrid(x, y)

plt.figure(figsize=(6, 4))
plt.contourf(X, Y, V, levels=50, cmap='jet')
plt.colorbar(label="Potential (V)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Electrostatic Potential (Poisson Solver)")
plt.show()