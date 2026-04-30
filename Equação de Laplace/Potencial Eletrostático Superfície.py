# Aproximação numérica do potencial eletrostático num plano (seção transversal de uma cavidade).

import numpy as np
import matplotlib.pyplot as plt

# Constantes e grandezas do problema
a, b = 0, 2                                     # a < x < b
c, d = 0, 1                                     # c < y < d
m = int(300)                                    # Subdivisões do eixo x
n = int(m*(d - c)/(b - a))                      # Subdivisões do eixo y
eixo_x, delta_x = np.linspace(a, b, m,          # Δx = (b - a) / (m - 1)
                              retstep=True)
eixo_y, delta_y = np.linspace(c, d, n,          # Δy = (d - c) / (n - 1)
                              retstep=True)
p = delta_x/delta_y                             # p ≈ 1 (critério de estabilidade)

epsilon_0 = 8.854187e-12                        # Permissividade do espaço livre (F/m)                 
epsilon_r = 1                                   # Permissividade relativa do meio (adimensonal, cte p/ meios homogêneos)
rho_s = lambda x, y: 5e-11*np.exp(-np.sqrt(     # Densidade superficial de cargas (C/m^2)
                                    (x - 2*(b - a)/3)**2 +
                                    (y - 2*(d - c)/3)**2))
parede_esq = lambda y: np.sin(y*np.pi/(d-c))    # Potencial elétrico nas paredes (V): x = a
parede_dir = lambda y: 0                        #                                     x = b
parede_inf = lambda x: 0                        #                                     y = c
parede_sup = lambda x: 0                        #                                     y = d

# Matriz dos coeficientes
diag_principal = np.full(m - 1, 2*(1 + p**2))
diag_inf = diag_sup = np.full(m - 2, -1)
B_p = (np.diag(diag_principal, k=0) +
       np.diag(diag_inf, k=-1) +
       np.diag(diag_sup, k=1))

L = np.zeros(((n - 1), (n - 1), (m - 1), (m - 1)))
U = np.zeros(((n - 1), (n - 1), (m - 1), (m - 1)))
Uj = [B_p]
Lj = []
for j in range(1, n - 1):
    Lj.append((-p**2)*np.linalg.inv(Uj[j - 1]))
    Uj.append(B_p + (p**2)*Lj[j - 1])

for k in range(n - 1):
    L[k][k] = np.eye(m - 1)
    U[k][k] = Uj[k]
    if (k > 0):
        L[k][k-1] = Lj[k - 1]
    if (k < n - 2):
        U[k][k + 1] = (-p**2)*np.eye(m - 1)

# Vetores dos termos independentes e incógnitas aq ta sus
f = []
for j in range(n - 1):
    fj = np.zeros(m - 1)
    for i in range(m - 1):
        x = a + (i + 1)*delta_x
        y = c + (j + 1)*delta_y
        fj[i] = - (delta_x**2 / (epsilon_0 * epsilon_r))*rho_s(x, y)
        if (i == 0):
            fj[i] += parede_esq(y)
        if(i == m - 2):
            fj[i] += parede_dir(y)
        if(j == 0):
            fj[i] += p**2*parede_inf(x)
        if(j == n - 2):
            fj[i] -= p**2*parede_sup(x)
    f.append(fj)

# Solução
z = np.zeros((n - 1, m - 1))
z[0] = f[0]
for j in range(1, n - 1):
    z[j] = f[j] - np.linalg.matmul(Lj[j - 1], z[j - 1])

w = np.zeros((n - 1, m - 1))
w[n - 2] = z[n - 2]
for k in range(n - 3, -1, -1):
    w[k] = np.linalg.matmul(np.linalg.inv(Uj[k]), w[k + 1] - (p**(-2))*z[k])

x, y = np.meshgrid(eixo_x[:-1], eixo_y[:-1])

fig, ax = plt.subplots(1, 1)
ax.contourf(x, y, w, cmap="jet")

plt.show()