# Aproximação explícita da equação do calor numa barra termicamente isolada, sujeita a condições de Dirichlet nas pontas.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constantes do problema
diffusividade = 1                                   # Taxa de transferência de calor (m^2/s)
l = 1                                               # Comprimento da barra (m)
N = int(100)                                        # Número de divisões da barra ()
eixo_x, delta_x = np.linspace(0, l, N,              # Largura dos intervalos discretizados da barra (m)
                              retstep=True,
                              endpoint=False)
delta_t = (delta_x)**2 / (2*diffusividade)          # Time step (s)
mu = diffusividade * delta_t / (delta_x)**2         # mu <= 0.5: critério de estabilidade (von Neumann)

# Matriz dos coeficientes ( A(n-1)X(n-1) )
A = np.zeros(N - 1)

for n in range(N - 1):
    a_nj = np.zeros(N - 1)
    a_nj[n] = 1 - 2*mu

    if (n > 0):
        a_nj[n - 1] = mu
    if (n < N - 2):
        a_nj[n + 1] = mu

    A = np.vstack([A, a_nj])

A = np.delete(A, 0, axis=0)

# Vetor dos valores de contorno (temperatura nas pontas da barra)
b = np.zeros(N - 1)
alpha = lambda t: 100*np.heaviside(t, 1)
beta  = lambda t: 0*np.heaviside(t, 1)

# Vetor de condições iniciais (perfil de temperatura da barra para t = 0)
u = np.zeros(N - 1)
f = lambda x: 40*(np.sin(np.pi*x))**3
for n in range(N - 1):
    u[n] = f(n*delta_x)

# Método iterativo
# u((j+1)*delta_t, m*delta_x) = μ*u(j*delta_t, (m+1)*delta_x) + (1-2μ)*u(j*delta_t, m*delta_x) + μ*u(j*delta_t, (m-1)*delta_x)
fig, ax = plt.subplots()
ax.grid(True)

ax.set_title(r"Difusão Térmica - Barra Isolada:   $\frac{\partial u}{\partial t}=\gamma\;\frac{\partial^{2}u}{\partial x^{2}}$")
ax.set_xlabel('Posição (m)')
ax.set_ylabel('Temperatura (k)')

if (alpha(0) > max(u)):
    ax.set_ylim(None, alpha(0))
if (alpha(0) < min(u)):
    ax.set_ylim(alpha(0), None)
if (beta(0) > max(u)):
    ax.set_ylim(None, beta(0))
if (beta(0) < min(u)):
    ax.set_ylim(beta(0), None)

line, = ax.plot(eixo_x[:99], u, linewidth=3)

def animate(j):
    global u, b
    b[0]  = mu*alpha(j*delta_t)
    b[-1] = mu*beta(j*delta_t)

    u_prox = np.matmul(A, u) + b
    u = u_prox

    line.set_ydata(u_prox)
    return line,

ani = animation.FuncAnimation(
    fig, animate, interval=4, blit=True, save_count=50)

plt.show()