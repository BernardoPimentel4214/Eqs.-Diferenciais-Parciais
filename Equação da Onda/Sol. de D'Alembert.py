import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.animation as animation

x, t, y, s = sp.symbols("x, t, y, s") # Posição, tempo, variável de integração de posição, variável de integração de tempo.
gauss = sp.exp(-(x - 2)**2 / (2 * 3**2))

# Monta a interface:

fig, axes = plt.subplots(1, 2, figsize=(14, 5), num="plot")

# Área das entradas:
axes[0].set_title("Problema de Valor Inicial")
axes[0].axis((0, 5, 0, 5))
axes[0].set_yticks([])
axes[0].set_xticks([])
axes[0].spines[["top", "right", "bottom", "left"]].set_visible(False)

# Caixas de texto:
caixa_ax1 = fig.add_axes([0.15, 0.8, 0.3, 0.075])
caixa_ax2 = fig.add_axes([0.15, 0.7, 0.3, 0.075])
caixa_ax3 = fig.add_axes([0.15, 0.6, 0.3, 0.075])
caixa_f = widgets.TextBox(caixa_ax1, "Deslocamento Inicial u(0,x):",
                          textalignment="center",
                          label_pad=0)
caixa_g = widgets.TextBox(caixa_ax2, r"Velocidade Inicial $\frac{\partial}{\partial t}u(0,x)$:",
                          textalignment="center",
                          label_pad=0.02)
caixa_F = widgets.TextBox(caixa_ax3, "Força Externa F(t,x):",
                          textalignment="center",
                          label_pad=0.125)

caixa_f.set_val("cos(x)*(u(x+5*pi/2)-u(x+3*pi/2)+u(x-3*pi/2)-u(x-5*pi/2))")
caixa_g.set_val("0")
caixa_F.set_val("0")

axes[0].text(1.7, 2.5, "Outros Parâmetros", ha="left", wrap=True, size="large")

caixa_ax4 = fig.add_axes([0.15, 0.4, 0.3, 0.075])
caixa_ax5 = fig.add_axes([0.15, 0.3, 0.3, 0.075])
caixa_ax6 = fig.add_axes([0.15, 0.2, 0.3, 0.075])
caixa_c = widgets.TextBox(caixa_ax4, "Velocidade da Onda c:",
                          textalignment="center",
                          label_pad=0.085)
caixa_D = widgets.TextBox(caixa_ax5, "Limites do Eixo x:",
                          textalignment="center",
                          label_pad=0.16)
caixa_T = widgets.TextBox(caixa_ax6, "Tempo Entre Quadros (ms):",
                          textalignment="center",
                          label_pad=0)

caixa_c.set_val("2")
caixa_D.set_val("10")
caixa_T.set_val("20")

# Botões:
bot_ax1 = fig.add_axes([0.150, 0.1, 0.145, 0.075])
bot_ax2 = fig.add_axes([0.305, 0.1, 0.145, 0.075])
bot_atualizar = widgets.Button(bot_ax1, 'Plotar / Atualizar')
bot_pausar = widgets.Button(bot_ax2, 'Pausar / Retomar')

# Área da animação:
axes[1].grid(True)
axes[1].set_title(r"Equação da Onda Unidimensional: $\frac{\partial^{2}u}{\partial t^{2}}=c^{2}\;\frac{\partial^{2}u}{\partial x^{2}}+F$")
axes[1].set_xlabel('x')
axes[1].set_ylabel('u(t,x)', rotation=0)


# Lógica da simulação:
class Simulador:
    def __init__(self):
        self.anim = None
        self.plotar_clicado = False
        self.pausar_clicado = False

    def iniciar_anim(self, event):
        if (self.pausar_clicado):
            return
        if (not self.plotar_clicado):
            # Obtém solução da EDP:
            f = sp.parse_expr(caixa_f.text.replace("u(", "Heaviside("))
            g = sp.parse_expr(caixa_g.text.replace("u(", "Heaviside("))
            F = sp.parse_expr(caixa_F.text.replace("u(", "Heaviside("))
            F = F.subs(x, y)
            F = F.subs(t, s)
            c = sp.parse_expr(caixa_c.text)
            D = float(caixa_D.text)
            T = int(caixa_T.text)

            xi = np.linspace(-D, D, 1000)
            ti = 0
            curva, = axes[1].plot(xi, [0]*len(xi), lw=2)

            sol = ((f.subs(x, x - c*t) + f.subs(x, x + c*t))/2
                    + 1/(2*c)*sp.integrate(g, (x, x + c*t, x - c*t))
                    + sp.integrate(F, (y, x + c*(t + s), x + c*(t - s)), (s, 0, t)))
            sol_numerica = sp.lambdify([x, t], sol)

            # Plota solução:
            axes[1].set_ylim([-max(sol_numerica(xi, 0)), 2 * max(sol_numerica(xi, 0))])
            axes[1].set_xlim([-D, D])

            def animar(i):
                curva.set_ydata(sol_numerica(xi, ti + i/50))
                return curva,

            self.anim = animation.FuncAnimation(fig, animar, interval=T, blit=True, save_count=50)
            self.plotar_clicado = True

        else:
            self.reiniciar_anim(event)

    def reiniciar_anim(self, event):
        self.anim.event_source.stop()
        self.plotar_clicado = False
        self.iniciar_anim(event)

    def pausar_anim(self, event):
        if (self.pausar_clicado):
            self.anim.resume()
        else:
            self.anim.pause()
            fig.canvas.draw_idle()
        self.pausar_clicado = not self.pausar_clicado

simulador = Simulador()
bot_atualizar.on_clicked(simulador.iniciar_anim)
bot_pausar.on_clicked(simulador.pausar_anim)

plt.show()
