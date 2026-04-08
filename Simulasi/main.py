"""
main.py — Entry point for Orbital Simulation.
"""

import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import physics
import ui
from simulator import Simulator

BG = '#f0f2f7'

root = tk.Tk()
root.title('Simulasi Orbit — N-Body 3D  |  Velocity Verlet')
root.geometry('1380x820')
root.configure(bg=BG)
root.minsize(900, 600)

# ── PENTING: pack panel DULU, baru canvas ────────────────────────────────────
# Kalau canvas di-pack duluan dia akan makan semua ruang

sim_placeholder = [None]  # diisi setelah ax dibuat

# Frame kanan untuk panel kontrol
right_frame = tk.Frame(root, bg=BG, width=270)
right_frame.pack(side=tk.RIGHT, fill=tk.Y)
right_frame.pack_propagate(False)

# Frame kiri untuk plot
left_frame = tk.Frame(root, bg=BG)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# ── Figure ────────────────────────────────────────────────────────────────────
PLOT_BG = '#ffffff'
fig = plt.figure(facecolor=PLOT_BG)
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#f7f8fc')
ax.tick_params(colors='#888899', labelsize=7)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = True
    pane.set_facecolor('#eeeef5')
    pane.set_edgecolor('#d0d2de')
ax.grid(True, color='#d8dae8', linewidth=0.4)
ax.set_xlabel('X (AU)', color='#666688', fontsize=7)
ax.set_ylabel('Y (AU)', color='#666688', fontsize=7)
ax.set_zlabel('Z (AU)', color='#666688', fontsize=7)
ax.view_init(elev=25, azim=45)

canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# ── Simulator ─────────────────────────────────────────────────────────────────
sim = Simulator(ax, dt=10 * physics.DAY)

# ── Control panel (dimasukkan ke right_frame) ─────────────────────────────────
ui.build_control_panel(right_frame, sim, fig=fig, ax=ax)

# ── Animation ─────────────────────────────────────────────────────────────────
def _animate(_frame):
    sim.update()
    sim.draw()
    canvas.draw_idle()
    return []

_anim = FuncAnimation(fig, _animate, interval=16, blit=False,
                      cache_frame_data=False)

root.mainloop()