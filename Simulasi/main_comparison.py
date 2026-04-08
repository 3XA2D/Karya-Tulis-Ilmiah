"""
main_comparison.py
──────────────────
Perbandingan 11 simulasi orbit dengan timestep berbeda.
dt = 1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100 hari

Layout visualisasi: grid 4 baris × 3 kolom (11 panel + 1 kosong)
"""

import os
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import physics
from comparison_sim import ComparisonSimulator, BodySpec, DT_DAYS

# ─────────────────────────────────────────────────────────────────────────────
#  Window
# ─────────────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Perbandingan Simulasi Orbit — 11 Timestep (Velocity Verlet)")
root.geometry("1500x860")
root.configure(bg='#1a1a1a')

# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4×3 (11 panel aktif)
# ─────────────────────────────────────────────────────────────────────────────
ROWS, COLS = 4, 3
fig = plt.figure(figsize=(11, 7.5), facecolor='#1a1a1a')
fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02,
                    wspace=0.05, hspace=0.22)

axes = []
for i in range(len(DT_DAYS)):
    ax = fig.add_subplot(ROWS, COLS, i + 1, projection='3d')
    ax.set_facecolor('#1a1a1a')
    axes.append(ax)

# Panel kosong terakhir
if len(DT_DAYS) < ROWS * COLS:
    ax_empty = fig.add_subplot(ROWS, COLS, ROWS * COLS)
    ax_empty.set_visible(False)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ComparisonSimulator
# ─────────────────────────────────────────────────────────────────────────────
csim = ComparisonSimulator(axes)

# ─────────────────────────────────────────────────────────────────────────────
#  Panel kontrol — kanan
# ─────────────────────────────────────────────────────────────────────────────
panel_outer = tk.Frame(root, bg='#252525', width=295)
panel_outer.pack(side=tk.RIGHT, fill=tk.Y)
panel_outer.pack_propagate(False)

# Scrollable canvas agar panel tidak terpotong di layar kecil
_sc = tk.Canvas(panel_outer, bg='#252525', highlightthickness=0)
_sb = tk.Scrollbar(panel_outer, orient='vertical', command=_sc.yview)
_sc.configure(yscrollcommand=_sb.set)
_sb.pack(side=tk.RIGHT, fill=tk.Y)
_sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

panel = tk.Frame(_sc, bg='#252525', padx=8, pady=8)
_sc_win = _sc.create_window((0, 0), window=panel, anchor='nw')

def _on_frame_configure(e):
    _sc.configure(scrollregion=_sc.bbox('all'))
def _on_canvas_configure(e):
    _sc.itemconfig(_sc_win, width=e.width)

panel.bind('<Configure>', _on_frame_configure)
_sc.bind('<Configure>', _on_canvas_configure)

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers UI  — semua widget dibuat dengan parent yang benar
# ─────────────────────────────────────────────────────────────────────────────
BG = '#252525'

def lbl(parent, text, fg='#cccccc', bold=False, size=9):
    f = ('Arial', size, 'bold') if bold else ('Arial', size)
    return tk.Label(parent, text=text, bg=BG, fg=fg, font=f, anchor='w')

def sep(parent):
    tk.Frame(parent, bg='#404040', height=1).pack(fill='x', pady=5)

def btn(parent, text, cmd, bg='#2980b9', size=9):
    return tk.Button(parent, text=text, command=cmd, bg=bg, fg='white',
                     relief='flat', font=('Arial', size, 'bold'),
                     activebackground=bg, cursor='hand2')

def field_row(parent, label, default='', width=14):
    """Buat satu baris label + entry, kembalikan Entry widget."""
    f = tk.Frame(parent, bg=BG)
    f.pack(fill='x', pady=1)
    tk.Label(f, text=label + ':', bg=BG, fg='#aaaaaa',
             font=('Arial', 8), width=17, anchor='e').pack(side='left')
    e = tk.Entry(f, width=width, bg='#3a3a3a', fg='white',
                 insertbackground='white', relief='flat', font=('Arial', 9))
    e.insert(0, default)
    e.pack(side='left', padx=(3, 0))
    return e

# ─────────────────────────────────────────────────────────────────────────────
#  Judul
# ─────────────────────────────────────────────────────────────────────────────
lbl(panel, "Simulasi Perbandingan", fg='white', bold=True, size=12).pack(pady=(2, 0))
lbl(panel, "dt = " + ", ".join(str(d) for d in DT_DAYS) + " hari",
    fg='#7ec8e3', size=8).pack(pady=(0, 4))

sep(panel)

# ─────────────────────────────────────────────────────────────────────────────
#  Form tambah benda
# ─────────────────────────────────────────────────────────────────────────────
lbl(panel, "Tambah Benda", bold=True, size=10, fg='white').pack(anchor='w')

body_fields = {
    'name'  : field_row(panel, "Nama",             "Matahari"),
    'pos_x' : field_row(panel, "Posisi X (AU)",    "0"),
    'pos_y' : field_row(panel, "Posisi Y (AU)",    "0"),
    'pos_z' : field_row(panel, "Posisi Z (AU)",    "0"),
    'vel_x' : field_row(panel, "Kec. X (AU/hari)", "0"),
    'vel_y' : field_row(panel, "Kec. Y (AU/hari)", "0"),
    'vel_z' : field_row(panel, "Kec. Z (AU/hari)", "0"),
    'mass'  : field_row(panel, "Massa (kg)",       "1.989e30"),
    'radius': field_row(panel, "Radius (km)",      "695700"),
    'color' : field_row(panel, "Warna",            "yellow"),
}

lbl(panel, "Benda ditambahkan:", size=8, fg='#888888').pack(anchor='w', pady=(4, 0))
body_listbox = tk.Listbox(panel, height=3, bg='#2a2a2a', fg='#cccccc',
                          font=('Arial', 8), relief='flat',
                          selectbackground='#3a3a3a')
body_listbox.pack(fill='x', pady=2)

specs_store: list = []

def on_add_body():
    try:
        name = body_fields['name'].get().strip()
        if not name:
            raise ValueError("Nama tidak boleh kosong.")

        pos = np.array([
            float(body_fields['pos_x'].get()),
            float(body_fields['pos_y'].get()),
            float(body_fields['pos_z'].get()),
        ]) * physics.AU

        vel = np.array([
            float(body_fields['vel_x'].get()),
            float(body_fields['vel_y'].get()),
            float(body_fields['vel_z'].get()),
        ]) * physics.AUDAY

        mass = float(body_fields['mass'].get())
        if mass <= 0:
            raise ValueError("Massa harus positif.")

        radius = float(body_fields['radius'].get()) * 1000.0
        if radius <= 0:
            raise ValueError("Radius harus positif.")

        color = body_fields['color'].get().strip() or 'white'

        spec = BodySpec(name=name, pos=pos, vel=vel,
                        mass=mass, radius=radius, color=color)
        specs_store.append(spec)
        body_listbox.insert('end', f" {name}   {mass:.2e} kg   {color}")

        # Kosongkan field kecuali color
        for k in ('name', 'pos_x', 'pos_y', 'pos_z',
                  'vel_x', 'vel_y', 'vel_z', 'mass', 'radius'):
            body_fields[k].delete(0, 'end')
        body_fields['color'].delete(0, 'end')
        body_fields['color'].insert(0, 'blue')

        csim.set_bodies(specs_store)
        canvas.draw_idle()
        messagebox.showinfo("Berhasil", f"'{name}' ditambahkan ke semua simulasi.")

    except ValueError as e:
        messagebox.showerror("Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", f"Gagal menambahkan benda:\n{e}")

def on_clear_bodies():
    if messagebox.askyesno("Konfirmasi", "Hapus semua benda?"):
        specs_store.clear()
        body_listbox.delete(0, 'end')
        csim.set_bodies([])
        canvas.draw_idle()

br = tk.Frame(panel, bg=BG)
br.pack(fill='x', pady=4)
btn(br, "✚  Tambah Benda", on_add_body, bg='#27ae60', size=9).pack(
    side='left', expand=True, fill='x', padx=(0, 2))
btn(br, "Reset", on_clear_bodies, bg='#7f8c8d', size=9).pack(
    side='left', expand=True, fill='x')

sep(panel)

# ─────────────────────────────────────────────────────────────────────────────
#  Konfigurasi simulasi
# ─────────────────────────────────────────────────────────────────────────────
lbl(panel, "Konfigurasi Simulasi", bold=True, size=10, fg='white').pack(anchor='w')

dur_e = field_row(panel, "Durasi (hari)", "1000")
spd_e = field_row(panel, "Langkah/tick",  "5")
lbl(panel, "  ↑ naikkan untuk simulasi lebih cepat", size=7, fg='#666666').pack(anchor='w')

# Folder output
outdir_var = tk.StringVar(value=os.path.expanduser("~"))

orow = tk.Frame(panel, bg=BG)
orow.pack(fill='x', pady=3)
tk.Label(orow, text="Output folder:", bg=BG, fg='#aaaaaa',
         font=('Arial', 8), width=17, anchor='e').pack(side='left')
tk.Button(orow, text="Pilih...",
          command=lambda: outdir_var.set(
              filedialog.askdirectory(title="Pilih folder output CSV") or outdir_var.get()
          ),
          bg='#444444', fg='white', relief='flat',
          font=('Arial', 8)).pack(side='left', padx=(3, 0))

tk.Label(panel, textvariable=outdir_var, bg=BG, fg='#555555',
         font=('Arial', 7), wraplength=265, anchor='w',
         justify='left').pack(fill='x', pady=(0, 3))

sep(panel)

# ─────────────────────────────────────────────────────────────────────────────
#  Status & progress per dt
# ─────────────────────────────────────────────────────────────────────────────
lbl(panel, "Progress per Timestep", bold=True, size=10, fg='white').pack(anchor='w')

status_var = tk.StringVar(value="Siap")
tk.Label(panel, textvariable=status_var, bg=BG, fg='#7ec8e3',
         font=('Arial', 8), wraplength=270, justify='left').pack(anchor='w', pady=(2, 4))

prog_cvs  = {}
prog_bars = {}
step_vars = {}

for dt in DT_DAYS:
    r = tk.Frame(panel, bg=BG)
    r.pack(fill='x', pady=1)

    tk.Label(r, text=f"dt={dt:3d}:", bg=BG, fg='#aaaaaa',
             font=('Courier', 7), width=7).pack(side='left')

    cv = tk.Canvas(r, height=8, bg='#3a3a3a', highlightthickness=0)
    cv.pack(side='left', fill='x', expand=True, padx=(2, 3))
    bar = cv.create_rectangle(0, 0, 0, 8, fill='#2ecc71', outline='')
    prog_cvs[dt]  = cv
    prog_bars[dt] = bar

    sv = tk.StringVar(value="  0/   0")
    tk.Label(r, textvariable=sv, bg=BG, fg='#888888',
             font=('Courier', 7), width=10).pack(side='left')
    step_vars[dt] = sv

sep(panel)

# ─────────────────────────────────────────────────────────────────────────────
#  Tombol kontrol
# ─────────────────────────────────────────────────────────────────────────────
def on_run():
    if not specs_store:
        messagebox.showerror("Error", "Tambahkan minimal satu benda terlebih dahulu.")
        return
    try:
        days = int(dur_e.get())
        if days <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Durasi harus bilangan bulat positif.")
        return

    out = outdir_var.get()
    if not os.path.isdir(out):
        messagebox.showerror("Error", f"Folder tidak valid:\n{out}")
        return

    try:
        spd = max(1, int(spd_e.get()))
    except ValueError:
        spd = 5

    import comparison_sim as _csm
    _csm.STEPS_PER_TICK = spd

    csim.configure(total_days=days, output_dir=out)
    csim.start()
    status_var.set(f"⏺ Berjalan...  0 / {csim.total_days} hari")

def on_stop():
    csim.stop()
    status_var.set(f"⏸ Dihentikan — hari {csim.elapsed_days:.0f}")

cr = tk.Frame(panel, bg=BG)
cr.pack(fill='x', pady=4)
btn(cr, "▶  Jalankan", on_run, bg='#27ae60').pack(
    side='left', expand=True, fill='x', padx=(0, 2))
btn(cr, "⏹  Stop", on_stop, bg='#c0392b').pack(
    side='left', expand=True, fill='x')

sep(panel)

# ─────────────────────────────────────────────────────────────────────────────
#  Info file output
# ─────────────────────────────────────────────────────────────────────────────
lbl(panel, "File CSV output:", size=8, fg='#888888', bold=True).pack(anchor='w')
for dt in DT_DAYS:
    lbl(panel, f"  hasil_dt{dt}hari.csv", size=7, fg='#5a9ec8').pack(anchor='w')

lbl(panel,
    "\nKolom CSV:\nLangkah, Hari, dt_hari, Nama,\n"
    "Pos_X/Y/Z (AU), Vel_X/Y/Z (AU/hari),\n"
    "Kecepatan (m/s), Jarak dari Origin (AU)",
    size=7, fg='#555555').pack(anchor='w', pady=(2, 8))

# ─────────────────────────────────────────────────────────────────────────────
#  Update progress UI
# ─────────────────────────────────────────────────────────────────────────────
def _update_ui():
    active_idx = csim.active_index
    for i, info in enumerate(csim.steps_info):
        dt  = info['dt']
        pct = info['progress']
        cv  = prog_cvs[dt]
        w   = max(cv.winfo_width(), 1)

        # Warna: hijau = aktif, abu = menunggu, biru = selesai
        if info['done']:
            color = '#3498db'   # selesai → biru
        elif i == active_idx:
            color = '#2ecc71'   # aktif → hijau
        else:
            color = '#555555'   # menunggu → abu

        cv.itemconfig(prog_bars[dt], fill=color)
        cv.coords(prog_bars[dt], 0, 0, int(w * pct), 8)
        step_vars[dt].set(f"{info['step']:5d}/{info['total']:5d}")

    active = csim.active_sim
    if csim.running and active:
        idx = csim.active_index + 1
        status_var.set(
            f"⏺ Sim {idx}/{len(DT_DAYS)}  dt={active.dt_days} hari\n"
            f"Langkah {active._step}/{active._total_steps}  "
            f"(hari {active.elapsed_days:.0f})"
        )
    elif csim.finished:
        status_var.set(
            f"✅ Selesai!\n"
            f"{csim.total_days} hari, {len(DT_DAYS)} file CSV tersimpan."
        )
        for dt, cv in prog_cvs.items():
            cv.coords(prog_bars[dt], 0, 0, cv.winfo_width(), 8)
            cv.itemconfig(prog_bars[dt], fill='#3498db')

# ─────────────────────────────────────────────────────────────────────────────
#  Animasi
# ─────────────────────────────────────────────────────────────────────────────
_draw_counter = 0
DRAW_EVERY    = 5

def _animate(_frame):
    global _draw_counter
    if csim.running:
        csim.tick()
        _draw_counter += 1
        if _draw_counter >= DRAW_EVERY:
            csim.draw_all()
            _draw_counter = 0
        _update_ui()
    artists = []
    for sim in csim.sims:
        for arts in sim._artists.values():
            artists.append(arts['scatter'])
            artists.append(arts['trail'])
    return artists

_anim = FuncAnimation(fig, _animate, interval=16,
                      blit=False, cache_frame_data=False)

root.mainloop()