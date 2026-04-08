"""
ui.py — Light theme control panel for orbital simulation.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import physics
import csv

# ── Palette (light theme) ─────────────────────────────────────────────────────
C = {
    'bg':        '#f0f2f7',
    'panel':     '#ffffff',
    'section':   '#f5f6fa',
    'border':    '#dde1ea',
    'accent':    '#2563eb',
    'green':     '#16a34a',
    'red':       '#dc2626',
    'amber':     '#d97706',
    'fg':        '#1a1a2e',
    'fg_dim':    '#9aa0b8',
    'fg_mid':    '#555577',
    'entry_bg':  '#f8f9fc',
    'header_bg': '#1e2235',
    'header_fg': '#ffffff',
}

FONT_MONO  = ('Courier New', 9)
FONT_MONO_B= ('Courier New', 9,  'bold')
FONT_TITLE = ('Courier New', 9, 'bold')
FONT_LABEL = ('Courier New', 8)
FONT_SMALL = ('Courier New', 7)


# ── Widget helpers ────────────────────────────────────────────────────────────

def _section_label(parent, row, text):
    tk.Label(parent, text=text, bg=C['panel'], fg=C['accent'],
             font=FONT_TITLE).grid(row=row, column=0, columnspan=2,
                                    sticky='w', padx=10, pady=(10, 3))

def _divider(parent, row):
    tk.Frame(parent, bg=C['border'], height=1).grid(
        row=row, column=0, columnspan=2, sticky='ew', padx=10, pady=(8, 0))

def _label(parent, row, text):
    tk.Label(parent, text=text, bg=C['panel'], fg=C['fg_mid'],
             font=FONT_LABEL, anchor='e').grid(
        row=row, column=0, sticky='e', padx=(8, 4), pady=2)

def _entry(parent, row, default='', width=15):
    e = tk.Entry(parent, width=width, bg=C['entry_bg'], fg=C['fg'],
                 insertbackground=C['accent'], relief='flat',
                 font=FONT_MONO, highlightthickness=1,
                 highlightbackground=C['border'],
                 highlightcolor=C['accent'])
    e.grid(row=row, column=1, sticky='ew', padx=(0, 10), pady=2)
    if default:
        e.insert(0, default)
    return e

def _btn(parent, text, cmd, bg=None, row=None, pady=3):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=bg or C['section'], fg=C['fg'],
                  activebackground=C['accent'], activeforeground='white',
                  relief='flat', font=FONT_MONO_B, cursor='hand2', pady=5,
                  highlightthickness=1, highlightbackground=C['border'])
    if row is not None:
        b.grid(row=row, column=0, columnspan=2, sticky='ew',
               padx=10, pady=pady)
    return b


# ── Main builder ──────────────────────────────────────────────────────────────

def build_control_panel(parent_frame, simulator, fig=None, ax=None):
    # Header bar
    hdr = tk.Frame(parent_frame, bg=C['header_bg'])
    hdr.pack(side=tk.TOP, fill=tk.X)
    tk.Label(hdr, text='ORBITAL SIM', bg=C['header_bg'], fg='white',
             font=('Courier New', 11, 'bold')).pack(side=tk.LEFT, padx=12, pady=10)
    tk.Label(hdr, text='Velocity Verlet · N-Body', bg=C['header_bg'],
             fg='#8899bb', font=FONT_SMALL).pack(side=tk.LEFT)

    # Scrollable body
    scroll_canvas = tk.Canvas(parent_frame, bg=C['panel'],
                              highlightthickness=0)
    scrollbar = tk.Scrollbar(parent_frame, orient='vertical',
                             command=scroll_canvas.yview)
    scroll_canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    panel = tk.Frame(scroll_canvas, bg=C['panel'])
    win_id = scroll_canvas.create_window((0, 0), window=panel, anchor='nw')

    def _on_cfg(e):
        scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all'))
        scroll_canvas.itemconfig(win_id, width=scroll_canvas.winfo_width())
    panel.bind('<Configure>', _on_cfg)

    def _scroll(e):
        scroll_canvas.yview_scroll(int(-1*(e.delta/120)), 'units')
    scroll_canvas.bind_all('<MouseWheel>', _scroll)

    panel.columnconfigure(1, weight=1)
    R = 0

    # ════════════════════════════════════════════════
    # 1. TAMBAH BENDA
    # ════════════════════════════════════════════════
    _section_label(panel, R, '▸ TAMBAH BENDA'); R += 1

    fields = [
        ('name',  'Nama',            ''),
        ('pos_x', 'Posisi X (AU)',   '0'),
        ('pos_y', 'Posisi Y (AU)',   '0'),
        ('pos_z', 'Posisi Z (AU)',   '0'),
        ('vel_x', 'Vel X (AU/hari)', '0'),
        ('vel_y', 'Vel Y (AU/hari)', '0'),
        ('vel_z', 'Vel Z (AU/hari)', '0'),
        ('mass',  'Massa (kg)',      ''),
        ('radius','Radius (km)',     ''),
        ('color', 'Warna',          'steelblue'),
    ]
    entries = {}
    for key, lbl, default in fields:
        _label(panel, R, lbl + ':')
        entries[key] = _entry(panel, R, default)
        R += 1

    _btn(panel, '＋  Tambah Benda',
         lambda: _on_add_body(simulator, entries),
         bg='#dcfce7', row=R); R += 1

    # ════════════════════════════════════════════════
    # 2. SIMULASI
    # ════════════════════════════════════════════════
    _divider(panel, R); R += 1
    _section_label(panel, R, '▸ SIMULASI'); R += 1

    _label(panel, R, 'Timestep (hari):')
    dt_entry = _entry(panel, R, '10'); R += 1

    _btn(panel, '↺  Perbarui Timestep',
         lambda: _on_update_dt(simulator, dt_entry),
         bg='#fef9c3', row=R); R += 1

    info_var = tk.StringVar(value='Hari simulasi: 0.0')
    tk.Label(panel, textvariable=info_var, bg=C['panel'], fg=C['fg_dim'],
             font=FONT_SMALL).grid(row=R, column=0, columnspan=2,
                                    sticky='w', padx=10, pady=(2, 0)); R += 1

    # ════════════════════════════════════════════════
    # 3. KAMERA
    # ════════════════════════════════════════════════
    _divider(panel, R); R += 1
    _section_label(panel, R, '▸ KAMERA'); R += 1

    tk.Label(panel, text='Preset sudut pandang:', bg=C['panel'],
             fg=C['fg_mid'], font=FONT_LABEL).grid(
        row=R, column=0, columnspan=2, sticky='w', padx=10, pady=(2, 3)); R += 1

    cam_row = tk.Frame(panel, bg=C['panel'])
    cam_row.grid(row=R, column=0, columnspan=2, sticky='ew', padx=10); R += 1

    presets = [('Top', 90, 0), ('Side', 0, 0), ('Iso', 30, 45), ('Front', 0, 90)]

    def _make_preset(elev, azim):
        def _apply():
            if ax:
                ax.view_init(elev=elev, azim=azim)
                if fig: fig.canvas.draw_idle()
        return _apply

    for i, (name, elev, azim) in enumerate(presets):
        tk.Button(cam_row, text=name, command=_make_preset(elev, azim),
                  bg=C['section'], fg=C['fg'], relief='flat',
                  font=FONT_SMALL, cursor='hand2', padx=6, pady=4,
                  activebackground=C['accent'], activeforeground='white',
                  highlightthickness=1,
                  highlightbackground=C['border']).grid(
            row=0, column=i, padx=2, sticky='ew')
        cam_row.columnconfigure(i, weight=1)

    def _reset_cam():
        if ax:
            ax.view_init(elev=25, azim=45)
            lim = 2 * physics.AU
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            if fig: fig.canvas.draw_idle()

    _btn(panel, '⟳  Reset Kamera', _reset_cam,
         bg=C['section'], row=R); R += 1

    tk.Label(panel, text='Drag kiri=putar  •  scroll=zoom  •  kanan=geser',
             bg=C['panel'], fg=C['fg_dim'], font=FONT_SMALL,
             wraplength=220, justify='left').grid(
        row=R, column=0, columnspan=2, sticky='w', padx=10, pady=(2, 0)); R += 1

    # ════════════════════════════════════════════════
    # 4. EKSPOR CSV
    # ════════════════════════════════════════════════
    _divider(panel, R); R += 1
    _section_label(panel, R, '▸ EKSPOR CSV'); R += 1

    _label(panel, R, 'Rekam s.d. hari:')
    until_entry = _entry(panel, R, '365'); R += 1

    # Status box
    status_box = tk.Frame(panel, bg=C['section'],
                          highlightthickness=1,
                          highlightbackground=C['border'])
    status_box.grid(row=R, column=0, columnspan=2,
                    sticky='ew', padx=10, pady=(4, 2)); R += 1
    status_var = tk.StringVar(value='Belum merekam.')
    tk.Label(status_box, textvariable=status_var, bg=C['section'],
             fg=C['fg_mid'], font=FONT_SMALL, wraplength=210,
             justify='left', padx=8, pady=6).pack(anchor='w')

    # Progress bar
    prog_outer = tk.Frame(panel, bg=C['border'], height=5)
    prog_outer.grid(row=R, column=0, columnspan=2, sticky='ew',
                    padx=10, pady=(0, 4)); R += 1
    prog_inner = tk.Frame(prog_outer, bg=C['accent'], height=5, width=0)
    prog_inner.place(x=0, y=0, relheight=1.0)

    btn_start = _btn(panel, '▶  Mulai Rekam',
                     lambda: _on_start_recording(
                         simulator, until_entry, status_var),
                     bg='#dbeafe', row=R); R += 1

    _btn(panel, '💾  Ekspor ke CSV',
         lambda: _on_export_csv(simulator, status_var),
         bg='#ede9fe', row=R); R += 1

    # padding bawah
    tk.Label(panel, text='', bg=C['panel']).grid(row=R, column=0); R += 1

    # ── Ticker ───────────────────────────────────────────────────────────────
    def _tick():
        info_var.set(f'Hari simulasi: {simulator.elapsed_days:.1f}')

        if simulator.recording:
            until   = _safe_int(until_entry)
            target  = simulator.record_until_day
            start   = target - until
            current = int(simulator.elapsed_days)
            pct     = max(0.0, min(1.0, (current - start) / max(1, until)))
            recorded= simulator.recorded_days
            status_var.set(f'⏺ Merekam {recorded}/{until} hari...')
            _set_prog(prog_outer, prog_inner, pct, C['accent'])
        elif simulator.recorded_days > 0:
            n = simulator.recorded_days
            rows = len(simulator._records)
            status_var.set(f'✓ {n} hari, {rows} baris — siap ekspor')
            _set_prog(prog_outer, prog_inner, 1.0, C['green'])
        else:
            status_var.set('Belum merekam.')
            _set_prog(prog_outer, prog_inner, 0.0, C['accent'])

        panel.after(300, _tick)

    _tick()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_prog(outer, inner, pct, color):
    outer.update_idletasks()
    w = outer.winfo_width()
    inner.configure(bg=color, width=max(0, int(w * pct)))
    inner.place(x=0, y=0, relheight=1.0)

def _safe_int(entry, default=365):
    try:
        return max(1, int(entry.get()))
    except ValueError:
        return default

def _on_add_body(simulator, entries):
    try:
        name = entries['name'].get().strip()
        if not name:
            messagebox.showerror('Error', 'Masukkan nama benda.'); return
        pos    = np.array([float(entries[k].get()) for k in ('pos_x','pos_y','pos_z')])
        vel    = np.array([float(entries[k].get()) for k in ('vel_x','vel_y','vel_z')])
        mass   = float(entries['mass'].get())
        radius = float(entries['radius'].get())
        color  = entries['color'].get().strip() or 'steelblue'
        if mass   <= 0: messagebox.showerror('Error', 'Massa harus positif.'); return
        if radius <= 0: messagebox.showerror('Error', 'Radius harus positif.'); return
        simulator.add_body(name, pos*physics.AU, vel*physics.AUDAY,
                           mass, radius*1000.0, color)
        for k in ('name','mass','radius'):
            entries[k].delete(0, tk.END)
        for k in ('pos_x','pos_y','pos_z','vel_x','vel_y','vel_z'):
            entries[k].delete(0, tk.END)
            entries[k].insert(0, '0')
        messagebox.showinfo('Berhasil', f"'{name}' ditambahkan.")
    except ValueError:
        messagebox.showerror('Error', 'Masukkan angka yang valid untuk semua field.')
    except Exception as exc:
        messagebox.showerror('Error', f'Gagal:\n{exc}')

def _on_update_dt(simulator, dt_entry):
    try:
        v = float(dt_entry.get())
        if v < 0: messagebox.showerror('Error', 'Timestep harus >= 0.'); return
        simulator.dt = v * physics.DAY
        messagebox.showinfo('Berhasil', f'Timestep → {v} hari.')
    except ValueError:
        messagebox.showerror('Error', 'Masukkan angka yang valid.')

def _on_start_recording(simulator, until_entry, status_var):
    until = _safe_int(until_entry)
    ok = simulator.start_recording(until)
    if ok:
        status_var.set(f'⏺ Memulai rekaman {until} hari...')
    else:
        messagebox.showerror('Error', 'Tambahkan benda ke simulasi terlebih dahulu.')

def _on_export_csv(simulator, status_var):
    if not simulator._records:
        messagebox.showwarning('Peringatan',
                               'Belum ada data.\nJalankan rekaman terlebih dahulu.')
        return
    n_days = simulator.recorded_days
    n_rows = len(simulator._records)
    filepath = filedialog.asksaveasfilename(
        title='Simpan Data Simulasi',
        defaultextension='.csv',
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
        initialfile=f'simulasi_{n_days}hari.csv',
    )
    if not filepath:
        return
    fieldnames = [
        'Hari', 'Nama',
        'Pos_X_AU', 'Pos_Y_AU', 'Pos_Z_AU',
        'Vel_X_AU_per_hari', 'Vel_Y_AU_per_hari', 'Vel_Z_AU_per_hari',
        'Kecepatan_m_per_s', 'Jarak_dari_Origin_AU',
    ]
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(simulator._records)
        status_var.set(f'✓ Tersimpan — {n_rows} baris')
        messagebox.showinfo('Ekspor Berhasil',
                            f'Berhasil disimpan.\n\n'
                            f'  Hari  : {n_days}\n'
                            f'  Baris : {n_rows}\n\n'
                            f'{filepath}')
    except Exception as e:
        messagebox.showerror('Error Ekspor', f'Gagal menyimpan:\n{e}')