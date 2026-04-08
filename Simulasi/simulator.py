import csv
import numpy as np
import physics
from body import Body
from tkinter import filedialog, messagebox

STEPS_PER_FRAME = 5


class Simulator:
    def __init__(self, ax, dt):
        self.ax           = ax
        self.dt           = dt          # detik per sub-langkah
        self.body_list    = []
        self._initialized = False
        self._artists     = {}          # id(body) -> {'scatter': artist, 'trail': artist}

        # ── Rekam data CSV ────────────────────────────────────────────────
        self._elapsed_seconds = 0.0     # total detik simulasi yang sudah berlalu
        self._last_recorded_day = -1    # hari terakhir yang sudah direkam
        self.recording        = False   # apakah sedang merekam
        self.record_until_day = 0       # rekam sampai hari ke-X
        self._records         = []      # list of dict, satu baris per benda per hari

        self._setup_axes()

    # ── Axes (dipanggil sekali) ───────────────────────────────────────────
    def _setup_axes(self):
        lim = 2 * physics.AU
        ax  = self.ax
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)

    # ── Manajemen benda ───────────────────────────────────────────────────
    def add_body(self, name, pos, vel, mass, radius, color):
        body = Body(name, pos, vel, mass, radius, color)
        self.body_list.append(body)

        size    = float(np.clip(body.radius / 1000, 20, 300))
        scatter = self.ax.scatter([], [], [], c=color, s=size,
                                  label=name, depthshade=False, zorder=5)
        trail,  = self.ax.plot([], [], [], c=color, alpha=0.35,
                               linewidth=0.8, zorder=4)
        self._artists[id(body)] = {'scatter': scatter, 'trail': trail}

        self._initialize_accelerations()
        self.ax.legend(loc='upper left', fontsize=8)

    # ── Fisika ────────────────────────────────────────────────────────────
    def _initialize_accelerations(self):
        physics._compute_all_accelerations(self.body_list)
        self._initialized = True

    def update(self):
        if not self.body_list:
            return
        if not self._initialized:
            self._initialize_accelerations()

        for _ in range(STEPS_PER_FRAME):
            physics.integrate_step(self.body_list, self.dt)
            self._elapsed_seconds += self.dt

        # Rekam satu titik trail per frame
        for body in self.body_list:
            body.trail_append(body.pos)

        # ── Rekam snapshot harian ─────────────────────────────────────────
        if self.recording:
            current_day = int(self._elapsed_seconds / physics.DAY)

            # Rekam setiap hari baru yang belum direkam
            if current_day > self._last_recorded_day:
                for day in range(self._last_recorded_day + 1, current_day + 1):
                    self._snapshot(day)
                self._last_recorded_day = current_day

            # Otomatis berhenti setelah hari target tercapai
            if current_day >= self.record_until_day:
                self.recording = False

    def _snapshot(self, day):
        """Simpan state semua benda pada hari tertentu."""
        for body in self.body_list:
            pos_au  = body.pos / physics.AU
            vel_aus = body.vel / physics.AUDAY
            speed   = float(np.linalg.norm(body.vel))
            dist    = float(np.linalg.norm(body.pos) / physics.AU)

            self._records.append({
                'Hari'             : day,
                'Nama'             : body.name,
                'Pos_X_AU'         : round(pos_au[0],  8),
                'Pos_Y_AU'         : round(pos_au[1],  8),
                'Pos_Z_AU'         : round(pos_au[2],  8),
                'Vel_X_AU_per_hari': round(vel_aus[0], 8),
                'Vel_Y_AU_per_hari': round(vel_aus[1], 8),
                'Vel_Z_AU_per_hari': round(vel_aus[2], 8),
                'Kecepatan_m_per_s': round(speed,      4),
                'Jarak_dari_Origin_AU': round(dist,    8),
            })

    # ── Render (tanpa ax.cla()) ───────────────────────────────────────────
    def draw(self):
        for body in self.body_list:
            arts = self._artists.get(id(body))
            if arts is None:
                continue

            arts['scatter']._offsets3d = (
                np.array([body.pos[0]]),
                np.array([body.pos[1]]),
                np.array([body.pos[2]]),
            )

            trail = body.trail_as_array()
            if trail is not None and len(trail) > 1:
                arts['trail'].set_data(trail[:, 0], trail[:, 1])
                arts['trail'].set_3d_properties(trail[:, 2])

    # ── Export CSV ────────────────────────────────────────────────────────
    def start_recording(self, until_day: int):
        """Mulai merekam dari hari saat ini sampai until_day."""
        if not self.body_list:
            messagebox.showerror("Error", "Tambahkan benda dulu sebelum merekam.")
            return False
        if until_day <= 0:
            messagebox.showerror("Error", "Hari target harus lebih dari 0.")
            return False

        # Reset hanya jika mulai rekam baru
        self._records.clear()
        self._last_recorded_day = int(self._elapsed_seconds / physics.DAY)
        self.record_until_day   = self._last_recorded_day + until_day
        self.recording          = True
        return True

    def export_csv(self):
        """Tulis _records ke file CSV yang dipilih pengguna."""
        if not self._records:
            messagebox.showwarning("Peringatan", "Belum ada data yang direkam.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Simpan CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="hasil_simulasi.csv",
        )
        if not filepath:
            return  # dibatalkan pengguna

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
                writer.writerows(self._records)

            hari_count = len(set(r['Hari'] for r in self._records))
            messagebox.showinfo(
                "Berhasil",
                f"Data {hari_count} hari × {len(self.body_list)} benda\n"
                f"({len(self._records)} baris) disimpan ke:\n{filepath}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan file:\n{e}")

    @property
    def elapsed_days(self):
        return self._elapsed_seconds / physics.DAY

    @property
    def recorded_days(self):
        return len(set(r['Hari'] for r in self._records)) if self._records else 0