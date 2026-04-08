"""
comparison_sim.py
─────────────────
Menjalankan 11 simulasi independen dengan timestep berbeda:
  dt = 1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100 hari

Setiap simulasi berjalan penuh selama total_days hari.
Setiap langkah integrasi dicatat ke CSV masing-masing.

Titik waktu per simulasi:
  dt=1  : hari 1, 2, 3, ..., N         (N baris)
  dt=11 : hari 11, 22, 33, ...,        (N//11 baris)
  dst.

Untuk perbandingan di titik waktu yang sama, kolom 'Hari' bisa
digunakan sebagai join key antar file CSV.
"""

import csv
import os
from dataclasses import dataclass
from typing import List

import numpy as np

import physics
from body import Body


DT_DAYS: List[int] = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100]

# Berapa langkah fisika yang dilakukan per tick animasi per sim.
# Makin besar → simulasi lebih cepat selesai, UI update lebih jarang.
STEPS_PER_TICK = 1

FIELDNAMES = [
    'Langkah', 'Hari',
    'dt_hari', 'Nama',
    'Pos_X_AU', 'Pos_Y_AU', 'Pos_Z_AU',
    'Vel_X_AU_per_hari', 'Vel_Y_AU_per_hari', 'Vel_Z_AU_per_hari',
    'Kecepatan_m_per_s', 'Jarak_dari_Origin_AU',
]


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BodySpec:
    name  : str
    pos   : np.ndarray  # meter
    vel   : np.ndarray  # m/s
    mass  : float
    radius: float
    color : str


# ─────────────────────────────────────────────────────────────────────────────
class SingleSim:
    """
    Satu simulasi untuk satu nilai dt.
    Merekam state ke CSV setiap langkah integrasi.
    """

    def __init__(self, dt_days: int, ax):
        self.dt_days   = dt_days
        self.dt        = dt_days * physics.DAY
        self.ax        = ax
        self.body_list : List[Body] = []
        self._artists  = {}

        self._step          = 0          # langkah ke-berapa
        self._elapsed_days  = 0.0
        self._total_steps   = 0          # total langkah yang harus diselesaikan
        self._initialized   = False
        self.done           = False

        # File CSV dibuka saat simulasi dimulai, ditutup saat selesai
        self._csv_file   = None
        self._csv_writer = None

        self._setup_axes()

    # ── Axes ─────────────────────────────────────────────────────────────
    def _setup_axes(self):
        lim = 2 * physics.AU
        ax  = self.ax
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel("X", fontsize=6)
        ax.set_ylabel("Y", fontsize=6)
        ax.set_zlabel("Z", fontsize=6)
        ax.set_title(f"dt = {self.dt_days} hari", fontsize=8,
                     color='white', pad=2)
        ax.tick_params(colors='white', labelsize=5)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False

    # ── Load bodies ───────────────────────────────────────────────────────
    def load_bodies(self, specs: List[BodySpec]):
        self.body_list.clear()
        self._artists.clear()
        self._step         = 0
        self._elapsed_days = 0.0
        self._initialized  = False
        self.done          = False
        self._close_csv()

        for spec in specs:
            body = Body(
                name=spec.name, pos=spec.pos.copy(), vel=spec.vel.copy(),
                mass=spec.mass, radius=spec.radius, color=spec.color,
            )
            self.body_list.append(body)

            size    = float(np.clip(body.radius / 1000, 10, 200))
            scatter = self.ax.scatter([], [], [], c=spec.color, s=size,
                                      label=spec.name, depthshade=False, zorder=5)
            trail,  = self.ax.plot([], [], [], c=spec.color, alpha=0.4,
                                   linewidth=0.6, zorder=4)
            self._artists[id(body)] = {'scatter': scatter, 'trail': trail}

        physics._compute_all_accelerations(self.body_list)
        self._initialized = True
        if self.body_list:
            self.ax.legend(loc='upper left', fontsize=6)

    # ── CSV ───────────────────────────────────────────────────────────────
    def open_csv(self, filepath: str):
        """Buka file CSV dan tulis header. Dipanggil saat simulasi dimulai."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self._csv_file   = open(filepath, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=FIELDNAMES)
        self._csv_writer.writeheader()

    def _close_csv(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None

    def _record_step(self):
        """Tulis state semua benda pada langkah saat ini langsung ke disk."""
        if not self._csv_writer:
            return
        day = round(self._elapsed_days, 6)
        for body in self.body_list:
            pos_au  = body.pos / physics.AU
            vel_aus = body.vel / physics.AUDAY
            speed   = float(np.linalg.norm(body.vel))
            dist    = float(np.linalg.norm(body.pos) / physics.AU)
            self._csv_writer.writerow({
                'Langkah'             : self._step,
                'Hari'                : round(day, 4),
                'dt_hari'             : self.dt_days,
                'Nama'                : body.name,
                'Pos_X_AU'            : round(pos_au[0],  8),
                'Pos_Y_AU'            : round(pos_au[1],  8),
                'Pos_Z_AU'            : round(pos_au[2],  8),
                'Vel_X_AU_per_hari'   : round(vel_aus[0], 8),
                'Vel_Y_AU_per_hari'   : round(vel_aus[1], 8),
                'Vel_Z_AU_per_hari'   : round(vel_aus[2], 8),
                'Kecepatan_m_per_s'   : round(speed,      4),
                'Jarak_dari_Origin_AU': round(dist,        8),
            })

    # ── Maju satu langkah ─────────────────────────────────────────────────
    def step(self) -> bool:
        """
        Maju satu langkah integrasi, catat ke CSV.
        Kembalikan True jika masih berjalan, False jika sudah selesai.
        """
        if self.done or not self.body_list:
            return False
        if self._step >= self._total_steps:
            self._finish()
            return False

        physics.integrate_step(self.body_list, self.dt)
        self._step         += 1
        self._elapsed_days += self.dt_days

        # Rekam setiap langkah
        self._record_step()

        # Update trail visual
        for body in self.body_list:
            body.trail_append(body.pos)

        if self._step >= self._total_steps:
            self._finish()
            return False
        return True

    def _finish(self):
        self.done = True
        self._close_csv()

    # ── Render ────────────────────────────────────────────────────────────
    def draw(self):
        for body in self.body_list:
            arts = self._artists.get(id(body))
            if not arts:
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

    # ── Info ──────────────────────────────────────────────────────────────
    @property
    def progress(self) -> float:
        if self._total_steps == 0:
            return 0.0
        return min(1.0, self._step / self._total_steps)

    @property
    def elapsed_days(self) -> float:
        return self._elapsed_days


# ─────────────────────────────────────────────────────────────────────────────
class ComparisonSimulator:
    """
    Mengelola 11 SingleSim secara bersamaan.
    Setiap sim berjalan sepenuhnya independen dan merekam
    setiap langkah ke CSV masing-masing.
    """

    def __init__(self, axes):
        assert len(axes) == len(DT_DAYS), \
            f"Butuh {len(DT_DAYS)} axes, diberikan {len(axes)}"
        self.sims : List[SingleSim] = [
            SingleSim(dt, ax) for dt, ax in zip(DT_DAYS, axes)
        ]
        self._specs      : List[BodySpec] = []
        self.total_days  : int   = 0
        self.running     : bool  = False
        self.finished    : bool  = False
        self._output_dir : str   = ""

    # ── Setup ─────────────────────────────────────────────────────────────
    def set_bodies(self, specs: List[BodySpec]):
        self._specs = specs
        for sim in self.sims:
            sim.load_bodies(specs)
        self.finished = False

    def configure(self, total_days: int, output_dir: str):
        """
        Siapkan semua sim untuk lari total_days hari.
        total_days dibulatkan ke atas ke kelipatan dt terbesar (100)
        agar semua sim punya titik akhir yang sama persis.
        """
        # Bulatkan ke kelipatan 100 (dt terbesar)
        self.total_days  = ((total_days + 99) // 100) * 100
        self._output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        for sim in self.sims:
            sim.load_bodies(self._specs)                       # reset state fisika
            sim._total_steps = self.total_days // sim.dt_days  # langkah penuh
            sim.done = False
            filepath = os.path.join(output_dir, f"hasil_dt{sim.dt_days}hari.csv")
            sim.open_csv(filepath)

        self.finished = False

    def start(self) -> bool:
        if not self._specs:
            return False
        self.running  = True
        self.finished = False
        return True

    def stop(self):
        self.running = False
        # Tutup semua file CSV yang masih terbuka
        for sim in self.sims:
            sim._close_csv()

    # ── Per-tick step (sequential) ────────────────────────────────────────
    def tick(self):
        """
        Sequential: hanya satu sim yang aktif dalam satu waktu.
        Setelah sim ke-i selesai, baru mulai sim ke-i+1.
        Dipanggil setiap frame animasi.
        """
        if not self.running:
            return

        # Cari sim pertama yang belum selesai
        active = None
        for sim in self.sims:
            if not sim.done:
                active = sim
                break

        if active is None:
            # Semua sim selesai
            self._finish()
            return

        # Maju STEPS_PER_TICK langkah pada sim yang aktif saja
        for _ in range(STEPS_PER_TICK):
            still_running = active.step()
            if not still_running:
                break

    def draw_all(self):
        """Gambar hanya sim yang sedang aktif (sequential mode)."""
        for sim in self.sims:
            if not sim.done:
                sim.draw()
                break  # hanya sim aktif yang perlu di-render

    def _finish(self):
        self.running  = False
        self.finished = True

    # ── Info ──────────────────────────────────────────────────────────────
    @property
    def active_sim(self):
        """Sim yang sedang berjalan (sequential). None jika semua selesai."""
        for sim in self.sims:
            if not sim.done:
                return sim
        return None

    @property
    def active_index(self) -> int:
        """Indeks sim yang sedang aktif (0-based). -1 jika semua selesai."""
        for i, sim in enumerate(self.sims):
            if not sim.done:
                return i
        return -1

    @property
    def progress(self) -> float:
        """Progress keseluruhan: sim selesai / total sim + progress sim aktif."""
        if not self.sims:
            return 0.0
        done_count = sum(1 for s in self.sims if s.done)
        active = self.active_sim
        active_pct = active.progress if active else 0.0
        return (done_count + active_pct) / len(self.sims)

    @property
    def steps_info(self) -> List[dict]:
        return [
            {
                'dt'      : s.dt_days,
                'step'    : s._step,
                'total'   : s._total_steps,
                'hari'    : s.elapsed_days,
                'progress': s.progress,
                'done'    : s.done,
            }
            for s in self.sims
        ]

    @property
    def elapsed_days(self) -> float:
        """Elapsed hari dari sim yang sedang aktif sebagai referensi."""
        active = self.active_sim
        if active:
            return active.elapsed_days
        # Semua selesai — kembalikan dari sim terakhir
        return self.sims[-1].elapsed_days if self.sims else 0.0