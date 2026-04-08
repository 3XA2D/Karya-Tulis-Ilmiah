import numpy as np

G     = 6.67430e-11       # m^3 kg^-1 s^-2
AU    = 1.495978707e11    # m
DAY   = 86400             # s
AUDAY = AU / DAY          # m/s


def _compute_accelerations_vec(pos, masses):
    """
    Fully vectorised O(n^2) pairwise gravity.

    Parameters
    ----------
    pos    : (n, 3) float64 array — positions in metres
    masses : (n,)   float64 array — masses in kg

    Returns
    -------
    accel  : (n, 3) float64 array — accelerations in m/s^2
    """
    # diff[i, j] = pos[j] - pos[i]
    diff    = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]   # (n, n, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)             # (n, n)
    np.fill_diagonal(dist_sq, np.inf)                          # avoid self-force
    inv_dist3 = dist_sq ** -1.5                                # (n, n)  = 1/|r|^3

    # accel[i] = G * sum_j( m[j] * diff[i,j] / |r_ij|^3 )
    accel = G * np.einsum('j,ijn,ij->in', masses, diff, inv_dist3)
    return accel


def integrate_step(body_list, dt):
    """
    Velocity Verlet — operates directly on numpy arrays cached in body objects.

    Steps:
        1. pos  += vel*dt + 0.5*a*dt^2
        2. a_new = f(pos_new)
        3. vel  += 0.5*(a_old + a_new)*dt
    """
    n = len(body_list)
    if n == 0:
        return

    # ── Build contiguous arrays from body state (cheap view, not copy) ──────
    # Using np.empty + in-place fill avoids repeated small allocations
    pos    = np.empty((n, 3), dtype=np.float64)
    vel    = np.empty((n, 3), dtype=np.float64)
    a_old  = np.empty((n, 3), dtype=np.float64)
    masses = np.empty(n,      dtype=np.float64)

    for i, b in enumerate(body_list):
        pos[i]    = b.pos
        vel[i]    = b.vel
        a_old[i]  = b.a
        masses[i] = b.mass

    dt2 = dt * dt

    # ── Step 1: update positions ─────────────────────────────────────────────
    pos += vel * dt + 0.5 * a_old * dt2

    # ── Step 2: new accelerations at updated positions ────────────────────────
    a_new = _compute_accelerations_vec(pos, masses)

    # ── Step 3: update velocities ─────────────────────────────────────────────
    vel += 0.5 * (a_old + a_new) * dt

    # ── Write back to body objects ────────────────────────────────────────────
    for i, b in enumerate(body_list):
        b.pos = pos[i]
        b.vel = vel[i]
        b.a   = a_new[i]


def _compute_all_accelerations(body_list):
    """Initialise body.a before the first integration step."""
    n = len(body_list)
    if n == 0:
        return
    pos    = np.array([b.pos  for b in body_list], dtype=np.float64)
    masses = np.array([b.mass for b in body_list], dtype=np.float64)
    accels = _compute_accelerations_vec(pos, masses)
    for i, b in enumerate(body_list):
        b.a = accels[i]
