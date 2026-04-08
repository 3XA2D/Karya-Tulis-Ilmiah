import numpy as np

TRAIL_LENGTH = 300  # reduced from 500 — shorter trail = less data to plot per frame


class Body:
    """
    Stores state for one gravitational body.
    Trail uses a pre-allocated circular buffer for O(1) append with zero GC pressure.
    """
    __slots__ = ('name', 'pos', 'vel', 'a', 'mass', 'radius', 'color',
                 '_trail_buf', '_trail_head', '_trail_count')

    def __init__(self, name, pos, vel, mass, radius, color):
        self.name   = name
        self.pos    = np.array(pos,   dtype=np.float64)
        self.vel    = np.array(vel,   dtype=np.float64)
        self.a      = np.zeros(3,     dtype=np.float64)
        self.mass   = float(mass)
        self.radius = float(radius)
        self.color  = color

        # Circular buffer: pre-allocated, avoids any list resizing or deque overhead
        self._trail_buf   = np.zeros((TRAIL_LENGTH, 3), dtype=np.float64)
        self._trail_head  = 0   # index of next write position
        self._trail_count = 0   # how many valid points are stored

    # ── Trail helpers ──────────────────────────────────────────────────────
    def trail_append(self, point):
        self._trail_buf[self._trail_head] = point
        self._trail_head = (self._trail_head + 1) % TRAIL_LENGTH
        if self._trail_count < TRAIL_LENGTH:
            self._trail_count += 1

    def trail_as_array(self):
        """Return trail points in chronological order as (n, 3) array."""
        n = self._trail_count
        if n == 0:
            return None
        if n < TRAIL_LENGTH:
            return self._trail_buf[:n]
        # Buffer is full — oldest point is at _trail_head
        idx = self._trail_head
        return np.roll(self._trail_buf, -idx, axis=0)
