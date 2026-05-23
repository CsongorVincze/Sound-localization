"""
28BYJ-48 stepper control for 4-wheel differential robot on Raspberry Pi.

Wiring (BCM pin numbers):
    FL (front-left)  ULN2003: IN1=5  IN2=6  IN3=13 IN4=19
    FR (front-right) ULN2003: IN1=17 IN2=18 IN3=22 IN4=23
    BL (back-left)   ULN2003: IN1=12 IN2=16 IN3=20 IN4=21
    BR (back-right)  ULN2003: IN1=24 IN2=25 IN3=26 IN4=27

If the robot moves backward on 'forward', flip the sign of all directions in forward().
If it turns the wrong way, flip the signs in turn_left() / turn_right().

DOA calibration:
    Stand directly in front of the robot, note the ReSpeaker DoA reading,
    set DOA_OFFSET to that value. Afterwards DoA 0 = user is in front.

Install dependency: pip install lgpio
"""

import math
import threading
import time

try:
    import lgpio
    _HAS_GPIO = True
except ImportError:
    _HAS_GPIO = False


# ── Pin assignment ─────────────────────────────────────────────────────────── #
MOTOR_PINS = {
    'FL': (5,  6,  13, 19),
    'FR': (17, 18, 22, 23),
    'BL': (12, 16, 20, 21),
    'BR': (24, 25, 26, 27),
}

# ── Calibration — measure your build and update these ─────────────────────── #
WHEEL_DIAMETER_MM = 62      # outer diameter of drive wheel
WHEELBASE_MM      = 140     # left-to-right wheel centre distance (track width)
STEPS_PER_REV     = 4076    # 28BYJ-48 half-step, gear ratio 63.68:1
STEP_DELAY        = 0.002   # 2 ms/step → ~25 mm/s with 65 mm wheel

# Stand in front of robot, read ReSpeaker DoA, put that value here
DOA_OFFSET = 0  # degrees

# ── Half-step sequence (8 phases) ─────────────────────────────────────────── #
_HALF_STEP = [
    (1, 0, 0, 0),
    (1, 1, 0, 0),
    (0, 1, 0, 0),
    (0, 1, 1, 0),
    (0, 0, 1, 0),
    (0, 0, 1, 1),
    (0, 0, 0, 1),
    (1, 0, 0, 1),
]


class _Stepper:
    def __init__(self, h, pins: tuple):
        self._h     = h
        self._pins  = pins
        self._phase = 0

    def step(self, direction: int):
        self._phase = (self._phase + direction) % 8
        for pin, val in zip(self._pins, _HALF_STEP[self._phase]):
            lgpio.gpio_write(self._h, pin, val)

    def release(self):
        for pin in self._pins:
            lgpio.gpio_write(self._h, pin, 0)


class Robot:
    def __init__(self):
        if not _HAS_GPIO:
            raise RuntimeError("lgpio not found — run: pip install lgpio")
        self._h = lgpio.gpiochip_open(0)
        for pins in MOTOR_PINS.values():
            for pin in pins:
                lgpio.gpio_claim_output(self._h, pin, 0)
        self._motors = {name: _Stepper(self._h, pins) for name, pins in MOTOR_PINS.items()}
        self._stop   = threading.Event()

    # ── internals ──────────────────────────────────────────────────────────── #

    def _mm_to_steps(self, mm: float) -> int:
        return round(abs(mm) / (math.pi * WHEEL_DIAMETER_MM) * STEPS_PER_REV)

    def _deg_to_steps(self, degrees: float) -> int:
        arc = math.pi * WHEELBASE_MM * abs(degrees) / 360.0
        return self._mm_to_steps(arc)

    def _run(self, fl: int, fr: int, bl: int, br: int, n_steps: int):
        """Step all 4 motors in lockstep. Stops early if stop() is called."""
        self._stop.clear()
        dirs = {'FL': fl, 'FR': fr, 'BL': bl, 'BR': br}
        for _ in range(n_steps):
            if self._stop.is_set():
                break
            for name, d in dirs.items():
                self._motors[name].step(d)
            time.sleep(STEP_DELAY)
        self._deenergise()

    def _deenergise(self):
        for m in self._motors.values():
            m.release()

    # ── public commands ────────────────────────────────────────────────────── #

    def forward(self, mm: float = 300):
        self._run(+1, +1, +1, +1, self._mm_to_steps(mm))

    def backward(self, mm: float = 300):
        self._run(-1, -1, -1, -1, self._mm_to_steps(mm))

    def turn_left(self, degrees: float = 90):
        self._run(-1, +1, -1, +1, self._deg_to_steps(degrees))

    def turn_right(self, degrees: float = 90):
        self._run(+1, -1, +1, -1, self._deg_to_steps(degrees))

    def go_towards(self, raw_doa: float, drive_mm: float = 400):
        """Turn to face the speaker (shortest path), then drive forward."""
        adjusted = (raw_doa - DOA_OFFSET) % 360
        angle = adjusted if adjusted <= 180 else adjusted - 360  # → (-180, 180]
        if angle > 1:
            self.turn_right(angle)
        elif angle < -1:
            self.turn_left(-angle)
        if not self._stop.is_set():
            self.forward(drive_mm)

    def stop(self):
        """Interrupt any ongoing movement immediately."""
        self._stop.set()

    def shutdown(self):
        self.stop()
        time.sleep(0.05)
        self._deenergise()
        lgpio.gpiochip_close(self._h)
