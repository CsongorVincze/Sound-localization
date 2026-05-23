"""
Motor diagnostic tool.

Auto-tests each motor individually first, then enters interactive mode.

Interactive commands:
    fl+ / fl-          step FL 200 steps forward / backward
    fr+ / fr-          same for FR
    bl+ / bl-          same for BL
    br+ / br-          same for BR
    forward            run all forward (default 300 mm)
    backward           run all backward
    left               tank-turn left 90 deg
    right              tank-turn right 90 deg
    go <angle>         go_towards at given DoA angle
    stop               interrupt current movement
    q                  quit
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from robot import Robot, MOTOR_PINS, STEP_DELAY

_SYM = {+1: '>>>',  -1: '<<<',  0: '---'}
_LABELS = {
    (+1, +1, +1, +1): 'FORWARD',
    (-1, -1, -1, -1): 'BACKWARD',
    (+1, -1, +1, -1): 'TURN RIGHT',
    (-1, +1, -1, +1): 'TURN LEFT',
}


def _show(fl, fr, bl, br, action=''):
    print(f"  FL:{_SYM[fl]}  FR:{_SYM[fr]}  BL:{_SYM[bl]}  BR:{_SYM[br]}  [{action}]")


def main():
    try:
        robot = Robot()
        print("Robot initialized.\n")
    except Exception as e:
        print(f"Robot init failed: {e}")
        sys.exit(1)

    # ── Patch _run so every movement prints a status line ─────────────────── #
    _orig_run = robot._run

    def _patched_run(fl, fr, bl, br, n_steps):
        action = _LABELS.get((fl, fr, bl, br), f'fl={fl} fr={fr} bl={bl} br={br}')
        _show(fl, fr, bl, br, action)
        _orig_run(fl, fr, bl, br, n_steps)
        _show(0, 0, 0, 0, 'STOPPED')

    robot._run = _patched_run

    # ── Auto-test: each motor individually ────────────────────────────────── #
    print("=== AUTO TEST: stepping each motor 200 steps forward then back ===\n")
    for name in ['FL', 'FR', 'BL', 'BR']:
        print(f"  Testing {name} ...", end='  ', flush=True)
        for _ in range(200):
            robot._motors[name].step(+1)
            time.sleep(STEP_DELAY)
        time.sleep(0.3)
        for _ in range(200):
            robot._motors[name].step(-1)
            time.sleep(STEP_DELAY)
        robot._motors[name].release()
        ans = input(f"did {name} move? [y/n] ").strip().lower()
        if ans != 'y':
            print(f"  !! {name} did not move — check wiring on pins {MOTOR_PINS[name]}")

    # ── Interactive mode ───────────────────────────────────────────────────── #
    print("\n=== INTERACTIVE MODE ===")
    print("Commands: forward | backward | left | right | go <angle>")
    print("          fl+/fl-  fr+/fr-  bl+/bl-  br+/br-  (200 steps each)")
    print("          stop | q\n")

    _move_thread = [None]

    def run(fn):
        if _move_thread[0] and _move_thread[0].is_alive():
            robot.stop()
            _move_thread[0].join(timeout=1.0)
        t = threading.Thread(target=fn, daemon=True)
        _move_thread[0] = t
        t.start()

    def single(name, direction):
        _show(
            direction if name == 'FL' else 0,
            direction if name == 'FR' else 0,
            direction if name == 'BL' else 0,
            direction if name == 'BR' else 0,
            f'{name} {"FWD" if direction > 0 else "BWD"}',
        )
        for _ in range(200):
            if robot._stop.is_set():
                break
            robot._motors[name].step(direction)
            time.sleep(STEP_DELAY)
        robot._motors[name].release()
        _show(0, 0, 0, 0, 'STOPPED')

    while True:
        try:
            cmd = input('> ').strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == 'q':
            break
        elif cmd == 'stop':
            robot.stop()
        elif cmd == 'forward':
            run(robot.forward)
        elif cmd == 'backward':
            run(robot.backward)
        elif cmd == 'left':
            run(robot.turn_left)
        elif cmd == 'right':
            run(robot.turn_right)
        elif cmd.startswith('go '):
            try:
                angle = float(cmd.split()[1])
                run(lambda a=angle: robot.go_towards(a))
            except (ValueError, IndexError):
                print("  usage: go <degrees>")
        elif len(cmd) == 3 and cmd[:2] in ('fl', 'fr', 'bl', 'br') and cmd[2] in '+-':
            name = cmd[:2].upper()
            d    = +1 if cmd[2] == '+' else -1
            run(lambda n=name, dv=d: single(n, dv))
        else:
            print("  unknown command")

    robot.shutdown()
    print("Done.")


if __name__ == '__main__':
    main()
