import sys
import time
import numpy as np
import sounddevice as sd
import torch
import matplotlib.pyplot as plt

from features import (
    SAMPLE_RATE, FRAME_SAMPLES, T_MAX,
    extract_frame_features, normalize_features
)
from slcnet import SLCnet

# --- Configuration ---
MODEL_PATH    = "best_slcnet_baseline.pth"
BUFFER_SAMPLES = FRAME_SAMPLES * T_MAX   # 73440 - 4.6s ring buffer
HOP_SAMPLES   = int(SAMPLE_RATE * 0.5)  # 8000  - 500ms update cadence
PLOT_WINDOW   = 60.0                     # seconds of history on time-series plot

CLASS_MAP = {
    # ESC-50 targets 0-49
    0: 'dog',           1: 'rooster',       2: 'pig',           3: 'cow',
    4: 'frog',          5: 'cat',           6: 'hen',           7: 'insects',
    8: 'sheep',         9: 'crow',
    10: 'rain',         11: 'sea_waves',    12: 'fire',         13: 'crickets',
    14: 'birds',        15: 'water_drops',  16: 'wind',         17: 'pouring_water',
    18: 'toilet',       19: 'thunderstorm',
    20: 'crying_baby',  21: 'sneezing',     22: 'clapping',     23: 'breathing',
    24: 'coughing',     25: 'footsteps',    26: 'laughing',     27: 'brushing_teeth',
    28: 'snoring',      29: 'drinking',
    30: 'door_knock',   31: 'mouse_click',  32: 'keyboard',     33: 'door_creak',
    34: 'can_opening',  35: 'washing_mach', 36: 'vacuum',       37: 'clock_alarm',
    38: 'clock_tick',   39: 'glass_break',
    40: 'helicopter',   41: 'chainsaw',     42: 'siren',        43: 'car_horn',
    44: 'engine',       45: 'train',        46: 'church_bells', 47: 'airplane',
    48: 'fireworks',    49: 'hand_saw',
}

device = torch.device("cpu")

# --- ReSpeaker hardware DoA (soft-fail if hardware not connected) ---
sys.path.insert(0, 'respeakeres_fileok')
mic_tuning = None
try:
    import usb.core
    import usb.util
    from tuning import Tuning
    _dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if _dev:
        mic_tuning = Tuning(_dev)
        print("ReSpeaker hardware DoA initialized.")
    else:
        print("ReSpeaker USB device not found - hardware DoA disabled.")
except Exception as e:
    print(f"Hardware DoA unavailable: {e}")


def _build_plot():
    fig = plt.figure(figsize=(13, 5))
    fig.suptitle("SLCnet Live DoA - Model vs ReSpeaker Hardware")

    ax_ts  = fig.add_subplot(1, 2, 1)
    ax_pol = fig.add_subplot(1, 2, 2, projection='polar')

    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("DoA (deg)")
    ax_ts.set_ylim(0, 365)
    ax_ts.set_yticks(range(0, 361, 45))
    ax_ts.grid(True, alpha=0.4)
    ax_ts.set_title("DoA over time")
    line_model, = ax_ts.plot([], [], color='steelblue',  lw=1.5, label='SLCnet model')
    line_hw,    = ax_ts.plot([], [], color='tomato', lw=1.5, ls='--', label='HW chip')
    ax_ts.legend(loc='upper right')

    ax_pol.set_theta_zero_location('N')
    ax_pol.set_theta_direction(-1)   # clockwise = compass convention
    ax_pol.set_ylim(0, 1.1)
    ax_pol.set_yticks([])
    ax_pol.set_title("Current angle")

    return fig, ax_ts, ax_pol, line_model, line_hw


def _update_plot(fig, ax_ts, ax_pol, line_model, line_hw,
                 log_t, log_model, log_hw):
    if not log_t:
        return

    t   = np.array(log_t)
    m   = np.array(log_model)
    hw  = np.array([v if v is not None else np.nan for v in log_hw])

    # Time-series: rolling window
    t_min = max(0.0, t[-1] - PLOT_WINDOW)
    mask  = t >= t_min
    line_model.set_data(t[mask], m[mask])
    hw_mask = mask & ~np.isnan(hw)
    line_hw.set_data(t[hw_mask], hw[hw_mask])
    ax_ts.set_xlim(t_min, t[-1] + 1)

    # Polar compass: clear and redraw needles
    ax_pol.clear()
    ax_pol.set_theta_zero_location('N')
    ax_pol.set_theta_direction(-1)
    ax_pol.set_ylim(0, 1.1)
    ax_pol.set_yticks([])

    # Faded trail of recent model predictions (last 10 updates)
    trail_n = min(10, len(m))
    for i, ang in enumerate(m[-trail_n:]):
        alpha = (i + 1) / trail_n * 0.35
        ax_pol.plot(np.deg2rad(ang), 0.75, 'o',
                    color='steelblue', ms=6, alpha=alpha)

    # Current model needle
    r_model = np.deg2rad(m[-1])
    ax_pol.plot([r_model, r_model], [0, 1.0],
                color='steelblue', lw=3, label=f'Model {m[-1]:.0f}°')
    ax_pol.plot(r_model, 1.0, 'o', color='steelblue', ms=9)

    # Hardware needle (if available)
    valid_hw = ~np.isnan(hw)
    if valid_hw.any():
        r_hw = np.deg2rad(hw[valid_hw][-1])
        ax_pol.plot([r_hw, r_hw], [0, 1.0],
                    color='tomato', lw=3, ls='--', label=f'HW {hw[valid_hw][-1]:.0f}°')
        ax_pol.plot(r_hw, 1.0, 's', color='tomato', ms=9)

    ax_pol.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.canvas.draw_idle()


def main():
    print("Loading SLCnet model...")
    try:
        model = SLCnet(input_dim=618, num_classes=50)
        sd_ = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        sd_ = {k[7:] if k.startswith('module.') else k: v for k, v in sd_.items()}
        model.load_state_dict(sd_)
        model.eval()
        print("Model loaded.")
    except FileNotFoundError:
        print(f"Error: '{MODEL_PATH}' not found.")
        sys.exit(1)

    ring_buffer = np.zeros((4, BUFFER_SAMPLES), dtype=np.float32)

    # Shared logs - list.append is GIL-atomic, safe across callback thread and main thread
    log_t     = []
    log_model = []
    log_hw    = []
    t0        = time.time()

    def audio_callback(indata, frames, time_info, status):
        new_audio = indata[:, :4].T.astype(np.float32)
        peak = np.abs(new_audio).max()
        if peak > 0:
            new_audio /= peak

        ring_buffer[:] = np.roll(ring_buffer, -frames, axis=1)
        ring_buffer[:, -frames:] = new_audio

        wav = torch.from_numpy(ring_buffer)
        feats = [
            extract_frame_features(
                wav[:, t * FRAME_SAMPLES:(t + 1) * FRAME_SAMPLES].unsqueeze(0)
            )
            for t in range(T_MAX)
        ]
        feat = normalize_features(torch.stack(feats).unsqueeze(0))

        with torch.no_grad():
            doa_pred, sec_pred = model(feat)

        model_angle = doa_pred[0].argmax().item() + 1
        class_idx   = sec_pred[0].argmax().item()
        confidence  = sec_pred[0][class_idx].item() * 100

        hw_angle = None
        if mic_tuning:
            try:
                hw_angle = mic_tuning.direction
            except Exception:
                pass

        hw_str = f"{hw_angle:03d} deg" if hw_angle is not None else "N/A"
        print(
            f"\rEvent: [{CLASS_MAP.get(class_idx,'?').upper():<10}] ({confidence:05.1f}%)"
            f"   |   Model: {model_angle:03d} deg   HW: {hw_str}     ",
            end="", flush=True
        )

        log_t.append(time.time() - t0)
        log_model.append(model_angle)
        log_hw.append(hw_angle)

    # --- Live plot ---
    plt.ion()
    fig, ax_ts, ax_pol, line_model, line_hw = _build_plot()
    plt.tight_layout()
    plt.show(block=False)

    print(f"\n[!] Buffer: {BUFFER_SAMPLES/SAMPLE_RATE:.1f}s  |  Update: {HOP_SAMPLES/SAMPLE_RATE:.1f}s")
    print("[!] Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=6,
                            blocksize=HOP_SAMPLES, callback=audio_callback):
            while True:
                _update_plot(fig, ax_ts, ax_pol, line_model, line_hw,
                             log_t, log_model, log_hw)
                plt.pause(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\n[X] Stream error: {e}")
    finally:
        plt.ioff()
        if log_t:
            _update_plot(fig, ax_ts, ax_pol, line_model, line_hw,
                         log_t, log_model, log_hw)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
