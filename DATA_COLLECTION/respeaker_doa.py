"""
ReSpeaker USB v2.0 chip DoA reader.

The XVF3000 chip estimates Direction of Arrival continuously and exposes
the result as parameter DOAANGLE (id=21) via USB vendor control transfers.
Range: 0-359 degrees, read-only.

Windows setup: install libusb-win32 for the ReSpeaker via Zadig tool.
Requires: pip install pyusb
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'respeakeres_fileok'))

log = logging.getLogger(__name__)

try:
    import usb.core
    from tuning import Tuning
    _USB_OK = True
except ImportError:
    _USB_OK = False
    log.warning("pyusb not installed — chip DoA reading disabled.")

VID = 0x2886
PID = 0x0018


class ReSpeakerDoA:
    def __init__(self):
        self._tuning  = None
        self.available = False
        if _USB_OK:
            self._connect()

    def _connect(self):
        try:
            dev = usb.core.find(idVendor=VID, idProduct=PID)
            if dev is None:
                log.warning("ReSpeaker USB device not found (0x2886:0x0018). "
                            "Is libusb-win32 installed via Zadig?")
                return
            self._tuning  = Tuning(dev)
            self.available = True
            log.info("ReSpeaker DoA reader connected.")
        except Exception as e:
            log.warning(f"ReSpeaker DoA init failed: {e}")

    def read_median(self, n: int = 3, interval: float = 0.15):
        """
        Read chip DoA n times and return the circular median, or None on failure.

        Circular median: the reading that minimises the sum of circular distances
        to all others.  Handles the 0°/359° wraparound correctly (e.g. readings
        [355, 358, 2] → 358°, not a spurious mid-range value).
        """
        if not self.available:
            return None
        readings = []
        for _ in range(n):
            try:
                readings.append(self._tuning.direction)
                if interval > 0:
                    time.sleep(interval)
            except Exception as e:
                log.warning(f"DoA read error: {e}")
                self.available = False
                return None

        def _circ_dist(a, b):
            d = abs(a - b) % 360
            return d if d <= 180 else 360 - d

        best = min(readings,
                   key=lambda c: sum(_circ_dist(c, r) for r in readings))
        return best

    def close(self):
        if self._tuning:
            try:
                self._tuning.close()
            except Exception:
                pass
