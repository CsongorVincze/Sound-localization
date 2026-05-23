"""
Rotating source audio pool with background auto-download.

Cycles through all available clips before repeating any.
Triggers a background download when the pool falls below LOW_THRESHOLD.
Thread-safe.
"""
import logging
import random
import threading
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)

DOWNLOAD_BATCH      = 1000   # clips per auto-download run
REUSE_TRIGGER       = 2      # trigger download after each clip has been used this many times
                             # (pool size stays constant during cycling, so we track uses instead)


class SourcePool:
    def __init__(self, sources_dir: Path, download_fn=None):
        """
        sources_dir : directory containing .wav files
        download_fn : callable(n_clips) that downloads n clips into sources_dir
        """
        self._dir         = sources_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._download_fn = download_fn
        self._lock        = threading.Lock()
        self._pool        = deque()
        self._downloading = False
        self._uses        = 0   # total get() calls; triggers download every N×pool_size
        self._reload()

    def _reload(self):
        files = list(self._dir.glob('*.wav'))
        random.shuffle(files)
        with self._lock:
            self._pool = deque(str(f) for f in files)
        log.info(f"Source pool: {len(self._pool)} clips in {self._dir}")

    def _download_more(self):
        log.info(f"Auto-downloading {DOWNLOAD_BATCH} more source clips…")
        try:
            self._download_fn(DOWNLOAD_BATCH)
            self._reload()
        except Exception as e:
            log.error(f"Auto-download failed: {e}")
        with self._lock:
            self._downloading = False

    def _maybe_trigger_download(self):
        with self._lock:
            pool_size = len(self._pool)
            # Trigger once every REUSE_TRIGGER full cycles through the pool.
            # pool size stays constant during cycling, so use a usage counter.
            trigger = (pool_size > 0
                       and self._uses > 0
                       and self._uses % (pool_size * REUSE_TRIGGER) == 0)
            active = self._downloading
        if trigger and not active and self._download_fn:
            with self._lock:
                self._downloading = True
            threading.Thread(target=self._download_more, daemon=True).start()

    def get(self) -> str:
        """Return next clip path, cycling through the pool indefinitely."""
        with self._lock:
            if not self._pool:
                raise RuntimeError(
                    "Source pool is empty and no download function provided.")
            path = self._pool.popleft()
            self._pool.append(path)    # cycle: put back at the end
            self._uses += 1
        self._maybe_trigger_download()
        return path

    def __len__(self) -> int:
        with self._lock:
            return len(self._pool)
