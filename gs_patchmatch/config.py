import time
from pathlib import Path

from .util import current_timestamp_filename


class Config:
    def __init__(self, output_path=None, debug_mode=False, debug_cam=None):
        if output_path is None:
            output_path = Path('output') / current_timestamp_filename()

        while output_path.exists():
            time.sleep(1)
            output_path = Path('output') / current_timestamp_filename()

        output_path.mkdir(parents=True)

        self._output_path = output_path
        self.debug_mode = debug_mode
        self.debug_cam = debug_cam

    def output_path(self, path, exist_ok=False):
        res = self._output_path / path
        res.parent.mkdir(parents=True, exist_ok=True)

        if not exist_ok:
            stem = res.stem
            i = 0
            while res.exists():
                i += 1
                res = res.with_stem(f'{stem}_{i}')

        return res
