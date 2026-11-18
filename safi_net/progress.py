"""Rank-zero terminal progress and optional atomic dashboard snapshots."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

from tqdm.auto import tqdm


class TrainingProgress:
    def __init__(self, stage, total, initial=0, *, rank=0, stream=None):
        self.stage = stage.upper()
        self.enabled = rank == 0
        self.stream = sys.stderr if stream is None else stream
        self.bar = None
        self.fields = {}
        self.early_stop_reported = False
        target = os.environ.get('SAFI_PROGRESS_FILE') if self.enabled else None
        self.progress_path = Path(target).expanduser().resolve() if target else None
        self.snapshot = {'stage': stage.lower(), 'step': int(initial), 'total': int(total),
                         'initial': int(initial), 'phase': 'startup', 'loss': None,
                         'clips_per_second': None, 'memory_bytes': None,
                         'validation_done': None, 'validation_total': None}
        if self.enabled and os.environ.get('SAFI_PROGRESS_ONLY') != '1':
            self.bar = tqdm(total=total, initial=initial, desc=f'Train {self.stage}',
                            unit='step', dynamic_ncols=True, mininterval=1.,
                            file=self.stream, position=0, leave=True)
        self._emit()

    def _emit(self, **changes):
        if not self.enabled:
            return
        self.snapshot.update(changes)
        self.snapshot['updated_at'] = time.time()
        if self.progress_path is None:
            return
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=self.progress_path.parent,
                                             prefix=f'.{self.progress_path.name}.',
                                             suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(self.snapshot, handle, allow_nan=False)
                handle.write('\n')
            os.replace(temporary, self.progress_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _finite(value):
        number = float(value)
        return number if math.isfinite(number) else None

    def _values(self, step, *, loss=None, clips_per_second=None, memory_bytes=None):
        self.snapshot['step'] = int(step)
        if loss is not None:
            self.snapshot['loss'] = self._finite(loss)
            self.fields['loss'] = f'{loss:.5f}'
        if clips_per_second is not None:
            self.snapshot['clips_per_second'] = self._finite(clips_per_second)
            self.fields['clips/s'] = f'{clips_per_second:.1f}'
        if memory_bytes is not None:
            self.snapshot['memory_bytes'] = self._finite(memory_bytes)
        if memory_bytes:
            self.fields['GPU GiB'] = f'{memory_bytes / 2**30:.1f}'
        if self.bar is not None:
            self.bar.set_postfix(self.fields, refresh=False)
            self.bar.update(max(0, step - self.bar.n))

    def update(self, step, *, loss=None, clips_per_second=None, memory_bytes=None):
        if not self.enabled:
            return
        self._values(step, loss=loss, clips_per_second=clips_per_second, memory_bytes=memory_bytes)
        self._emit(phase='training')

    def status(self, record):
        if not self.enabled:
            return
        phase = record['phase']
        if record.get('best_score') is not None:
            self.fields['best'] = f"{record['best_score']:.6f}"
        if 'validation' in record:
            self.fields['val'] = f"{record['validation']['score']:.6f}"
        early_stopping = record.get('early_stopping', {})
        settings = early_stopping.get('settings', {})
        if settings.get('enabled'):
            self.fields['ES'] = f"{early_stopping.get('bad_validations', 0)}/{settings['patience']}"
        else:
            self.fields.pop('ES', None)
        early_stopped = record.get('termination_reason') == 'early_stopping'
        self.fields['state'] = 'early_stopped' if early_stopped else phase
        self._values(record['step'], loss=record.get('loss', {}).get('total'),
                     clips_per_second=record.get('clips_per_second'),
                     memory_bytes=record.get('memory', {}).get('reserved_bytes'))
        changes = {'phase': self.fields['state']}
        if record.get('error') is not None:
            changes['error'] = str(record['error'])
        self._emit(**changes)
        if self.bar is not None:
            self.bar.refresh()
        if record.get('error') and self.bar is not None:
            tqdm.write(f"Stage {self.stage} failed: {record['error']}", file=self.stream)
        if early_stopped and not self.early_stop_reported:
            self.early_stop_reported = True
            if self.bar is not None:
                tqdm.write(
                    f"Stage {self.stage} early stopped at step {record['step']}: "
                    f"{early_stopping.get('bad_validations', 0)} validations without a score decrease "
                    f"of at least {settings.get('min_delta', 0):g} from the early-stopping baseline. "
                    "Best weights remain saved in best.pt.", file=self.stream)

    @contextmanager
    def validation(self):
        if not self.enabled:
            yield None
            return
        self.fields['state'] = 'validating'
        self._emit(phase='validating', validation_done=0, validation_total=0)
        child = None
        if self.bar is not None:
            self.bar.set_postfix(self.fields, refresh=True)
            child = tqdm(total=0, desc=f'Validate {self.stage}', unit='clip',
                         dynamic_ncols=True, mininterval=1., position=1, leave=False,
                         file=self.stream)

        def advance(done, total):
            self._emit(phase='validating', validation_done=int(done), validation_total=int(total))
            if child is not None:
                if child.total != total:
                    child.total = total
                    child.refresh()
                child.update(max(0, done - child.n))

        try:
            yield advance
        finally:
            if child is not None:
                child.close()
            self.fields['state'] = 'training'
            if self.snapshot['phase'] == 'validating':
                self._emit(phase='training')
            if self.bar is not None:
                # Exclude long validation pauses from the recent step-rate estimate.
                self.bar.last_print_t = self.bar._time()
                self.bar.last_print_n = self.bar.n
                self.bar.set_postfix(self.fields, refresh=True)

    def close(self):
        if self.bar is not None:
            self.bar.close()
        terminal = {'completed', 'failed', 'interrupted', 'early_stopped'}
        self._emit(phase=self.snapshot['phase'] if self.snapshot['phase'] in terminal else 'closed')
