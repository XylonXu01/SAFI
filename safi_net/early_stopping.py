"""Validation-score early stopping, independent of best-weight selection."""
from collections.abc import Mapping
from copy import deepcopy
import math


def normalize_early_stopping(config=None, stage='b'):
    if stage not in ('a', 'b', 'c', 'joint'):
        raise ValueError('Early stopping requires stage a/b/c')
    defaults = {'enabled': True, 'patience': 5 if stage in ('a', 'joint') else 3,
                'min_delta': 1e-4, 'min_steps': 10000 if stage == 'a' else 2000 if stage == 'joint' else 1000}
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('early_stopping must be a mapping')
    supplied = dict(config or {})
    if set(supplied) - set(defaults):
        raise ValueError(f'Unknown early_stopping settings: {sorted(set(supplied)-set(defaults))}')
    result = {**defaults, **supplied}
    if not isinstance(result['enabled'], bool):
        raise ValueError('early_stopping.enabled must be boolean')
    for field, minimum in (('patience', 1), ('min_steps', 0)):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'early_stopping.{field} must be an integer >= {minimum}')
    delta = result['min_delta']
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise ValueError('early_stopping.min_delta must be finite and nonnegative')
    result['min_delta'] = float(delta)
    return result


class EarlyStopping:
    def __init__(self, settings, validate_every, *, saved_state=None, resume_step=None, best_score=None):
        self.settings = normalize_early_stopping(settings)
        if isinstance(validate_every, bool) or not isinstance(validate_every, int) or validate_every < 1:
            raise ValueError('validate_every must be positive')
        self.validate_every = validate_every
        self.reference_score = None if best_score is None else float(best_score)
        if self.reference_score is not None and not math.isfinite(self.reference_score):
            raise ValueError('Early stopping baseline must be finite')
        self.bad_validations = 0
        self.last_validation_step = resume_step
        self.stopped = False
        self.reset_reason = 'legacy_checkpoint' if resume_step is not None else 'new_stage'
        if saved_state is None:
            return
        if not isinstance(saved_state, dict) or saved_state.get('version') != 1:
            raise ValueError('Unsupported early stopping state')
        if (saved_state.get('settings') != self.settings
                or saved_state.get('validate_every') != validate_every):
            self.reset_reason = 'policy_changed'
            return
        reference = saved_state.get('reference_score')
        bad = saved_state.get('bad_validations')
        last = saved_state.get('last_validation_step')
        stopped = saved_state.get('stopped')
        if (reference is not None and (isinstance(reference, bool)
                or not isinstance(reference, (int, float)) or not math.isfinite(reference))):
            raise ValueError('Invalid early stopping reference score')
        if isinstance(bad, bool) or not isinstance(bad, int) or bad < 0:
            raise ValueError('Invalid early stopping patience counter')
        if last is not None and (isinstance(last, bool) or not isinstance(last, int) or last < 0):
            raise ValueError('Invalid early stopping validation step')
        if resume_step is not None and last is not None and last > resume_step:
            raise ValueError('Early stopping validation step exceeds checkpoint step')
        if not isinstance(stopped, bool):
            raise ValueError('Invalid early stopping stopped flag')
        if stopped != (self.settings['enabled'] and bad >= self.settings['patience']):
            raise ValueError('Early stopping counter and stopped flag disagree')
        self.reference_score = reference
        self.bad_validations = bad
        self.last_validation_step = last
        self.stopped = stopped
        self.reset_reason = 'restored'

    def update(self, step, score):
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError('Validation step must be a nonnegative integer')
        score = float(score)
        if not math.isfinite(score):
            raise ValueError('Validation score must be finite for early stopping and best selection')
        # Startup validation on resume must never consume patience twice.
        if self.last_validation_step is not None and step <= self.last_validation_step:
            return self.stopped
        if self.stopped:
            return True
        self.last_validation_step = step
        if (not self.settings['enabled'] or step <= self.settings['min_steps']
                or self.reference_score is None):
            self.reference_score = score if self.reference_score is None else min(self.reference_score, score)
            self.bad_validations = 0
            return False
        improvement = self.reference_score - score
        if score < self.reference_score and improvement >= self.settings['min_delta']:
            self.reference_score = score
            self.bad_validations = 0
        else:
            self.bad_validations += 1
        self.stopped = self.bad_validations >= self.settings['patience']
        return self.stopped

    def state_dict(self):
        return {'version': 1, 'settings': deepcopy(self.settings),
                'validate_every': self.validate_every, 'reference_score': self.reference_score,
                'bad_validations': self.bad_validations, 'last_validation_step': self.last_validation_step,
                'stopped': self.stopped, 'reset_reason': self.reset_reason}


def early_stopped_checkpoint(checkpoint):
    return bool(checkpoint.get('completed') and checkpoint.get('termination_reason') == 'early_stopping')
