"""JSON-backed evaluation-result cache shared by the eval tools.

Every expensive evaluation stores its numeric results in
``eval_cache.json`` next to the evaluated checkpoint (or ONNX file), keyed
by tool, checkpoint signature (name/size/mtime) and the evaluation
parameters. Re-running the same command — e.g. only to switch between the
plain and ``--markdown`` table formats — reuses the stored numbers and
skips feature extraction entirely; pass ``--recompute`` to force a fresh
run (the cache entry is then overwritten). Freshly computed results are
also appended human-readably to ``eval_log.txt`` in the same directory,
building a timestamped evaluation history per run.
"""

import json
import os
import time


def checkpoint_signature(path):
    """Identity of the evaluated file: name, size, and mtime.

    The best-checkpoint naming already embeds epoch and mAP; size/mtime
    guard against reused names such as checkpoint_last.pth.
    """
    stat = os.stat(path)
    return '{}:{}:{}'.format(
        os.path.basename(path), stat.st_size, int(stat.st_mtime))


class EvalCache:
    def __init__(self, anchor_path, enabled=True):
        directory = os.path.dirname(os.path.abspath(anchor_path))
        self.path = os.path.join(directory, 'eval_cache.json')
        self.log_path = os.path.join(directory, 'eval_log.txt')
        self.enabled = enabled
        self.data = {}
        if os.path.isfile(self.path):
            try:
                with open(self.path) as handle:
                    self.data = json.load(handle)
            except (json.JSONDecodeError, OSError):
                self.data = {}

    @staticmethod
    def _serialize(key):
        return json.dumps(key, sort_keys=True)

    def get(self, key):
        if not self.enabled:  # --recompute: read nothing, still overwrite below
            return None
        entry = self.data.get(self._serialize(key))
        return entry['results'] if entry else None

    def put(self, key, results, log_lines=()):
        self.data[self._serialize(key)] = {
            'results': results,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as handle:
            json.dump(self.data, handle, sort_keys=True, indent=1)
        os.replace(tmp, self.path)
        if log_lines:
            with open(self.log_path, 'a') as handle:
                handle.write('\n[{}] {}\n'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S'), self._serialize(key)))
                for line in log_lines:
                    handle.write(line + '\n')
