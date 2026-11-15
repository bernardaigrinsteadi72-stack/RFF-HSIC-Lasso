"""Read the distributed JSONL records, including a ZIP containing a GZIP file."""
from contextlib import ExitStack, contextmanager
import gzip
import io
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


@contextmanager
def open_records(path=None):
    if path is None:
        candidates = [ROOT / 'experiment_records.jsonl.gz',
                      ROOT / 'experiment_records.jsonl.zip',
                      ROOT / 'experiment_records.jsonl']
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            candidates = sorted((ROOT / 'experiment_records.jsonl').rglob('*.gz'))
            path = next(iter(candidates), None)
        if path is None:
            raise FileNotFoundError('No experiment_records.jsonl, .gz, or .zip was found.')
    path = Path(path)
    with ExitStack() as stack:
        if path.suffix == '.zip':
            archive = stack.enter_context(ZipFile(path))
            names = [n for n in archive.namelist()
                     if not n.endswith('/') and n.endswith(('.jsonl', '.jsonl.gz'))]
            if len(names) != 1:
                raise ValueError('The records ZIP must contain exactly one JSONL or JSONL.GZ file.')
            stream = stack.enter_context(archive.open(names[0]))
            if names[0].endswith('.gz'):
                stream = stack.enter_context(gzip.GzipFile(fileobj=stream))
            stream = stack.enter_context(io.TextIOWrapper(stream, encoding='utf-8-sig'))
        elif path.suffix == '.gz':
            stream = stack.enter_context(gzip.open(path, 'rt', encoding='utf-8-sig'))
        else:
            stream = stack.enter_context(path.open(encoding='utf-8-sig'))
        yield stream


def records(path=None):
    with open_records(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def dimension_rule(n, d, minimum=10, maximum=80, budget=15000):
    import math
    return min(maximum, max(minimum, math.floor(min(math.sqrt(n), budget / d))))
