"""
DA_output/<case>/run_summary.json (review 469): a small record of every job step of a case, kept with the outputs (not
in logs/<case>/, so it survives the deletion of the logs). One entry per step (DA, collect_DA, ...), the latest job of
that step replacing the earlier one:

    settings snapshot (path, md5 of DA_setting.json / perturbation.json / Config_ReWaterGAP.json of the run copy),
    git commit of the code (+ dirty flag), observation file (path, md5), ensemble size, host / job, start / end time,
    exit status, final filter statistics if available (chi^2 and innovation RMS from adaptive_inflation_log.csv),
    output files (md5 of the small files; count and size of the member dailies / yearly files), rank-log folder.

Written by one rank (the summary is read-modify-written under an exclusive lock and replaced atomically, so a
failing rank can add its status at the same time). Writing it never stops a run: an error is printed and ignored.
"""
import os
import sys
import json
import socket
import hashlib
from datetime import datetime
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
BIG = 256 * 1024 ** 2                    # files above this size are listed with size only (no md5)


def md5(path, chunk=16 * 1024 ** 2):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def git_state(root=CODE_ROOT):
    """(commit, dirty) of the code tree; read-only git calls (no index refresh), (None, None) without git"""
    import subprocess
    try:
        env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
        c = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                           timeout=30, env=env)
        if c.returncode != 0:
            return None, None
        d = subprocess.run(['git', '--no-optional-locks', '-C', str(root), 'status', '--porcelain',
                            '--untracked-files=no'], capture_output=True, text=True, timeout=60, env=env)
        return c.stdout.strip(), (bool(d.stdout.strip()) if d.returncode == 0 else None)
    except Exception:
        return None, None


def settings_snapshot(setting_dir):
    d = Path(setting_dir)
    return dict(path=str(d), md5={fn.name: md5(fn) for fn in sorted(d.glob('*.json')) if fn.is_file()})


def filter_stats(da_case_dir):
    """chi^2 = d^2 / (sd_fc^2 + sigma^2) and the innovation RMS over all windows and sub-basins of the filter's
    adaptive_inflation_log.csv (None if the file is absent, e.g. without adaptive inflation)"""
    import csv
    import math
    fn = Path(da_case_dir) / 'adaptive_inflation_log.csv'
    if not fn.is_file():
        return None
    chi, d2, last = [], [], None
    with open(fn) as fh:
        for r in csv.DictReader(fh):
            try:
                d, s, o = float(r['innovation_mm']), float(r['spread_forecast_mm']), float(r['obs_error_mm'])
            except (KeyError, ValueError):
                continue
            if all(math.isfinite(v) for v in (d, s, o)) and s * s + o * o > 0:
                chi.append(d * d / (s * s + o * o))
                d2.append(d * d)
                last = r.get('date', last)
    if not chi:
        return None
    chi_s = sorted(chi)
    n = len(chi_s)
    med = chi_s[n // 2] if n % 2 else 0.5 * (chi_s[n // 2 - 1] + chi_s[n // 2])
    return dict(source=str(fn), n=n, last_window=last, chi2_mean=sum(chi) / n, chi2_median=med,
                innovation_rms_mm=math.sqrt(sum(d2) / n))


def list_outputs(folder, pattern='*', recursive=True):
    """{relative path: md5 or {'bytes': n}} of the files in folder; member dailies are summarised by count / size"""
    folder = Path(folder)
    out, daily_n, daily_b = {}, 0, 0
    if not folder.is_dir():
        return dict(files=out)
    it = folder.rglob(pattern) if recursive else folder.glob(pattern)
    for f in sorted(it):
        if not f.is_file():
            continue
        rel = str(f.relative_to(folder))
        if f.name in ('run_summary.json', '.run_summary.lock') or f.name.startswith('run_summary.json.tmp'):
            continue
        if f.name.startswith('daily_output_') or f.name.startswith('restartwatergap_'):
            daily_n += 1
            daily_b += f.stat().st_size
            continue
        sz = f.stat().st_size
        out[rel] = md5(f) if sz <= BIG else dict(bytes=sz)
    return dict(folder=str(folder), files=out, daily_and_restart_files=dict(count=daily_n, bytes=daily_b))


def update(case, data_root, step, **fields):
    """merge `fields` into run_summary.json -> steps -> <step> (exclusive lock, atomic replace); never raises"""
    try:
        import fcntl
        d = Path(data_root) / 'DA_output' / case
        d.mkdir(parents=True, exist_ok=True)
        fn = d / 'run_summary.json'
        with open(d / '.run_summary.lock', 'a') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                s = json.load(open(fn)) if fn.is_file() else {}
            except Exception:
                s = {}
            s.setdefault('case', case)
            s.setdefault('steps', {})
            if fields.pop('_new_job', False):
                s['steps'][step] = {}
            s['steps'].setdefault(step, {}).update(fields)
            s['updated'] = datetime.now().astimezone().isoformat(timespec='seconds')
            tmp = fn.with_name(fn.name + '.tmp%d' % os.getpid())
            with open(tmp, 'w') as fh:
                json.dump(s, fh, indent=1, default=str)
            os.replace(tmp, fn)
            fcntl.flock(lk, fcntl.LOCK_UN)
        return fn
    except Exception as err:
        print('[run_summary] not written (%s: %s)' % (type(err).__name__, err), file=sys.__stderr__, flush=True)
        return None


def start(case, data_root, step, setting_dir=None, ens=None, period=None, log_dir=None):
    """the start record of a job step (replaces the earlier record of that step)"""
    from src_FlowControl import error_log
    commit, dirty = git_state()
    f = dict(_new_job=True, status='running', start=datetime.now().astimezone().isoformat(timespec='seconds'),
             end=None, host=socket.gethostname(), job=error_log._job(), pid=os.getpid(),
             code=dict(root=str(CODE_ROOT), git_commit=commit, git_dirty=dirty),
             ensemble_size=ens, period=period, logs=None if log_dir is None else str(log_dir))
    if setting_dir is not None:
        try:
            f['settings'] = settings_snapshot(setting_dir)
        except Exception as err:
            f['settings'] = dict(path=str(setting_dir), error=str(err))
    return update(case, data_root, step, **f)


def finish(case, data_root, step, status='ok', **fields):
    return update(case, data_root, step, status=status,
                  end=datetime.now().astimezone().isoformat(timespec='seconds'), **fields)
