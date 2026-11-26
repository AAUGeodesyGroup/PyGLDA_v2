"""
Run-time logs of PyGLDA v2: everything goes to the data root, nothing into the code folder.

    <data root>/logs/
        errors_YYYY-MM.log                      every error of every case / job / rank, appended, one file per month
        <case>/<step>_<YYYYmmdd-HHMMSS>[_rank<k>].log   the per-case logs (MPI rank logs of the stages); several jobs
                                                of a case share the folder, the time stamp tells them apart

<data root> is the data root of the run (RDA.external_data_path, the folder that ${DATA_ROOT} of the settings resolves
to; without it the environment variable PYGLDA_DATA). Retention: delete logs/<case>/ once the case is verified and
DA_output/<case>/run_summary.json exists (keep the logs of failed runs until the cause is understood); the monthly
errors files are kept.

Use (one line at the start of every runner / entry script; calling it again only changes case / step):

    from src_FlowControl import error_log
    error_log.install(case, step, data_root)

install() adds, once per process:
  - sys.excepthook and threading.excepthook: an uncaught exception -> one entry in errors_YYYY-MM.log, then the
    previous hook runs as before (traceback on stderr, exit code unchanged);
  - one logging handler on the root logger for level ERROR and above (e.g. WaterGAP's config_logger errors);
    warnings stay in the case log / terminal only;
  - PYGLDA_LOG_DIR=<data root>/logs/<case> for WaterGAP's own log files (watergap_logger.py; before, they were
    written into the working directory src_demo/).
MPI stages call report_exception() in their `except` block before MPI.Abort (src_FlowControl.Regional_DA); scripts
can wrap their main part in `with error_log.catch():` so that sys.exit('message') / SystemExit(<non-zero>) is
recorded too; run_step() runs a subprocess and records a non-zero exit with the last 20 lines of its stderr.

Entry format (ISO 8601 time with the UTC offset; written under fcntl.flock, one write per entry, so ranks and jobs
never interleave):

    2026-11-26T15:42:07.123+01:00 | ERROR | case=<case> | step=<step> | host=<host> job=<job-id> rank=<k> | <Type>: <message>
      traceback / stderr lines (indented)
      → case log: logs/<case>/<file>.log

The module never masks the original error: it re-raises, keeps the exit code, and if the errors file cannot be
written it prints the entry and the reason to stderr and carries on.
"""
import os
import sys
import socket
import threading
import traceback
import logging
from datetime import datetime
from pathlib import Path

ERRORS_PREFIX = 'errors_'
_state = dict(case=None, step=None, data_root=None, case_log=None, installed=False)


# ------------------------------------------------------------------------------------------------ paths
def data_root():
    """data root of the run: the one given to install(), else $PYGLDA_DATA, else None"""
    return _state['data_root'] or os.environ.get('PYGLDA_DATA') or None


def log_root(root=None):
    """<data root>/logs (created)"""
    root = root or data_root()
    if not root:
        raise RuntimeError('error_log: no data root (call error_log.install(case, step, data_root) or set PYGLDA_DATA)')
    p = Path(root) / 'logs'
    p.mkdir(parents=True, exist_ok=True)
    return p


def case_log_dir(case=None, root=None):
    """<data root>/logs/<case> (created)"""
    p = log_root(root) / str(case or _state['case'] or 'no_case')
    p.mkdir(parents=True, exist_ok=True)
    return p


def stamp(comm=None):
    """YYYYmmdd-HHMMSS of this job step; with an MPI communicator rank 0's stamp is broadcast (all ranks must call)"""
    s = datetime.now().strftime('%Y%m%d-%H%M%S')
    if comm is not None:
        s = comm.bcast(s, root=0)
    return s


def case_log_path(step, rank=None, comm=None, case=None):
    """<data root>/logs/<case>/<step>_<YYYYmmdd-HHMMSS>[_rank<k>].log (collective if comm is given)"""
    name = '%s_%s%s.log' % (step, stamp(comm), '' if rank is None else '_rank%d' % rank)
    return case_log_dir(case) / name


def redirect_rank(step, rank, comm, terminal_rank, case=None):
    """
    MPI stages: every rank except `terminal_rank` writes its stdout and stderr (line-buffered, so the traceback is in
    the file even if MPI.Abort follows) to logs/<case>/<step>_<stamp>_rank<k>.log; the terminal rank keeps the terminal.
    Collective (the stamp of rank 0 is broadcast). Returns the log path of this rank (None for the terminal rank).
    """
    path = case_log_path(step, rank=rank, comm=comm, case=case)
    _state['step'] = step
    if rank == terminal_rank:
        _state['case_log'] = None
        return None
    sys.stdout = open(path, 'w', encoding='utf-8', buffering=1)
    sys.stderr = sys.stdout
    _state['case_log'] = path
    return path


def latest_logs(step, case=None, root=None):
    """the rank logs of the most recent job of `step` of the case (logs/<case>/<step>_<stamp>_rank*.log), sorted"""
    d = Path(root or data_root() or '.') / 'logs' / str(case or _state['case'])
    files = sorted(d.glob('%s_????????-??????*.log' % step))
    if not files:
        return []
    last = max(f.name[len(step) + 1:len(step) + 16] for f in files)
    return [f for f in files if f.name[len(step) + 1:len(step) + 16] == last]


# ------------------------------------------------------------------------------------------------ context
def _rank():
    mpi = sys.modules.get('mpi4py.MPI')              # never import MPI here (a serial script would initialise it)
    if mpi is not None:
        try:
            if mpi.Is_initialized() and not mpi.Is_finalized():
                return str(mpi.COMM_WORLD.Get_rank())
        except Exception:
            pass
    for k in ('PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'PMIX_RANK'):
        if os.environ.get(k):
            return os.environ[k]
    return '-'


def _job():
    for k in ('PYGLDA_JOB_ID', 'UCLOUD_JOB_ID', 'SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID'):
        if os.environ.get(k):
            return os.environ[k]
    return '-'


def _case_log_text():
    p = _state['case_log']
    if p is None:
        return 'logs/%s/ (this process wrote to the terminal / the job output, no rank log)' % (_state['case'] or '-')
    try:
        return str(Path('logs') / Path(p).relative_to(log_root()))
    except Exception:
        return str(p)


# ------------------------------------------------------------------------------------------------ writing
def write_entry(head, lines=(), level='ERROR', step=None):
    """one entry in <data root>/logs/errors_YYYY-MM.log under an exclusive file lock; never raises"""
    now = datetime.now().astimezone()
    text = '%s | %s | case=%s | step=%s | host=%s job=%s rank=%s | %s\n' % (
        now.isoformat(timespec='milliseconds'), level, _state['case'] or '-', step or _state['step'] or '-',
        socket.gethostname(), _job(), _rank(), ' '.join(str(head).split('\n')).strip())
    for ln in lines:
        for sub in str(ln).rstrip('\n').split('\n'):
            text += '  ' + sub + '\n'
    text += '  → case log: %s\n' % _case_log_text()
    try:
        import fcntl
        fn = log_root() / ('%s%s.log' % (ERRORS_PREFIX, now.strftime('%Y-%m')))
        with open(fn, 'a', encoding='utf-8') as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(text)
                fh.flush()
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
    except Exception as err:                        # the errors file must never hide or replace the original error
        try:
            print('[error_log] errors file not written (%s: %s); entry:\n%s' % (type(err).__name__, err, text),
                  file=sys.__stderr__, flush=True)
        except Exception:
            pass
    return text


def report_exception(exc_type=None, exc=None, tb=None, step=None):
    """one entry for an exception (default: the one being handled); each exception object is written once"""
    if exc_type is None:
        exc_type, exc, tb = sys.exc_info()
    if exc_type is None:
        return None
    if exc is not None:
        if getattr(exc, '_pyglda_error_logged', False):
            return None
        try:
            exc._pyglda_error_logged = True
        except Exception:
            pass
    msg = str(exc).strip() if exc is not None else ''
    head = '%s: %s' % (exc_type.__name__, msg) if msg else exc_type.__name__
    lines = ''.join(traceback.format_exception(exc_type, exc, tb)).rstrip('\n').split('\n')
    return write_entry(head, lines, step=step)


class _ErrorHandler(logging.Handler):
    """logging records at ERROR / CRITICAL -> the errors file (the shared handler of install())"""

    def emit(self, record):
        try:
            if record.exc_info and record.exc_info[1] is not None:
                e = record.exc_info[1]
                if getattr(e, '_pyglda_error_logged', False):
                    return
                try:
                    e._pyglda_error_logged = True
                except Exception:
                    pass
                lines = [record.getMessage()] + \
                    ''.join(traceback.format_exception(*record.exc_info)).rstrip('\n').split('\n')
                head = '%s: %s' % (type(e).__name__, str(e).strip())
            else:
                lines = []
                if record.pathname:
                    lines.append('logged at %s:%s (%s)' % (record.pathname, record.lineno, record.funcName))
                head = 'logging.%s(%s): %s' % (record.levelname, record.name, record.getMessage())
            write_entry(head, lines, level='CRITICAL' if record.levelno >= logging.CRITICAL else 'ERROR')
        except Exception:
            pass


def _excepthook(exc_type, exc, tb):
    try:
        if not issubclass(exc_type, SystemExit):
            report_exception(exc_type, exc, tb)
    finally:
        _state['prev_excepthook'](exc_type, exc, tb)


def _thread_excepthook(args):
    try:
        if args.exc_type is not None and not issubclass(args.exc_type, SystemExit):
            report_exception(args.exc_type, args.exc_value, args.exc_traceback,
                             step='%s/thread %s' % (_state['step'] or '-', getattr(args.thread, 'name', '?')))
    finally:
        _state['prev_thread_excepthook'](args)


def install(case, step, data_root=None):
    """
    Called at the start of every runner and entry script: sets case / step / data root of the entries and, once per
    process, installs the hooks (sys.excepthook, threading.excepthook) and the shared logging handler (ERROR and
    above). Calling it again (the next stage of the same process) only changes case, step and data root.
    """
    _state['case'] = case if case is not None else _state['case']
    _state['step'] = step
    if data_root:
        _state['data_root'] = str(data_root)
    root = _state['data_root'] or os.environ.get('PYGLDA_DATA')
    if root and _state['case']:
        '''WaterGAP's own logger (src_GHM/ReWaterGAP/watergap_logger.py) writes its files there, not into the cwd'''
        os.environ['PYGLDA_LOG_DIR'] = str(Path(root) / 'logs' / str(_state['case']))
    if not _state['installed']:
        _state['prev_excepthook'] = sys.excepthook
        sys.excepthook = _excepthook
        _state['prev_thread_excepthook'] = threading.excepthook
        threading.excepthook = _thread_excepthook
        h = _ErrorHandler(level=logging.ERROR)
        h._pyglda_error_handler = True
        logging.getLogger().addHandler(h)
        _state['installed'] = True
    return _state


class catch:
    """
    with error_log.catch(): main()
    records an exception or a non-zero SystemExit (sys.exit('message'), raise SystemExit(2)) and re-raises it
    unchanged, so the exit code and the traceback stay as they were.
    """

    def __init__(self, step=None):
        self.step = step

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            return False
        if issubclass(exc_type, SystemExit):
            code = exc.code if exc is not None else None
            if code not in (None, 0):
                write_entry('SystemExit: %s' % code, [], step=self.step)
        else:
            report_exception(exc_type, exc, tb, step=self.step)
        return False                                 # re-raise


def run_step(cmd, step=None, tail=20, **kw):
    """
    run a subprocess (a model step, mpiexec, a prepare / collect script) with its stderr passed through to ours; a
    non-zero exit writes one entry with the last `tail` stderr lines. Returns the CompletedProcess-like
    (returncode, stderr_tail); the caller decides what to do with the exit code (nothing is raised here).
    """
    import subprocess
    from collections import deque
    last = deque(maxlen=tail)
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, errors='replace', **kw)
    for line in p.stderr:
        sys.stderr.write(line)
        last.append(line.rstrip('\n'))
    rc = p.wait()
    if rc != 0:
        write_entry('CalledProcessError: %s exited with %d' % (cmd if isinstance(cmd, str) else ' '.join(map(str, cmd)), rc),
                    ['last %d stderr lines:' % len(last)] + list(last), step=step)
    return subprocess.CompletedProcess(cmd, rc, stderr='\n'.join(last))
