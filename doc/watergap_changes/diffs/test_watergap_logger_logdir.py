"""watergap_logger.py (logs step 1, reviews 469 / 470): without PYGLDA_LOG_DIR the log file is written to the working
directory exactly as upstream; with it, to $PYGLDA_LOG_DIR/watergap-<module>_<stamp>.log. Same message in both.
usage: python test_watergap_logger_logdir.py <code root>        exit 0 = pass"""
import logging, os, sys, tempfile, glob
sys.path.insert(0, os.path.join(sys.argv[1] if len(sys.argv) > 1 else '.', 'src_GHM', 'ReWaterGAP'))
import watergap_logger as wl

with tempfile.TemporaryDirectory() as cwd, tempfile.TemporaryDirectory() as logs:
    os.chdir(cwd)
    os.environ.pop('PYGLDA_LOG_DIR', None)
    wl.config_logger(logging.WARNING, 'testmod.py', 'message A')
    a = glob.glob(os.path.join(cwd, 'testmod.py_*.log'))
    assert len(a) == 1 and 'message A' in open(a[0]).read(), a
    os.environ['PYGLDA_LOG_DIR'] = os.path.join(logs, 'CASE')
    wl.config_logger(logging.ERROR, 'testmod.py', 'message B')
    b = glob.glob(os.path.join(logs, 'CASE', 'watergap-testmod_*.log'))
    assert len(b) == 1 and 'message B' in open(b[0]).read(), b
    assert len(glob.glob(os.path.join(cwd, '*.log'))) == 1, 'second file written to the cwd'
print('PASS: cwd without PYGLDA_LOG_DIR (%s), log folder with it (%s)' % (os.path.basename(a[0]), os.path.basename(b[0])))
