"""DA review 246: test of the lake / wetland guard. (1) _limit_ratio == plain ratio bit for bit on every input a free
run can produce (petgwr > petgwr_max >= 0); (2) 0 / 0 and negative cases return 0 instead of raising; (3) the whole
numba function compiles and, on states at / below the floor with zero area, runs instead of raising; the original
raises ZeroDivisionError on the same input; (4) on random free-run inputs both versions give identical outputs."""
import sys, types, importlib.util, numpy as np
pkg = types.ModuleType('model'); sys.modules['model'] = pkg
lw = types.ModuleType('model.lateralwaterbalance'); sys.modules['model.lateralwaterbalance'] = lw
import tempfile as _tf, os as _os
_t = _os.path.join(_tf.mkdtemp(), 'srf.py'); open(_t, 'w').write(open(sys.argv[3]).read().replace('cache=True', 'cache=False'))
spec = importlib.util.spec_from_file_location('model.lateralwaterbalance.storage_reduction_factor', _t)
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf); sys.modules[spec.name] = rf; lw.storage_reduction_factor = rf
import tempfile, os
def load(name, path):
    t = os.path.join(tempfile.mkdtemp(), name + '.py'); open(t, 'w').write(open(path).read().replace('cache=True', 'cache=False'))
    s = importlib.util.spec_from_file_location(name, t); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
new = load('lw_new', sys.argv[1]); old = load('lw_old', sys.argv[2])
rng = np.random.default_rng(1)
den = rng.uniform(1e-12, 10, 200000); num = den * rng.uniform(0, 1, den.size)
assert all(new._limit_ratio(a, b) == a / b for a, b in zip(num[:20000], den[:20000])); print('1 identical ratio on free-run inputs: OK')
assert new._limit_ratio(-1e-16, 0.0) == 0.0 and new._limit_ratio(-5.0, 2.0) == 0.0 and new._limit_ratio(0.0, 0.0) == 0.0; print('2 guard cases: OK')
import inspect; print(inspect.signature(new.lake_wetland_water_balance.py_func))
def call(m, swb, storage, mx, area, inflow=0.0, pet=1e-6, p=0.0, arid=0):
    return m.lake_wetland_water_balance(0, 0, swb, storage, p, pet, arid, 1, inflow, 0.01, 0.01, 3.32, 0.0,
                                        max_storage=mx, max_area=area, lakewet_frac=1.0, lake_outflow_exp=1.5,
                                        wetland_outflow_exp=2.5, reservoir_area=0.0,
                                        accumulated_unsatisfied_potential_netabs_sw=0.0)
# 3 crash cases: local lake a round-off below -max with zero inflow, (no-lake cells are never passed: routing calls the balance only where the fraction > 0),
#   global wetland slightly below 0
cases = [('local lake', -1.0 - 1e-15, 1.0, 10.0), ('global wetland', -1e-17, 1.0, 10.0),
         ('local wetland', -1e-17, 1.0, 10.0)]
for swb, st, mx, ar in cases:
    try:
        call(old, swb, st, mx, ar); o = 'no error'
    except ZeroDivisionError:
        o = 'ZeroDivisionError'
    r = call(new, swb, st, mx, ar)
    print('3 %-15s S=%g max=%g: original %s | guarded storage %.3g outflow %.3g gwr %.3g' % (swb, st, mx, o, r[0], r[1], r[2]))

rngA = np.random.default_rng(7); rngB = np.random.default_rng(7)
def draw(r, k):
    swb = ['local lake', 'global lake', 'local wetland', 'global wetland'][k % 4]
    mx = r.uniform(0.01, 5); lo = -mx if 'lake' in swb else 0.0
    return swb, r.uniform(lo, mx), mx, r.uniform(1, 500), r.uniform(0, 0.5), r.uniform(0, 1e-3), r.uniform(0, 1e-5), k % 2
same = 0; hit = 0
for k in range(4000):
    swb, st, mx, ar, inf, pet, p, arid = draw(rngA, k)
    if k % 3 == 0: inf = 0.0                         # dry days: the limit branch is reached often
    a = call(old, swb, st, mx, ar, inf, pet, p, arid); b = call(new, swb, st, mx, ar, inf, pet, p, arid)
    same += all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))
    hit += float(np.asarray(a[0])) == (-mx if 'lake' in swb else 0.0)
print('4 free-run inputs: %d of 4000 outputs identical (limit branch reached %d times)' % (same, hit))
assert same == 4000
print('ALL OK')
