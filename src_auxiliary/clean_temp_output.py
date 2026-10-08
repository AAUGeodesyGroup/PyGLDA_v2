"""
Removal of the temporary daily model output (daily_output_YYYY-MM-DD.nc) that the open-loop and
data-assimilation runs write into <OL_output_temp_dir>/Ens_k and <DA_output_temp_dir>/Ens_k.

The entry point used by the flow control is RDA.clean_temp_output(stage, ...) in
src_FlowControl/Regional_DA.py, which reads the folders from DA_setting.json and calls
clean_daily_output() below. Running this file directly is only a fallback for a manual clean-up.

Safety rules
  * dry run by default: reports what would be deleted and how much space it frees
  * only files named daily_output_*.nc inside Ens_* folders are touched; no other files, no folders
  * a stage whose results have not been collected yet (Res/<case>/<STAGE>/Ens_k/basin_ts_<STAGE>.h5
    missing for one of the members) is refused unless force=True
  * the open-loop scratch <OL_output_temp_dir>/Ens_k is shared between cases (a global open loop is reused by
    other cases, e.g. demo_Global on the open loop of demo_Danube); it is refused unless force=True
"""
import re
from pathlib import Path
if __name__ == '__main__':                        # run directly: make the package root importable
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src_DA.EnumDA import Stage

PATTERN = re.compile(r'^daily_output_\d{4}-\d{2}-\d{2}\.nc$')


def clean_daily_output(temp_dir, res_dir, stage: Stage, case=None, dry_run=True, force=False):
    """
    temp_dir  : OL_output_temp_dir / DA_output_temp_dir from DA_setting.json. The OL scratch is organised as
                <temp_dir>/Ens_k, the DA scratch as <temp_dir>/<case>/Ens_k; both layouts are handled:
                if <temp_dir>/<case> exists it is used, otherwise <temp_dir> itself.
    res_dir   : Res/<case> folder; used to check that Res/<case>/<stage>/Ens_k/basin_ts_<stage>.h5 exists
    stage     : Stage.OL or Stage.DA
    returns   : dict(files=n, bytes=b, deleted=bool, skipped_reason=str|None)
    """
    if stage not in (Stage.OL, Stage.DA):
        raise ValueError('clean_daily_output: stage must be Stage.OL or Stage.DA, got %r' % (stage,))
    stage_name = stage.name                       # folder / file names: Res/<case>/OL, basin_ts_OL.h5, ...
    temp_dir, res_dir = Path(temp_dir), Path(res_dir)
    out = dict(files=0, bytes=0, deleted=False, skipped_reason=None)
    if case is not None and (temp_dir / case).is_dir():
        temp_dir = temp_dir / case
    elif stage == Stage.OL and not force:
        '''<OL_output_temp_dir>/Ens_k is not case-specific: the global open loop there is reused by other cases
        (e.g. demo_Global assimilates on the open loop of demo_Danube), and the collection check below only looks at
        Res/<this case>. Deleting it needs force=True.'''
        out['skipped_reason'] = ('%s is shared between cases (no folder %s): the open loop there may be used by '
                                 'other cases' % (temp_dir, temp_dir / (case or '<case>')))
        print('[%s] %s -> skipped; use force=True to delete it anyway' % (stage_name, out['skipped_reason']))
        return out

    if not temp_dir.is_dir():
        out['skipped_reason'] = 'no folder %s' % temp_dir
        print('[%s] %s' % (stage_name, out['skipped_reason']))
        return out

    ens_dirs = sorted(d for d in temp_dir.iterdir() if d.is_dir() and re.fullmatch(r'Ens_\d+', d.name))
    if not ens_dirs:
        out['skipped_reason'] = 'nothing under %s' % temp_dir
        print('[%s] %s' % (stage_name, out['skipped_reason']))
        return out

    missing = [d.name for d in ens_dirs if not (res_dir / stage_name / d.name / ('basin_ts_%s.h5' % stage_name)).exists()]
    if missing and not force:
        out['skipped_reason'] = 'not collected yet for %s (%s missing)' % (
            ', '.join(missing), res_dir / stage_name / 'Ens_k' / ('basin_ts_%s.h5' % stage_name))
        print('[%s] %s -> skipped; use force=True to override' % (stage_name, out['skipped_reason']))
        return out

    files = [f for d in ens_dirs for f in d.iterdir() if f.is_file() and PATTERN.match(f.name)]
    out['files'] = len(files)
    out['bytes'] = sum(f.stat().st_size for f in files)
    span = (min(f.name[13:23] for f in files), max(f.name[13:23] for f in files)) if files else ('-', '-')
    print('[%s] %6d daily files, %7.2f GB, %s .. %s, in %d member folders under %s' %
          (stage_name, len(files), out['bytes'] / 1e9, span[0], span[1], len(ens_dirs), temp_dir))

    if dry_run:
        print('[%s] dry run - nothing deleted (call with dry_run=False to delete)' % stage_name)
    else:
        for f in files:
            f.unlink()
        out['deleted'] = True
        print('[%s] deleted' % stage_name)
    return out


if __name__ == '__main__':
    '''manual fallback; the normal way is RDA.clean_temp_output() from RDA_demo.py'''
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--temp_dir', required=True, help='e.g. <external_data>/DA_output')
    p.add_argument('--res_dir', required=True, help='e.g. <external_data>/Res/demo_3')
    p.add_argument('--stage', choices=['OL', 'DA'], required=True)
    p.add_argument('--case', default=None, help='case name, needed for the per-case DA scratch layout')
    p.add_argument('--yes', action='store_true', help='actually delete (default: dry run)')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    clean_daily_output(a.temp_dir, a.res_dir, Stage[a.stage], case=a.case, dry_run=not a.yes, force=a.force)
