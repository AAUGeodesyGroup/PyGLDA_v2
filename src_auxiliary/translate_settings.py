"""
Translate a DA_setting.json written for an older PyGLDA v2 code version (a template or the run copy of a run) to the
keys of the current code.

Rules (mechanical; REMOVED below lists the keys that the current code no longer reads):
  - a removed key that is present but switched off (null, false, {}, [] or its "off" value) is dropped;
  - a removed key that is switched ON stops the translation: the current code cannot reproduce that run (use the code
    version the run was made with);
  - every other key is kept as it is. If nothing is dropped, the file is copied byte for byte.
In v2.0 no key has been removed or renamed (the sensitivity options of the WRR and GMD experiments are all still
read), so every settings file of the earlier code versions is read as it is and the tool only confirms this.
perturbation.json and Config_ReWaterGAP.json are copied as they are.

    python -m src_auxiliary.translate_settings <old DA_setting.json | settings folder> <output file | folder>
    python -m src_auxiliary.translate_settings --check <old DA_setting.json | settings folder>

Exit status: 0 translated (or nothing to translate), 2 a removed option is switched on.
"""
import json
import shutil
import sys
from pathlib import Path

SETTING_FILES = ('DA_setting.json', 'perturbation.json', 'Config_ReWaterGAP.json')


def _off(v):
    return v is None or v is False or v == {} or v == [] or v == ''


'''removed keys: (path in DA_setting.json, test "switched off", what it was); empty in v2.0'''
REMOVED = []


def translate(setting):
    """(new settings dict, dropped [(key, value)], active [(key, value, meaning)]); the input is not modified"""
    new = json.loads(json.dumps(setting))
    dropped, active = [], []
    for path, is_off, meaning in REMOVED:
        node = new
        for k in path[:-1]:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if not isinstance(node, dict) or path[-1] not in node:
            continue
        key = '.'.join(path)
        if is_off(node[path[-1]]):
            dropped.append((key, node.pop(path[-1])))
        else:
            active.append((key, node[path[-1]], meaning))
    return new, dropped, active


def translate_file(src, dst=None):
    """translate one DA_setting.json; dst None = check only. Returns the exit status."""
    src = Path(src)
    setting = json.load(open(src))
    new, dropped, active = translate(setting)
    for key, v in dropped:
        print('%s: dropped %s = %s (switched off)' % (src, key, json.dumps(v)))
    if active:
        for key, v, meaning in active:
            print('%s: %s = %s is switched ON - %s is no longer in the code; this run cannot be reproduced with the '
                  'current code' % (src, key, json.dumps(v), meaning))
        return 2
    if not dropped:
        print('%s: nothing to translate (the current code reads it as it is)' % src)
    if dst is not None:
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dropped:
            with open(dst, 'w') as fh:
                json.dump(new, fh, indent=4)
        elif src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        print('%s -> %s' % (src, dst))
    return 0


def main(argv):
    check = '--check' in argv
    args = [a for a in argv if a != '--check']
    if len(args) != (1 if check else 2):
        sys.exit(__doc__)
    src = Path(args[0])
    dst = None if check else Path(args[1])
    if src.is_dir():
        rc = translate_file(src / 'DA_setting.json', None if dst is None else dst / 'DA_setting.json')
        if dst is not None and rc == 0:
            for fn in SETTING_FILES[1:]:
                if (src / fn).exists() and (src / fn).resolve() != (dst / fn).resolve():
                    shutil.copyfile(src / fn, dst / fn)
        return rc
    return translate_file(src, dst)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
