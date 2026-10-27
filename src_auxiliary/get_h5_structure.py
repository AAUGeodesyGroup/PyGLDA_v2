"""
Print the tree of groups and datasets (with shapes) of an HDF5 file:
    python -m src_auxiliary.get_h5_structure <file.h5|file.hdf5>
"""
import sys
from pathlib import Path

import h5py


def get_hdf5_structure(filepath):
    """text tree of the groups and datasets (with their shapes) of an .h5 / .hdf5 file"""
    def append_structure(fdata, flevel=0, texts=None):
        if texts is None:
            texts = []

        texts.append(f"{'|  ' * flevel}|--{fdata.name.split('/')[-1]}")

        if isinstance(fdata, h5py.Group):
            flevel += 1
            texts.append('|  ' * flevel + '|')
            for fkey in fdata.keys():
                append_structure(fdata[fkey], flevel, texts)
            flevel -= 1
            texts.append('|  ' * flevel + '|')

        elif isinstance(fdata, h5py.Dataset):
            texts[-1] += f' {fdata.shape}'

        return texts

    filepath = Path(filepath)
    assert filepath.suffix in ('.h5', '.hdf5'), filepath
    lines = []

    with h5py.File(filepath, 'r') as f:
        lines.append(f'{filepath.name}')
        lines.append('|')

        for key in f.keys():
            lines = append_structure(f[key], texts=lines)

    for i in range(len(lines) - 1, -1, -1):
        if lines[i].replace('|', '').replace(' ', '') == '':
            lines.pop(i)
        else:
            break

    return '\n'.join(lines)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: python -m src_auxiliary.get_h5_structure <file.h5|file.hdf5>')
    print(get_hdf5_structure(sys.argv[1]))
