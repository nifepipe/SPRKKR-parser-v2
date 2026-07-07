"""
parse_pshift.py  —  Extract d-band phase shifts from SPR-KKR PSHIFT .agr files.

Reads all eta subdirectories under PHASE/ and writes one CSV per eta value:
    PHASE/pshift_dband_eta{X.X}.csv

Columns: eta, element, site, channel, energy_eV, pshift_rad

Only the G0 (exchange-field-off) d-band channels are extracted:
    S3 = d_{3/2}  (κ = +2, j = l−½)  → labelled 'd3/2' (spin down)
    S4 = d_{5/2}  (κ = −3, j = l+½)  → labelled 'd5/2' (spin up)
"""

import os
import re
from pathlib import Path
import numpy as np

BASE    = Path(__file__).parents[2] / 'PHASE'
OUT_DIR = BASE

TARGET_CHANNELS = {'G0.S3': 'd3/2', 'G0.S4': 'd5/2'}
FILE_RE = re.compile(r'pshift_([A-Za-z]+)_(\d+)\.agr$')


def parse_agr(path):
    """Return {channel_label: (energy_array, pshift_array)} for target G0 d-channels."""
    results = {}
    current_target = None
    collecting = False
    buf = []

    with open(path, 'r') as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('@target'):
                if current_target in TARGET_CHANNELS and buf:
                    arr = np.array(buf)
                    results[TARGET_CHANNELS[current_target]] = (arr[:, 0], arr[:, 1])
                current_target = line.split()[1]
                collecting = current_target in TARGET_CHANNELS
                buf = []
            elif line == '&':
                if collecting and buf:
                    arr = np.array(buf)
                    results[TARGET_CHANNELS[current_target]] = (arr[:, 0], arr[:, 1])
                    buf = []
                collecting = False
                current_target = None
            elif collecting and line and not line.startswith('@'):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        buf.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass

    # flush if file ended without trailing &
    if collecting and current_target in TARGET_CHANNELS and buf:
        arr = np.array(buf)
        results[TARGET_CHANNELS[current_target]] = (arr[:, 0], arr[:, 1])

    return results


def process_eta(eta_dir, eta_val):
    rows = []
    for fname in sorted(os.listdir(eta_dir)):
        m = FILE_RE.search(fname)
        if not m:
            continue
        element, site = m.group(1), int(m.group(2))
        data = parse_agr(os.path.join(eta_dir, fname))
        for channel, (energies, pshifts) in data.items():
            for e, d in zip(energies, pshifts):
                rows.append((eta_val, element, site, channel, e, d))
    return rows


def main():
    eta_dirs = sorted(
        d for d in os.listdir(BASE)
        if d.startswith('eta') and os.path.isdir(os.path.join(BASE, d))
    )

    for dname in eta_dirs:
        eta_val = float(dname.replace('eta', ''))
        rows = process_eta(os.path.join(BASE, dname), eta_val)

        if not rows:
            print(f'WARNING: no data for {dname}')
            continue

        out_path = OUT_DIR / f'pshift_dband_eta{eta_val:.1f}.csv'
        with open(out_path, 'w') as fh:
            fh.write('eta,element,site,channel,energy_eV,pshift_rad\n')
            for r in rows:
                fh.write(f'{r[0]:.1f},{r[1]},{r[2]},{r[3]},{r[4]:.10E},{r[5]:.10E}\n')

        print(f'{dname}: {len(rows)} rows -> {out_path.name}')


if __name__ == '__main__':
    main()
