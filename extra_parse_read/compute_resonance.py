"""
compute_resonance.py  —  Extract d-band resonance energies from phase-shift CSVs.

Reads PHASE/pshift_dband_eta{X.X}.csv (produced by parse_pshift.py) and writes:
    PHASE/RES_PHASE_data_extracted.csv

One row per eta value; 20 columns of resonance energies (eV) for all
element × site × spin combinations.

Resonance energy: E_res = argmax_E(dδ/dE) via central finite differences.
"""

import os
from pathlib import Path
import numpy as np

CSV_DIR  = Path(__file__).parents[2] / 'PHASE'
OUT_FILE = CSV_DIR / 'RES_PHASE_data_extracted.csv'

ELEMENTS = ['Cr', 'Mn', 'Fe', 'Co', 'Ni']
SITES    = [1, 2]
SPIN_MAP = {'d5/2': 'up', 'd3/2': 'down'}

COLS = [f'{el}_{s}_{sp}' for el in ELEMENTS for s in SITES for sp in ('up', 'down')]


def resonance_energy(energies, pshifts):
    """argmax of dδ/dE via central differences."""
    dE     = np.diff(energies)
    deriv  = np.diff(pshifts) / dE
    e_mid  = 0.5 * (energies[:-1] + energies[1:])
    return e_mid[np.argmax(deriv)]


def load_csv(path):
    """Return {(element, site, channel): (energy_array, pshift_array)}."""
    data = {}
    with open(path) as fh:
        next(fh)  # skip header
        for line in fh:
            parts = line.strip().split(',')
            eta, element, site, channel = parts[0], parts[1], int(parts[2]), parts[3]
            e, d = float(parts[4]), float(parts[5])
            key = (element, site, channel)
            if key not in data:
                data[key] = ([], [])
            data[key][0].append(e)
            data[key][1].append(d)
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in data.items()}


def main():
    eta_vals = sorted(
        float(f.replace('pshift_dband_eta', '').replace('.csv', ''))
        for f in os.listdir(CSV_DIR)
        if f.startswith('pshift_dband_eta') and f.endswith('.csv')
    )

    rows = []
    for eta in eta_vals:
        fname = CSV_DIR / f'pshift_dband_eta{eta:.1f}.csv'
        data  = load_csv(fname)
        row   = {'eta': eta}
        for el in ELEMENTS:
            for s in SITES:
                for channel, spin in SPIN_MAP.items():
                    key = (el, s, channel)
                    col = f'{el}_{s}_{spin}'
                    energies, pshifts = data[key]
                    row[col] = resonance_energy(energies, pshifts)
        rows.append(row)
        print(f'eta={eta:.1f}  Cr_1_up={row["Cr_1_up"]:.4f} eV  Fe_2_down={row["Fe_2_down"]:.4f} eV')

    with open(OUT_FILE, 'w') as fh:
        fh.write('eta;' + ';'.join(COLS) + '\n')
        for row in rows:
            vals = ';'.join(f'{row[c]:.6f}' for c in COLS)
            fh.write(f'{row["eta"]:.1f};{vals}\n')

    print(f'\nWritten: {OUT_FILE}')


if __name__ == '__main__':
    main()
