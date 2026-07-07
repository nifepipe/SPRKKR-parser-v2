"""
read_dos.py — Parse SPR-KKR .dos (OLD-SPRKKR ASCII format) and plot element-resolved DOS.

Usage:
    python scripts/readers/read_dos.py <path/to/file.dos>

Outputs (written next to the input file):
    <name>_bands_eV_abs_conc.dat       band-resolved DOS, concentration-corrected
    <name>_bands_eV_abs_conc.dat_export_final.dat   reorganised by site/element/orbital
    <name>_dos_plot.pdf                plot of element-resolved total DOS
"""

import argparse
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BANDS = {1: 's', 2: 'p', 3: 'd', 4: 'f', 5: 'g', 6: 'h', 7: 'i'}
COLORS = ['k', 'r', 'b', 'g', 'm', 'c', 'orange', 'purple', 'brown', 'gray']
XLIM = (-10, 5)

plt.rcParams.update({
    'font.size': 7, 'axes.labelsize': 7, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'font.family': 'sans-serif', 'pdf.fonttype': 42,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'legend.frameon': False, 'axes.linewidth': 0.5,
})


def load_data(fname):
    with open(fname) as f:
        lines = f.readlines()
    idx = next(i for i, l in enumerate(lines) if 'DOS-FMT:  OLD-SPRKKR' in l)
    return lines[:idx], lines[idx + 1:]


def header_parser(header):
    NE = int(next(l.split()[1] for l in header if l.startswith('NE')))

    def extract(keyword, dtype):
        line = next((l for l in header if keyword in l), None)
        return dtype(line.split()[1]) if line else None

    NQ_eff = extract('NQ_eff', int)
    NT_eff = extract('NT_eff', int)
    EFERMI = extract('EFERMI', float)
    IREL   = extract('IREL', int)

    iq_line = next((i for i, l in enumerate(header) if 'IQ' in l and 'NLQ' in l), None)
    it_line = next((i for i, l in enumerate(header) if 'IT' in l and 'TXT_T' in l), None)

    IQ, NLQ = [], []
    for l in header[iq_line + 1:it_line]:
        if l.strip():
            a, b = map(int, l.split())
            IQ.append(a); NLQ.append(b)
    IQ, NLQ = np.array(IQ), np.array(NLQ)

    it_data = [l.split() for l in header[it_line + 1:] if l.strip()]
    IT    = np.array([int(e[0])   for e in it_data])
    TXT_T = np.array([e[1]        for e in it_data])
    CONC  = np.array([float(e[2]) for e in it_data])
    NAT   = np.array([int(e[3])   for e in it_data])
    IQAT  = np.array([int(e[4])   for e in it_data])

    header_new = ['E', '???']
    for n in range(len(IQAT)):
        idx_nlq = np.where(IQ == IQAT[n])[0][0]
        for m in range(2):
            spin = 'up' if m == 0 else 'dn'
            for l in range(NLQ[idx_nlq]):
                header_new.append(f'{TXT_T[n]} {BANDS[l+1]} {spin}')

    return (np.array(header_new), NE, NQ_eff, NT_eff, EFERMI, IREL,
            IT, TXT_T, CONC, NAT, IQAT, NLQ, IQ)


def data_parser(data, NE, IT, NLQ):
    block_lines = len(data) // NE
    parsed = np.array([
        [float(data[block_start + row][i:i+10].strip())
         for row in range(block_lines)
         for i in range(0, len(data[block_start + row]), 10)
         if data[block_start + row][i:i+10].strip()]
        for block_start in range(0, len(data), block_lines)
    ])
    return parsed


def unit_fixes(data, EFERMI, CONC, TXT_T, HEADER, IT):
    data[:, 0] = (data[:, 0] - EFERMI) * 13.605693122994
    data[:, 2:] /= 13.605693122994
    EFERMI = 0.0
    for n in range(len(IT)):
        conc = CONC[n]
        cols = [i for i, lbl in enumerate(HEADER) if lbl.startswith(TXT_T[n] + ' ')]
        if cols:
            data[:, cols] *= conc
    return data, EFERMI


def data_processor(data, TXT_T, IQ, IQAT, NLQ, HEADER, CONC, EFERMI):
    data_tot = data[:, [0]]
    HEADER_tot = ['E']
    for n in range(len(TXT_T)):
        idx_nlq = np.where(IQ == IQAT[n])[0][0]
        lmax = NLQ[idx_nlq]
        up_cols = [np.where(HEADER == f'{TXT_T[n]} {BANDS[l+1]} up')[0][0] for l in range(lmax)]
        dn_cols = [np.where(HEADER == f'{TXT_T[n]} {BANDS[l+1]} dn')[0][0] for l in range(lmax)]
        up_sum = np.sum(data[:, up_cols], axis=1)
        dn_sum = np.sum(data[:, dn_cols], axis=1)
        data_tot = np.column_stack((data_tot, up_sum, dn_sum, up_sum + dn_sum))
        HEADER_tot += [f'{TXT_T[n]} up', f'{TXT_T[n]} dn', f'{TXT_T[n]} tot']

    atom_up = defaultdict(lambda: np.zeros(data_tot.shape[0]))
    atom_dn = defaultdict(lambda: np.zeros(data_tot.shape[0]))
    atom_tot = defaultdict(lambda: np.zeros(data_tot.shape[0]))
    for n in range(len(TXT_T)):
        aid = IQAT[n]
        atom_up[aid]  += data_tot[:, HEADER_tot.index(f'{TXT_T[n]} up')]
        atom_dn[aid]  += data_tot[:, HEADER_tot.index(f'{TXT_T[n]} dn')]
        atom_tot[aid] += data_tot[:, HEADER_tot.index(f'{TXT_T[n]} tot')]
    for aid in sorted(atom_tot):
        data_tot = np.column_stack((data_tot, atom_up[aid], atom_dn[aid], atom_tot[aid]))
        HEADER_tot += [f'ATOM_{aid} up', f'ATOM_{aid} dn', f'ATOM_{aid} tot']

    return data_tot, HEADER_tot


def save_parsed(data, fname, HEADER):
    out = fname + '_bands_eV_abs_conc.dat'
    np.savetxt(out, data, delimiter=',', header=','.join(HEADER), comments='')
    return out


def sort_and_export(fname_bands, fname_out):
    pattern = re.compile(r'([A-Za-z]+)(?:_([0-9]+))?\s+([spdf])\s+(up|dn)')
    with open(fname_bands, 'r') as f:
        header = np.array(f.readline().strip().split(','))
    data = np.genfromtxt(fname_bands, delimiter=',', skip_header=1)

    col_map = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'up': [], 'dn': []})))
    for i, col in enumerate(header):
        m = pattern.match(col)
        if m:
            elem, site, orb, spin = m.groups()
            site = site or '1'
            col_map[site][elem][orb][spin].append(i)

    energy = data[:, 0]
    out_data = [energy]
    out_header = ['E']
    for site, elems in col_map.items():
        t_site = 'A' + site if site.isdigit() else site
        for elem, orbs in elems.items():
            for orb, spins in orbs.items():
                for spin, indices in spins.items():
                    if indices:
                        out_data.append(np.sum(data[:, indices], axis=1))
                        out_header.append(f'{t_site} {elem} {orb} {spin}')

    out_arr = np.column_stack(out_data)
    np.savetxt(fname_out, out_arr, delimiter=',', header=','.join(out_header), comments='')
    return out_arr, out_header


def plot_dos(data, header, title, outpath):
    fig, ax = plt.subplots(figsize=(6, 4))
    energy = data[:, 0]
    elem_pattern = re.compile(r'A\d+ (\S+) [spdf] (up|dn)')
    elem_totals = defaultdict(lambda: np.zeros(len(energy)))
    for i, col in enumerate(header[1:], 1):
        m = elem_pattern.match(col)
        if m:
            elem_totals[m.group(1)] += data[:, i]

    total = sum(elem_totals.values())
    ax.plot(energy, total, color='k', lw=1.5, label='Total')
    ax.fill_between(energy, total, color='k', alpha=0.05)
    for i, (elem, dos) in enumerate(elem_totals.items()):
        ax.plot(energy, dos, color=COLORS[(i + 1) % len(COLORS)], lw=1.2, label=elem)

    ax.set_xlabel(r'$E - E_\mathrm{F}$ (eV)')
    ax.set_ylabel(r'DOS (states/eV/atom)')
    ax.set_title(title)
    ax.set_xlim(XLIM)
    ax.axvline(0, color='gray', ls=':', lw=0.8)
    ax.axhline(0, color='gray', lw=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Parse SPR-KKR .dos file and plot DOS.')
    parser.add_argument('dos_file', type=Path, help='Path to .dos file')
    args = parser.parse_args()

    fname = str(args.dos_file)
    stem = fname  # output filenames are built by appending to the full path

    print(f'Reading {fname}')
    header_raw, data_raw = load_data(fname)
    HEADER, NE, NQ_eff, NT_eff, EFERMI, IREL, IT, TXT_T, CONC, NAT, IQAT, NLQ, IQ = header_parser(header_raw)
    data = data_parser(data_raw, NE, IT, NLQ)
    data, EFERMI = unit_fixes(data, EFERMI, CONC, TXT_T, HEADER, IT)
    data_tot, HEADER_tot = data_processor(data, TXT_T, IQ, IQAT, NLQ, HEADER, CONC, EFERMI)

    fname_bands = save_parsed(data, stem, HEADER)
    fname_export = fname_bands + '_export_final.dat'
    export_data, export_header = sort_and_export(fname_bands, fname_export)

    plot_out = str(args.dos_file.with_name(args.dos_file.stem + '_dos_plot.pdf'))
    plot_dos(export_data, export_header, title=args.dos_file.stem, outpath=plot_out)
    print(f'Saved: {fname_bands}')
    print(f'Saved: {fname_export}')
    print(f'Saved: {plot_out}')


if __name__ == '__main__':
    main()
