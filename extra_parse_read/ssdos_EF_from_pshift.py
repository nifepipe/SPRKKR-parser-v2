# ssdos_EF_from_pshift.py
#
# Single-site (scattering) DOS at E_F from the phase-shift data, via the
# Friedel relation per relativistic channel kappa:
#     n_kappa(E) = (2j+1)/pi * d delta_kappa / dE      [states/eV]
# aggregated to l:  n_l = sum over kappa in l.
#
# This is the quantity that controls how the screening of inter-species
# charge differences partitions among the l channels -- the core of the
# sp-protection argument.  Independent of the ssdos code path (and its
# TAUQ bug); uses only PHASE/eta*/...pshift_{El}_{site}.agr (G0 = Bxc off).
#
# Run from Data_Cantor/:  python scripts/parsers/ssdos_EF_from_pshift.py

import os
import re
import numpy as np

RY_EV = 13.605693
PHASE = 'PHASE'
ELEMENTS = ['Cr', 'Mn', 'Fe', 'Co', 'Ni']
ETAS = ['0.0', '1.0']
# G0 set index -> (label, l, 2j+1)   [order as in PHASE/README.md]
G0_SETS = {0: ('s12', 0, 2), 1: ('p12', 1, 2), 2: ('p32', 1, 4),
           3: ('d32', 2, 4), 4: ('d52', 2, 6), 5: ('f52', 3, 6),
           6: ('f72', 3, 8)}


def read_ef_ev(eta):
    pot = os.path.join(PHASE, 'eta' + eta,
                       'HEA_FM_L10_eta%s_final.pot' % eta)
    with open(pot) as fh:
        for line in fh:
            if line.startswith('EF'):
                return float(line.split()[1]) * RY_EV
    raise RuntimeError('EF not found in ' + pot)


def parse_g0(path):
    """All G0 sets of one .agr -> {set_index: (E_eV, delta_rad)}.
    Same parser as scripts/figures/pshift_EF_sites.py."""
    sets, target, buf = {}, None, []
    rx = re.compile(r'@target G(\d)\.S(\d+)')
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            m = rx.match(line)
            if m:
                if target is not None and buf:
                    sets[target] = np.array(buf)
                g, s = int(m.group(1)), int(m.group(2))
                target = s if g == 0 else None
                buf = []
            elif line == '&':
                if target is not None and buf:
                    sets[target] = np.array(buf)
                target, buf = None, []
            elif target is not None and line and not line.startswith('@'):
                p = line.split()
                if len(p) == 2:
                    buf.append((float(p[0]), float(p[1])))
    return {k: (v[:, 0], v[:, 1]) for k, v in sets.items()}


def n_l_at_ef(agr_path, ef_ev):
    """Single-site DOS n_l(E_F) in states/eV from d(delta)/dE."""
    sets = parse_g0(agr_path)
    n = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for idx, (_, l, deg) in G0_SETS.items():
        e, d = sets[idx]
        ddde = np.gradient(d, e)              # d delta / dE  (1/eV)
        n[l] += deg / np.pi * np.interp(ef_ev, e, ddde)
    return n


LN = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
for eta in ETAS:
    ef = read_ef_ev(eta)
    print('=== L1_0 FM eta=%s   (E_F = %.4f eV) ===' % (eta, ef))
    print('single-site DOS n_l(E_F) in states/eV  [per element, site 1a]')
    for el in ELEMENTS:
        f = os.path.join(PHASE, 'eta' + eta,
                         'HEA_FM_PSHIFT_eta%s_PSHIFT_pshift_%s_1.agr'
                         % (eta, el))
        n = n_l_at_ef(f, ef)
        tot = sum(n.values())
        dfrac = 100 * n[2] / tot if tot != 0 else float('nan')
        print('  %s:  ' % el
              + '  '.join('%s %+7.4f' % (LN[l], n[l]) for l in range(4))
              + '   | d-fraction %.1f %%' % dfrac)
    # site splitting of n_d at eta=1 (environment effect on the resonance)
    if eta == '1.0':
        print('site splitting n_d(1a) - n_d(1d) at E_F (states/eV):')
        for el in ELEMENTS:
            n1 = n_l_at_ef(os.path.join(
                PHASE, 'eta1.0',
                'HEA_FM_PSHIFT_eta1.0_PSHIFT_pshift_%s_1.agr' % el), ef)
            n2 = n_l_at_ef(os.path.join(
                PHASE, 'eta1.0',
                'HEA_FM_PSHIFT_eta1.0_PSHIFT_pshift_%s_2.agr' % el), ef)
            print('  %s: %+7.4f' % (el, n1[2] - n2[2]))
