# parse_bsf.py
#
# Parser for SPRKKR 9.7 (Alberto patch) BLOCHSF *_ALF.bsf / *_spol.bsf files
# in MODE k-k-plot (constant-E 2D cut). Verified line-by-line against
# blochsf.f of the 9.7_alberto source:
#
#   NBSFOP=4                                  (blochsf.f:54)
#   _ALF.bsf written as:                      (blochsf.f:1214-1222)
#       DO IE=1,NE                            (NE=1 here)
#         DO IOP=1,NOP   (NOP=4 for BSFALF)
#           DO IK1=1,NK1
#             record: ((AKE(IK1,IK2,IE,IQ,3,IOP),IK2=1,NK2),IQ=IQBOT,IQTOP)
#   FMT_BSF='(E12.5)'  -> exactly one value per line, IK2 fastest, then IQ.
#   IOP=1 : total BSF  A_BSF  (negatives clamped to 0 in code, l.1090-1094)
#   IOP=2 : ALF current component 1 (x)
#   IOP=3 : ALF current component 2 (y)
#   IOP=4 : ALF current component 3 (z)
#   k-grid (blochsf.f:1051-1054):
#       k = s1*VECK1 + s2*VECK2 + 1e-4 + VECKA
#       s1=(IK1-1)/(NK1-1)   (IK1 <-> NK1 <-> VECK1)
#       s2=(IK2-1)/(NK2-1)   (IK2 <-> NK2 <-> VECK2)
#   Data starts after the LAST line that is all '#' (FORMAT 80('#'), l.1414).
#
# Velocity conversion (see info.md, re-checked):
#   ratio_i = A_ALF,i / A_BSF = spectral-weighted <j_i> = m_e <v_i>  (j=m c alpha, v=c alpha)
#   v_i [m/s] = ratio_i * alpha_fs * c_SI   (= ratio_i * 2.18770e6 m/s)
#
# Usage:
#   python parse_bsf.py BSF.inp HEA_FM_A1_eta0.0_BLOCHSF_ALF.bsf [--save out.pdf]

import os
import sys
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

ALPHA_FS = 7.2973525693e-3
C_SI = 299792458.0
V_FAC = ALPHA_FS * C_SI          # ~2.18770e6 m/s : ratio -> v in m/s

plt.rcParams.update({
    'font.size': 7, 'axes.labelsize': 7, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.titlesize': 7,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'legend.frameon': False, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
})


def read_inp(path):
    """Parse the SPRKKR input so we know what was computed and can cross-check."""
    txt = open(path, 'r', errors='replace').read()

    def grab(pat, default=None, cast=str):
        m = re.search(pat, txt, re.IGNORECASE)
        return cast(m.group(1)) if m else default

    inp = {}
    inp['dataset'] = grab(r'CONTROL\s+DATASET\s*=\s*(\S+)')
    inp['potfil'] = grab(r'POTFIL\s*=\s*(\S+)')
    inp['nl'] = grab(r'SITES\s+NL\s*=\s*(\d+)', None, int)
    inp['nktab'] = grab(r'NKTAB\s*=\s*(\d+)', None, int)
    inp['ime'] = grab(r'ImE\s*=\s*([0-9.eEdD+-]+)')
    inp['emin'] = grab(r'EMINEV\s*=\s*([0-9.eEdD+-]+)', None, float)
    inp['emax'] = grab(r'EMAXEV\s*=\s*([0-9.eEdD+-]+)', None, float)
    inp['nk1'] = grab(r'NK1\s*=\s*(\d+)', None, int)
    inp['nk2'] = grab(r'NK2\s*=\s*(\d+)', None, int)
    m = re.search(r'K1\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE)
    inp['k1'] = [float(x) for x in m.group(1).split(',')] if m else None
    m = re.search(r'K2\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE)
    inp['k2'] = [float(x) for x in m.group(1).split(',')] if m else None
    inp['bsfalf'] = bool(re.search(r'^\s*BSFALF\b', txt, re.IGNORECASE | re.MULTILINE))
    m = re.search(r'MODE\s+ZEROED_L\s*=\s*\{([^}]*)\}', txt, re.IGNORECASE)
    inp['zeroed_l'] = [int(x) for x in m.group(1).split(',')] if m else None
    return inp


BOHR_TO_ANG = 0.52917721067         # CODATA: 1 a0 = 0.529177 Angstrom


def read_pot_lattice(pot_path):
    """ALAT (Bohr) + BRAVAIS code from a SPRKKR *_final.pot file."""
    out = {'alat': None, 'bravais': None}
    try:
        for ln in open(pot_path, 'r', errors='replace'):
            m = re.match(r'\s*ALAT\s+([0-9.eEdD+-]+)', ln)
            if m:
                out['alat'] = float(m.group(1).replace('D', 'E').replace('d', 'e'))
            m = re.match(r'\s*BRAVAIS\s+(\d+)', ln)
            if m:
                out['bravais'] = int(m.group(1))
            if out['alat'] and out['bravais']:
                break
    except OSError:
        pass
    return out


def frame_grid(d, leaf_pot, frame):
    """Return (X, Y, xlabel, ylabel) for plotting.

    frame='native' : fractional k along the scan vectors, units 2pi/a
    frame='l10'    : absolute k in 1/Angstrom; a cubic (fcc, A1) scan along
                     [100]/[010] is rotated 45 deg so its axes coincide with
                     the L10 fct frame ([110]/[1-10]). L10 itself is already
                     in that frame -> only the absolute rescale is applied.
                     The sqrt(2) zone-size ratio is carried automatically by
                     each calculation's own ALAT (no hardcoded factor).
    """
    s1, s2 = d['s1'], d['s2']
    S1, S2 = np.meshgrid(s1, s2, indexing='ij')      # (NK1,NK2)

    if frame != 'l10':
        return d['KX'], d['KY'], r'$k_1$  ($2\pi/a$)', r'$k_2$  ($2\pi/a$)'

    lat = read_pot_lattice(leaf_pot) if leaf_pot else {'alat': None,
                                                       'bravais': None}
    alat = lat['alat']
    if not alat:
        # fall back: cannot rescale -> native
        return (d['KX'], d['KY'],
                r'$k_1$  ($2\pi/a$, no ALAT)', r'$k_2$  ($2\pi/a$)')

    # absolute Cartesian along the two scan vectors, in 1/Angstrom
    g = 2.0 * np.pi / (alat * BOHR_TO_ANG)           # 2pi/ALAT [1/Ang]
    KX = S1 * g
    KY = S2 * g
    cubic = lat['bravais'] in (13, 1, 2, 3)          # fcc/bcc/sc cubic codes
    if cubic:
        # rotate -45 deg: u along [110], v along [1-10]
        U = (KX + KY) / np.sqrt(2.0)
        V = (KY - KX) / np.sqrt(2.0)
    else:
        # tetragonal L10: scan x already along fcc [110]
        U, V = KX, KY
    return (U, V,
            r'$k_{[110]}$  ($\mathrm{\AA}^{-1}$)',
            r'$k_{[1\bar{1}0]}$  ($\mathrm{\AA}^{-1}$)')


def _to_float(tok):
    # Fortran may emit 'D' exponents; E12.5 here uses 'E', but be safe.
    return float(tok.replace('D', 'E').replace('d', 'e'))


def read_bsf(path):
    """Parse a BLOCHSF k-k-plot .bsf file. Returns dict with grids + blocks."""
    with open(path, 'r', errors='replace') as fh:
        lines = fh.read().splitlines()

    hdr = {}

    def hgrab(key, cast=str):
        for ln in lines[:60]:
            m = re.match(r'\s*' + key + r'\s+(\S.*?)\s*$', ln)
            if m:
                return cast(m.group(1).split()[0])
        return None

    hdr['keyword'] = hgrab('KEYWORD')
    hdr['ne'] = hgrab('NE', int)
    hdr['irel'] = hgrab('IREL', int)
    hdr['efermi'] = hgrab('EFERMI', float)
    hdr['nq'] = hgrab('NQ_eff', int)

    # #DATASET / MODE / NK1 / NK2 / ERYD / VECK1 / VECK2 block
    def bgrab(key):
        for ln in lines[:80]:
            m = re.match(r'\s*' + key + r'\s+(.*\S)\s*$', ln)
            if m:
                return m.group(1).split()
        return None

    nk1 = bgrab('NK1')
    nk2 = bgrab('NK2')
    eryd = bgrab('ERYD')
    veck1 = bgrab('VECK1')
    veck2 = bgrab('VECK2')
    hdr['nk1'] = int(nk1[0]) if nk1 else None
    hdr['nk2'] = int(nk2[0]) if nk2 else None
    hdr['eryd'] = [float(x) for x in eryd] if eryd else None
    hdr['veck1'] = np.array([float(x) for x in veck1]) if veck1 else None
    hdr['veck2'] = np.array([float(x) for x in veck2]) if veck2 else None

    # Data begins after the LAST all-'#' line (FORMAT 80('#'), blochsf.f:1414)
    last_hash = max(i for i, ln in enumerate(lines)
                    if ln.strip() and set(ln.strip()) == {'#'})
    data_tokens = []
    for ln in lines[last_hash + 1:]:
        s = ln.strip()
        if s:
            data_tokens.append(s)
    vals = np.array([_to_float(t) for t in data_tokens], dtype=np.float64)

    nk1, nk2 = hdr['nk1'], hdr['nk2']
    ne = hdr['ne'] or 1
    nq = hdr['nq'] or 1
    if None in (nk1, nk2):
        raise ValueError('NK1/NK2 not found in header')

    per_op = ne * nk1 * nq * nk2
    if vals.size % per_op != 0:
        raise ValueError(
            'data count %d not divisible by NE*NK1*NQ*NK2=%d '
            '(NE=%d NK1=%d NQ=%d NK2=%d)'
            % (vals.size, per_op, ne, nk1, nq, nk2))
    nop = vals.size // per_op

    # Flat write order (blochsf.f:1214-1222): IE, IOP, IK1, [IQ, IK2 with IK2 fastest]
    blocks = vals.reshape(ne, nop, nk1, nq, nk2)
    # collapse the (NE=1) and sum over equivalent sites IQ (NQ_eff=1 -> no-op)
    blocks = blocks[0].sum(axis=2)          # -> (NOP, NK1, NK2)
    hdr['nop'] = nop

    # k-grid exactly as in blochsf.f:1051-1054 (VECKA=0 default, KSHIFT dropped
    # for plotting; it is a 1e-4 numerical nudge, irrelevant for axes).
    s1 = np.arange(nk1) / (nk1 - 1)
    s2 = np.arange(nk2) / (nk2 - 1)
    v1 = hdr['veck1'] if hdr['veck1'] is not None else np.array([1., 0., 0.])
    v2 = hdr['veck2'] if hdr['veck2'] is not None else np.array([0., 1., 0.])
    # kx,ky as the in-plane projection along the two spanning vectors
    KX = np.outer(s1, np.ones(nk2)) * np.linalg.norm(v1[:2] if v1[2] == 0 else v1)
    KY = np.outer(np.ones(nk1), s2) * np.linalg.norm(v2[:2] if v2[2] == 0 else v2)

    return {'hdr': hdr, 'blocks': blocks, 'KX': KX, 'KY': KY,
            's1': s1, 's2': s2}


def cross_check(inp, d):
    h = d['hdr']
    print('--- cross-check BSF.inp  vs  .bsf header ---')
    issues = []

    def chk(name, a, b):
        ok = (a == b)
        print('  %-12s inp=%-14s bsf=%-14s %s'
              % (name, a, b, 'OK' if ok else '!! MISMATCH'))
        if not ok:
            issues.append(name)

    chk('NK1', inp['nk1'], h['nk1'])
    chk('NK2', inp['nk2'], h['nk2'])

    kw = (h['keyword'] or '').upper()
    if inp['bsfalf']:
        if 'ALF' not in kw:
            issues.append('keyword/BSFALF')
        print('  BSFALF       inp=requested  bsf KEYWORD=%s  %s'
              % (kw, 'OK' if 'ALF' in kw else '!! expected BSF-ALF'))
        if h.get('nop') != 4:
            issues.append('NOP')
        print('  NOP          expected=4     bsf=%s  %s'
              % (h.get('nop'), 'OK' if h.get('nop') == 4 else '!! expected 4'))
    else:
        print('  BSFALF       inp=not set     bsf KEYWORD=%s' % kw)

    if inp['zeroed_l'] is not None:
        print('  ZEROED_L     inp=%s  -> l kept: %s'
              % (inp['zeroed_l'],
                 sorted(set(range(inp['nl'] or 4)) - set(inp['zeroed_l']))))
    if inp['emin'] == 0.0 and inp['emax'] == 0.0:
        print('  energy       E_min=E_max=0 eV -> 2D cut at E_F (E_F=%.5f Ry)'
              % (h['efermi'] if h['efermi'] else float('nan')))
    print('  issues:', issues if issues else 'none')
    return issues


def analyse(d):
    """Velocity field + Alberto equilibrium check (sum_k A_ALF ~ 0)."""
    blocks = d['blocks']
    nop = blocks.shape[0]
    A = blocks[0]
    out = {'A': A}
    if nop >= 4:
        jx, jy, jz = blocks[1], blocks[2], blocks[3]
        good = A > 1e-12
        vx = np.where(good, jx / np.where(good, A, 1.0) * V_FAC, np.nan)
        vy = np.where(good, jy / np.where(good, A, 1.0) * V_FAC, np.nan)
        vz = np.where(good, jz / np.where(good, A, 1.0) * V_FAC, np.nan)
        vF = np.sqrt(vx**2 + vy**2 + vz**2)
        out.update(vx=vx, vy=vy, vz=vz, vF=vF, jx=jx, jy=jy, jz=jz)

        print('--- analysis ---')
        print('  A_BSF      : min=%.3e max=%.3e (negatives clamped in code)'
              % (np.nanmin(A), np.nanmax(A)))
        print('  |v_F|      : median=%.3e  p90=%.3e  max=%.3e  m/s'
              % (np.nanmedian(vF), np.nanpercentile(vF, 90), np.nanmax(vF)))
        # Alberto check: net spectral current over the cut should ~vanish.
        for nm, j in (('jx', jx), ('jy', jy), ('jz', jz)):
            denom = np.sum(np.abs(j))
            r = np.sum(j) / denom if denom > 0 else 0.0
            print('  Sum %s / Sum|%s| = %+.3e  (Alberto: ~0 in equilibrium)'
                  % (nm, nm, r))
    else:
        print('  only %d blocks (no ALF current) -> total BSF only' % nop)
    return out


def _one_map(X, Y, Z, clabel, cmap, xl, yl, save, show, vmin=0, vmax=None):
    if vmax is None:
        vmax = np.nanpercentile(Z, 98)
    fig = plt.figure(figsize=(8.5 / 2.54, 6 / 2.54))
    fig.subplots_adjust(left=1.25 / 8.5, bottom=1.25 / 6,
                        right=1 - 0.25 / 8.5, top=1 - 0.25 / 6)
    ax = fig.add_subplot(111)
    pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap=cmap,
                        vmin=vmin, vmax=vmax)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(clabel)
    cb.outline.set_linewidth(0.5)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_aspect('equal')
    if save:
        fig.savefig(save, bbox_inches='tight')
        print('saved %s' % save)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot(d, res, save=None, frame='native', leaf_pot=None,
         only='both', show=True):
    """only = 'bsf' | 'vf' | 'both'."""
    X, Y, xl, yl = frame_grid(d, leaf_pot, frame)

    if only in ('bsf', 'both'):
        A = d['blocks'][0]
        s = save
        if save and only == 'both':
            s = save.replace('.pdf', '_bsf.pdf')
        _one_map(X, Y, A, r'$A_\mathrm{BSF}$  (states/Ry)', 'viridis',
                 xl, yl, s, show)

    if only in ('vf', 'both'):
        if 'vF' not in res:
            print('no ALF data -> no v_F map')
            return
        s = save
        if save and only == 'both':
            s = save.replace('.pdf', '_vF.pdf')
        _one_map(X, Y, res['vF'] / 1e6,
                 r'$|v_\mathrm{F}|$  ($10^{6}$ m/s)', 'inferno',
                 xl, yl, s, show)


def _autopot(bsf_path):
    """Find the *_final.pot sitting next to the .bsf leaf."""
    g = glob.glob(os.path.join(os.path.dirname(bsf_path) or '.',
                               '*_final.pot'))
    return g[0] if g else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp', help='BSF.inp')
    ap.add_argument('bsf', help='*_ALF.bsf or *_spol.bsf')
    ap.add_argument('--save', default=None, help='output figure path (pdf)')
    ap.add_argument('--frame', default='native', choices=['native', 'l10'],
                    help="'l10': absolute k, cubic A1 rotated into L10 frame")
    ap.add_argument('--only', default='both', choices=['bsf', 'vf', 'both'],
                    help='which map(s) to plot')
    ap.add_argument('--pot', default=None,
                    help='*_final.pot for ALAT (default: auto next to .bsf)')
    a = ap.parse_args()

    inp = read_inp(a.inp)
    d = read_bsf(a.bsf)
    print('inp :', inp)
    print('hdr :', d['hdr'])
    cross_check(inp, d)
    res = analyse(d)
    leaf_pot = a.pot or _autopot(a.bsf)
    plot(d, res, a.save, a.frame, leaf_pot, a.only)


if __name__ == '__main__':
    main()
