#!/usr/bin/env python3
import sys
import os
import numpy as np
import re
from decimal import Decimal
from pathlib import Path
from copy import deepcopy
import shutil
import argparse
from Fermions_config import configure_fermions
from PyFermiONs.fermions_helpers import read_xyz
from PyFermiONs.PyFermiONsInterface import PyFermiONs
from collections import OrderedDict
from joblib import Parallel, delayed
from parmed import amber


class StringsInColumns:

    def __init__(self, ncol: int, sep=''):
        """
        :param ncol: number of columns
        :param sep: separator between columns
        """
        self.ncol = ncol
        self.col = 0
        self.sep = sep

    def __call__(self, element: str):
        self.col += 1
        if self.col == self.ncol:
            self.col = 0
            return element + '\n'
        return element + self.sep

    def end(self):
        if self.col != 0:
            return '\n'
        return ''


def run(command):
    return os.popen(command).read().rstrip()


def fexp(number):
    """
    Get digits and exponent of a number in base 10

    :param number:
    :return: digits: list of 0-9, exponent: int
    """
    (sign, digits, exponent) = Decimal(number).as_tuple()

    return digits, len(digits) + exponent


# TODO: find a better way to do the formatting
def fortran_float(num, prec: int) -> str:
    """

    :param num: number to format (float)
    :param prec: number of decimal digits (int)
    :return:
    """
    sign = {0: '+', 1: '-'}
    sign2 = {0: '0', 1: '-'}
    (digits, exponent) = fexp(num)
    string = sign2[num < 0]
    string += '.'
    string += ''.join(map(str, digits[0:prec]))
    string += 'D'
    string += sign[exponent < 0]
    string += '%02d' % abs(exponent)

    return string


def setup(mol, additional_options=None):
    """
    Set up the Fermions interface

    :param additional_options: additional options for Fermions (these can overwrite configure, units and reorient so be careful)
    :param mol: molecular geometry
    :return: PyFermiONs object, tdscf options, tdscf_deriv options
    """

    Fermions = PyFermiONs(mol)
    options = configure_fermions(Fermions)
    if additional_options:
        for key, value in additional_options:
            if key in options:
                options[key] += "\n" + value
            else:
                options[key] = value
    # 'bohr' is fundementally broken with qmmm, so we only allow it via additional options
    Fermions.units = 'angstrom'
    # reorient must be false for dynamics, otherwise wierd stuff happens, so we only allow it via additional options
    Fermions.reorient = False
    # it seems to be necessary to write an inpcrd file and read it to properly initialize qmmm
    if "qmmm_sys" in options:
        # For some reason we have to write and read an .inpcrd file
        amb_inpcrd = amber.AmberAsciiRestart("tmp.inpcrd", mode="w")
        amb_inpcrd.coordinates = [[i[1], i[2], i[3]] for i in mol]
        amb_inpcrd.close()
        with open("tmp.inpcrd", 'r') as f:
            inpcrd_in = f.read()
        Fermions.set_qmmm(options["qmmm_sys"], options["mm_env"], options["prmtop"], inpcrd_in=inpcrd_in, pdb_in=None)
    Fermions.set_scratch()
    Fermions.init(extra_sys=options["sys"])
    Fermions.setup()
    return Fermions, options["tdscf"], options["tdscf_deriv"]


def extract_wfovl(ovl, wfovl_out, state1, state2):
    """
    Extract the relevant elements from the wfoverlap output and write them to the ovl matrix

    :param ovl: overlap matrix (np.array, nstates x nstates)
    :param wfovl_out: output of wfoverlap program for the calculation of <state1|state2> (string)
    :param state1: index of state1 (int)
    :param state2: index of state2 (int)
    :return: nothing
    """
    lines = wfovl_out.splitlines()
    rfloat = r'([+-]?\d+\.\d+)'
    rem2 = re.compile(r'<PsiA\s*2\|\s*' + rfloat + r'\s*' + rfloat)

    # extract full matrix
    if state1 == state2:
        rem1 = re.compile(r'<PsiA\s*1\|\s*' + rfloat + r'\s*' + rfloat)
        for line in lines:
            m1 = re.match(rem1, line)
            if m1:
                ovl[0, 0] = m1.group(1)
                ovl[0, state1] = m1.group(2)
            m2 = re.match(rem2, line)
            if m2:
                ovl[state1, 0] = m2.group(1)
                ovl[state1, state1] = m2.group(2)
                return
    # extract only second diagonal element
    else:
        for line in lines:
            m2 = re.match(rem2, line)
            if m2:
                ovl[state1, state2] = m2.group(2)
                return

    print("Error extracting wfoverlap output.")
    sys.exit(3434)


def format_dets_tmol(dets, eigenvalues, symmetry="c1"):
    """
    format fermions tda amplitudes to Turbomole ciss format

    :param dets: tda amplitudes as produces by Fermions.load_td_amplitudes, list (nstates) of np.array (nvirt x nocc)
    :param eigenvalues: excitation energies, np.array(nstates)
    :param symmetry: symmetry of molecule, string in tmol format default "c1"
    :return: formatted determinant file (string)
    """

    factor = np.sqrt(2)  # in turbomole all amplitudes are multiplied by sqrt(2)

    # Everything we need for the first line
    nocc = dets[0].shape[1]
    nvirt = dets[0].shape[0]

    header = """$title
$symmetry {symmetry}
$tensor space dimension{ndim}
$scfinstab ciss
$current subspace dimension{nstates}
$current iteration converged
$eigenpairs
"""
    string = header.format(symmetry=symmetry, ndim='{:9d}'.format(nocc * nvirt), nstates='{:9d}'.format(len(dets)))

    for s, state in enumerate(dets):
        string += '{:9d}   '.format(s + 1) + 'eigenvalue =  ' + fortran_float(eigenvalues[s], 16) + '\n'
        columnize = StringsInColumns(4)
        for i in range(nocc):
            for j in range(nvirt):
                string += columnize(fortran_float(factor * state[j][i], 14))
        string += columnize.end()
    return string


def read_tmol_basis(basisfile):
    """
    For each element, get the number of functions for each l quantum-number (l in ['s', 'p', 'd', 'f'])
    For each element, get the name of the basis set
    """

    # TODO! check if basis sets are always ordered s,p,d,f,g,...
    # TODO! implement for higher angular momenta than f

    element = ''
    regex_l = r'([spdfg])'
    # Ordered dict, such that we can calc multiplicity from key position
    zeros_dict_l = OrderedDict([('s', 0), ('p', 0), ('d', 0), ('f', 0)])
    basis_info = {}
    basis_name = {}

    with open(basisfile) as f:
        basis_lines = f.readlines()

    for line in basis_lines:
        # this is a comment
        if line.lstrip()[0] == '#':
            continue

        # this is the element name
        m = re.match(r'^(\w\w?)\s+(.+)', line)
        if m:
            element = m.group(1)
            basis_name[element] = m.group(2)
            basis_info[element] = deepcopy(zeros_dict_l)

        # this is the start of an ao
        m = re.match(r'^\s*\d+\s+' + regex_l, line)
        if m:
            if m.group(1) == 'g':
                print('Basis set: g-functions or higher not yet implemented.')
                sys.exit(3469)
            basis_info[element][m.group(1)] += 1

        # here we can read the ao parameters
        pass  # we don't need the actual parameters (yet)

    return basis_name, basis_info


def format_tmol_control(mol, nstates, basis_name, basis_info, nocc, nvirt):
    string = "$coord    file=coord\n$atoms\n"
    elem_positions = {}
    nvirt_cartesian = nvirt
    additional_cartesians = {"s": 0, "p": 0, "d": 1, "f": 3}
    for line in mol:
        elem_positions.update({line[0].lower(): []})
    for i, line in enumerate(mol):
        elem_positions[line[0].lower()].append(str(i + 1))
        for l, norb_for_this_l in basis_info[line[0].lower()].items():
            nvirt_cartesian += norb_for_this_l * additional_cartesians[l]

    for elem, basis in basis_name.items():
        string += f"{elem}  {','.join(elem_positions[elem])} \\\nbasis ={elem} {basis} \\\njbas  ={elem} universal\n"
    string += """$basis    file=basis
$scfmo   file=mos
$dft
   functional bh-lyp
   gridsize   m5
$scfinstab ciss
$soes    
"""
    string += f"a            {nstates}\n"
    string += "$rundimensions\n"

    string += f"   natoms={len(mol)}\n"
    string += f"   nbf(CAO)={nocc + nvirt_cartesian}\n"
    string += f"   nbf(AO)={nocc + nvirt}\n"
    string += "$closed shells\n"
    string += f" a       1-{nocc}                                   ( 2 )\n"
    string += "$end"
    return string


def format_mo_tmol(mo, atoms, basis):
    # we need to reorder p,d,f orbitals
    # That's how we need to reorder and adjust the signs
    permute_ml = {"s": [0], "p": [0, 1, 2], "d": [2, 3, 1, 0, 4], "f": [3, 4, 2, 1, 5, 6, 0]}
    adjust_sign_ml = {"s": [1], "p": [1, 1, 1], "d": [1, 1, 1, 1, 1], "f": [1, 1, 1, 1, 1, 1, -1]}

    # we do the actual printing
    string = """$scfmo    scfconv=7   format(4d20.14)
# SCF total energy is     xxx a.u.
#
"""
    nmo = mo.shape[0]
    for i in range(nmo):
        string += '{:6d}  a      '.format(i + 1) + 'eigenvalue=  ' + fortran_float(0.0, 14) + f"   nsaos={nmo}\n"
        j = 0
        columnize = StringsInColumns(4)
        for atom in atoms:
            for l_index, (l, norb_for_this_l) in enumerate(basis[atom.lower()].items()):
                p = permute_ml[l]
                s = adjust_sign_ml[l]
                for n in range(norb_for_this_l):
                    for ml in range(2 * l_index + 1):
                        string += columnize(fortran_float(s[ml] * mo[j + p[ml]][i], 14))
                    j += 2 * l_index + 1
        string += columnize.end()
    return string


def format_dets(dets, wfthres):
    """
    format sharc tda amplitudes to Columbus? determinant format

    :param dets: tda amplitudes as produces by Fermions.load_td_amplitudes (np.array (nvirt x nocc))
    :param wfthres: wavefunction cutoff, the largest determinants are considered until |Psi|^2 > wfthresh (float)
    :return: formatted determinant file (string)
    """
    # Everything we need for the first line
    nocc = dets.shape[1]
    nvirt = dets.shape[0]
    norb = nocc + nvirt
    ndets = 1

    fformat = "\t%.7f"
    # fformat = "\t%e"

    # Closed shell determinant
    string = nocc * "d" + nvirt * "e" + (fformat % 1) + (fformat % 0) + "\n"

    # All other determinants
    to_sort = []
    for oi in range(nocc):
        for vi in range(nvirt):
            # every determinant appears once for an excited an electron and once for an excited b electron
            string_a = oi * "d" + "a" + (nocc - oi - 1) * "d" + vi * "e" + "b" + (
                    nvirt - vi - 1) * "e" + fformat % 0 + fformat % dets[vi, oi]
            string_b = oi * "d" + "b" + (nocc - oi - 1) * "d" + vi * "e" + "a" + (
                    nvirt - vi - 1) * "e" + fformat % 0 + fformat % dets[vi, oi]
            to_sort.append([dets[vi, oi], string_a, string_b])

    to_sort.sort(key=lambda x: x[0] ** 2, reverse=True)
    wfsum = 0.0
    for i in to_sort:
        wfsum += 2 * i[0] ** 2
        string += i[1] + "\n" + i[2] + "\n"
        ndets += 2
        if wfsum > wfthres:
            break

    return f"2 {norb} {ndets}\n" + string


def format_mo(mo_a, mo_b, nao, nmo, restr=True):
    """
    taken from SHARC_ORCA.py, format fermions mo to Columbus format

    :param mo_a: alpha-mo matrix as returned by Fermions.load("mo") (numpy-array, nbf x norb)
    :param mo_b: beta-mo matrix as returned by Fermions.load("mo"), can be empty when restricted (numpy-array, nbf x norb)
    :param nao: number of atomic orbitals (int)
    :param nmo: number of molecular orbitals (int)
    :param restr: set to false if these are unrestricted orbitals (bool)
    :return: formatted mos (string)
    """
    mo_a = np.transpose(mo_a)
    if not restr:
        mo_b = np.transpose(mo_b)

    string = '''2mocoef
header
 1
MO-coefficients from Fermions
 1
 %i   %i
 a
mocoef
(*)
''' % (nao, nmo)
    x = 0
    for imo, mo in enumerate(mo_a):
        for c in mo:
            if x >= 3:
                string += '\n'
                x = 0
            string += '% 6.12e ' % c
            x += 1
        if x > 0:
            string += '\n'
            x = 0
    if not restr:
        x = 0
        for imo, mo in enumerate(mo_b):
            for c in mo:
                if x >= 3:
                    string += '\n'
                    x = 0
                string += '% 6.12e ' % c
                x += 1
            if x > 0:
                string += '\n'
                x = 0
    string += 'orbocc\n(*)\n'
    x = 0
    for i in range(nmo):
        if x >= 3:
            string += '\n'
            x = 0
        string += '% 6.12e ' % 0.0
        x += 1

    return string


def double_ao_overlap(mol1, mol2, fname=None):
    # Calculate Double AO Overlap
    mol_sum = mol1 + mol2
    Fermions_ovl, _, _ = setup(mol_sum, additional_options=={"sys": "safety_checks false\nri_j false"})
    overlap = Fermions_ovl.load("metric")
    norb = int(overlap.shape[0] / 2)

    # print upper right quadrant of Double AO Overlap
    if fname:
        np.savetxt(fname, overlap[norb:, :norb], header=f"{norb} {norb}", comments='')
    Fermions_ovl.finish()
    return overlap[norb:, :norb]


class WFOverlap:

    def __init__(self, path, mol1, mol2, wfthres: float):
        self.path = path
        self.wfthres = wfthres
        self.nroots = 0
        _ = double_ao_overlap(mol1, mol2, fname="aoovl")

    @staticmethod
    def save_mo(coeffs, i):
        Path(f"wfovl_{i}").mkdir(exist_ok=True)
        norb = coeffs.shape[0]
        with open(f"wfovl_{i}/mo", "w") as f:
            f.write(format_mo(mo_a=coeffs, mo_b='', nao=norb, nmo=norb, restr=True))

    def save_dets(self, dets, i, exc_energies):
        self.nroots = len(dets)
        for state, det in enumerate(dets):
            with open(f"wfovl_{i}/det.{state + 1}", "w") as f:
                f.write(format_dets(det, self.wfthres))

    def get_overlap(self, i, j):

        wfovl_inp = """mix_aoovl=aoovl
                a_mo={mo0}
                b_mo={mo1}
                a_det={det0}
                b_det={det1}
                a_mo_read=0
                b_mo_read=0
                ao_read=0
                force_direct_dets"""

        ovl_matrix = np.zeros([nroots + 1, nroots + 1])
        fnames = []
        for state1 in range(1, nroots + 1):
            for state2 in range(1, nroots + 1):
                # write wfoverlap input
                fnames.append(f"wfovl_{j}/cioverlap{state1}_{state2}.input")
                with open(fnames[-1], "w") as f:
                    f.write(wfovl_inp.format(mo0=f"wfovl_{i}/mo", mo1=f"wfovl_{j}/mo",
                                             det0=f"wfovl_{i}/det.{state1}", det1=f"wfovl_{j}/det.{state2}"))
        wfovl_out = Parallel(n_jobs=32)(delayed(run)(f"{self.path} -f {name}") for name in fnames)
        for state1 in range(1, nroots + 1):
            for state2 in range(1, nroots + 1):
                extract_wfovl(ovl_matrix, wfovl_out[(state1 - 1) * nroots + state2 - 1], state1, state2)
        return ovl_matrix


class CisNto:

    def __init__(self, path, mol1, mol2, i, j, basis):
        self.path = path
        self.mol = {i: mol1, j: mol2}
        self.basis = basis
        self.basis_name, self.basis_info = read_tmol_basis(basis)

    def save_mo(self, coeffs, i):
        Path(f"cis_nto_{i}").mkdir(exist_ok=True)
        shutil.copyfile(self.basis, f"cis_nto_{i}/basis")
        string = "$coord\n"
        for line in self.mol[i]:
            string += ("%.14f" % (float(line[1]) * 1 / sharc.au2a)).rjust(20)
            string += ("%.14f" % (float(line[2]) * 1 / sharc.au2a)).rjust(24)
            string += ("%.14f" % (float(line[3]) * 1 / sharc.au2a)).rjust(24)
            string += line[0].lower().rjust(5)
            string += "\n"
        string += "$end"
        with open(f"cis_nto_{i}/coord", "w") as f:
            f.write(string)
        with open(f"cis_nto_{i}/mos", "w") as f:
            f.write(format_mo_tmol(coeffs, [element[0] for element in self.mol[i]], self.basis_info))

    def save_dets(self, dets, i, exc_energies):
        with open(f"cis_nto_{i}/ciss_a", "w") as f:
            f.write(format_dets_tmol(dets, exc_energies))
        with open(f"cis_nto_{i}/control", "w") as f:
            f.write(format_tmol_control(self.mol[i], len(dets), self.basis_name, self.basis_info,
                                        dets[0].shape[1], dets[0].shape[0]))

    def get_overlap(self, i, j):
        cisovl_out = run(self.path + f" cis_nto_{i} cis_nto_{j}")
        print(cisovl_out)
        m = re.search(r'Raw overlap matrix:(.*)Writing WF overlap matrix', cisovl_out, re.DOTALL)
        ovl_strings = m.groups(1)[0].split('\n')[1:-1]
        ovl_matrix = np.float_([x.split() for x in ovl_strings])
        return ovl_matrix


def main():
    """
    Print the overlap matrix <Psi(t1)|Psi(t2)> using Fermions TDA
    Also calculates SCF-energies and excitation energies for both Psi(t1), Psi(t2)
    """

    parser = argparse.ArgumentParser(description='Print the overlap matrix <Psi(t1)|Psi(t2)> using Fermions TDA.\
    Also calculates SCF-energies and excitation energies for both Psi(t1), Psi(t2)')

    parser.add_argument('xyz1', type=str, help='xyz-file for first structure')
    parser.add_argument('xyz2', type=str, help='xyz-file for second structure')
    parser.add_argument('--program', type=str,
                        help='program to use to calculate overlap, either wfoverlap or cis_nto. Default: cis_nto',
                        default='cis_nto')
    parser.add_argument('--wfovl_path', type=str,
                        help='path to wfoverlap executable. Default: $SHARC/wfoverlap.x',
                        default='$SHARC/wfoverlap.x')
    parser.add_argument('--cis_nto_path', type=str,
                        help='path to cis_overlap executable. Default: $CIS_NTO/cis_overlap.exe',
                        default='$CIS_NTO/cis_overlap.exe')
    parser.add_argument('--wfthres', type=float,
                        help='wavefunction cutoff for wfoverlap. Default: 0.995', default=0.995)
    parser.add_argument('--basis', type=str,
                        help='turbomole basis set file for cis_nto. Default: basis', default='basis')
    args = parser.parse_args()

    # Read xyz(t-dt), xyz(t)
    mol1 = read_xyz(args.xyz1)
    mol2 = read_xyz(args.xyz2)

    if args.program == 'wfoverlap':
        program = WFOverlap(args.wfovl_path, mol1, mol2, wfthres=args.wfthres)
    elif args.program == 'cis_nto':
        program = CisNto(args.cis_nto_path, mol1, mol2, 0, 1, basis=args.basis)
    else:
        print(f"ERROR: overlap calculation not implemented for: {overlap_program}")
        sys.exit(12324)

    for i, mol in enumerate([mol1, mol2]):
        Fermions, tdscf_options, tdscf_deriv_options = setup(mol)

        # Ground state energy
        the_scf = Fermions.get_scf()
        the_scf.doSCF()
        energy_gs = the_scf.get_SCFenergy()
        print(f"SCF-Energy: {energy_gs}")

        # alpha-MO-coefficients (numpy-array, nbf x norb),
        # for beta-MOs in unrestricted calculations `Fermions.load("mo", "b")`
        coeffs = Fermions.load("mo")
        program.save_mo(coeffs, i)

        # Excited state energies
        exc_state = Fermions.get_excited_states(tdscf_options)
        exc_state.evaluate()
        exc_energies = exc_state.get_exc_energies(method='tda', st='singlet')
        print(f"Excitation Energies: {exc_energies}")

        tda_amplitudes = []
        for state in range(1, exc_state.get_nroots() + 1):
            # load_td_amplitudes returns a Tuple (X, Y), for TDA Y is `None`
            tda_amplitude, _ = Fermions.load_td_amplitudes(td_method="tda", td_spin="singlet", td_state=state)
            tda_amplitudes.append(tda_amplitude)

        program.save_dets(tda_amplitudes, i, exc_energies)
        Fermions.finish()

    ovl_matrix = program.get_overlap(0, 1)

    print("Overlap Matrix:")
    with np.printoptions(precision=7, suppress=False):
        print(ovl_matrix)


if __name__ == "__main__":
    main()
