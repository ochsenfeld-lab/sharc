#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2023 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************


"""

Standalone script for performing SHARC/FermiONs++ dynamics.

@version: 0.1.0
@author: Martin Peschel
@description: Python Module for SHARC Dynamics using the FermiONs++ python Interface.

"""

# GENERAL PYTHON IMPORTS
import sys
import os
import re
import numpy as np
import argparse
import signal
import traceback  # to get the actual error message after we crash

from time import perf_counter
from overrides import override
from decimal import Decimal
from copy import deepcopy
from collections import OrderedDict
from shutil import copyfile, rmtree
from pathlib import Path


# SHARC INTERFACE
from sharc.pysharc.interface import SHARC_INTERFACE
import sharc_helpers as sharc

# FERMIONS INTERFACE
from Fermions_config import configure_fermions
from PyFermiONs.PyFermiONsInterface import PyFermiONs
from parmed import amber

# ******************************
#
# SHARC_FERMIONS.py functions
#
# ******************************

IToMult = {
    1: 'singlet',
    2: 'doublet',
    3: 'triplet',
    4: 'quartet',
    5: 'quintet',
    6: 'sextet',
    7: 'septet',
    8: 'octet',
    'singlet': 1,
    'doublet': 2,
    'triplet': 3,
    'quartet': 4,
    'quintet': 5,
    'sextet': 6,
    'septet': 7,
    'octet': 8
}


def ml_from_n(n) -> np.array:
    """
    Arguments:  n: multiplicity
    Returns:    np.array of all possible values of m_l
    """
    return np.arange(-(n - 1) / 2, (n - 1) / 2 + 1, 1)


def matrix_size_from_number_of_elements_in_triangular(n: int) -> int:
    """
    Arguments:  n: number of elements in a triangular matrix (excluding the diagonal)
    Returns:    size of the corresponding quare matrix
    """
    return int(1 / 2 + np.sqrt(1 / 4 + 2 * n))


def linear_index_upper_triangular(size: int, index1: int, index2: int) -> int:
    """
    Arguments:  size: size of a square matrix
                index1: row index into the square matrix
                index2: column index into the square matrix
    Returns:    linear index into the corresponding upper triangular matrix (excluding the diagonal)
    """
    return int((size * (size - 1) / 2) - (size - index1) * (
            (size - index1) - 1) / 2 + index2 - index1 - 1)


def key_from_value(mydict: dict, value):
    """
    Arguments:  mydict: dict
                value: value to look for
    Returns:    the key associated with value
    """
    return list(mydict.keys())[list(mydict.values()).index(value)]


def run(command: str) -> str:
    """
    Runs a shell command and returns the result as string
    """
    return os.popen(command).read().rstrip()


def fexp(number: float) -> (list[int], int):
    """
    Get digits and exponent of a number in base 10

    :param number:
    :return: digits: list of 0-9, exponent: int
    """
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return digits, len(digits) + exponent


def fortran_float(num: float, prec: int) -> str:
    """
    Format a floating point number fortran-style

    :param num: number to format (float)
    :param prec: number of decimal digits (int)
    :return: the formatted number (string)
    """
    sign = {0: '+', 1: '-'}
    sign2 = {0: '0', 1: '-'}
    (digits, exponent) = fexp(num)
    string = (sign2[num < 0] + '.' + ''.join(map(str, digits[0:prec]))
              + 'D' + sign[exponent < 0] + '%02d' % abs(exponent))
    return string


def checkscratch(scratchdir: Path):
    """Checks whether SCRATCHDIR is a file or directory.
    If a file, it quits with exit code 1, if its a directory, it passes.
    If SCRATCHDIR does not exist, tries to create it.

    Arguments:
    1 string: path to SCRATCHDIR
    """
    exist = os.path.exists(scratchdir)
    if exist:
        isfile = os.path.isfile(scratchdir)
        if isfile:
            print('$SCRATCHDIR=%s exists and is a file!' % scratchdir)
            sys.exit(16)
    else:
        try:
            os.makedirs(scratchdir)
        except OSError:
            print('Can not create SCRATCHDIR=%s\n' % scratchdir)
            sys.exit(17)


class TableFormatter:
    """
    Given elements of a table via __call__ a formatted table can be obtained via __str__
    """

    def __init__(self, ncol: int, sep: str = ''):
        """
        :param ncol: number of columns
        :param sep: separator between columns
        """
        self.table = ''
        self.ncol = ncol
        self.col = 0
        self.sep = sep

    def __call__(self, element: str):
        self.col += 1
        if self.col == self.ncol:
            self.col = 0
            self.table += element + '\n'
        else:
            self.table += element + self.sep

    def __str__(self) -> str:
        if self.col != 0:
            return self.table + '\n'
        return self.table


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
        table = TableFormatter(4)
        for i in range(nocc):
            for j in range(nvirt):
                table(fortran_float(factor * state[j][i], 14))
        string += str(table)
    return string


def read_tmol_basis(basisfile: Path):
    """
    Read a turbomole basis set file and extract some information

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

    with basisfile.open() as f:
        basis_lines = f.readlines()

    for line in basis_lines:

        # this is an empty line or a comment
        if line.isspace():
            continue

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


def format_tmol_control(elements, nstates, basis_name, basis_info, nocc, nvirt):
    """
    Format a tmol control file as an input to cis_nto
    """
    string = "$coord    file=coord\n$atoms\n"
    elem_positions = {}
    nvirt_cartesian = nvirt
    additional_cartesians = {"s": 0, "p": 0, "d": 1, "f": 3}
    for elem in elements:
        elem_positions.update({elem: []})
    for i, elem in enumerate(elements):
        elem_positions[elem].append(str(i + 1))
        for l, norb_for_this_l in basis_info[elem].items():
            nvirt_cartesian += norb_for_this_l * additional_cartesians[l]

    for elem, position in elem_positions.items():
        string += f"{elem}  {','.join(elem_positions[elem])} \\\nbasis ={elem} {basis_name[elem]} \\\njbas  ={elem} universal\n"
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

    string += f"   natoms={len(elements)}\n"
    string += f"   nbf(CAO)={nocc + nvirt_cartesian}\n"
    string += f"   nbf(AO)={nocc + nvirt}\n"
    string += "$closed shells\n"
    string += f" a       1-{nocc}                                   ( 2 )\n"
    string += "$end"
    return string


def format_mo_tmol(mo, atoms, basis):
    """
    Format a TUrbomole mo file as input for cis_nto from the basis set information and the fermions mo coefficients
    """
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
        table = TableFormatter(4)
        for atom in atoms:
            for l_index, (l, norb_for_this_l) in enumerate(basis[atom].items()):
                p = permute_ml[l]
                s = adjust_sign_ml[l]
                for n in range(norb_for_this_l):
                    for ml in range(2 * l_index + 1):
                        table(fortran_float(s[ml] * mo[j + p[ml]][i], 14))
                    j += 2 * l_index + 1
        string += str(table)
    return string


def elements_from_mol(mol):
    return [line[0].lower() for line in mol]


def create_gradmap(grad, statemap):
    gradmap = dict()
    for i in grad:
        gradmap[tuple(statemap[i][0:2])] = statemap[i]
    return {i: v for i, v in enumerate(gradmap.values())}


def copy_restart():
    primary_dir = os.getenv('PRIMARY_DIR')
    run(f"rsync -r output* {primary_dir}/.")
    run(f"rsync -r restart* {primary_dir}/.")


def setup_fermions(mol, additional_options=None):
    """
    Set up the Fermions interface

    :param additional_options: additional options for Fermions (these can overwrite configure, units and reorient so be careful)
    :param mol: molecular geometry
    :return: PyFermiONs object, tdscf options, tdscf_deriv options
    """
    fermions = PyFermiONs(mol)
    options = configure_fermions(fermions)
    if additional_options:
        for key, value in additional_options.items():
            if key in options:
                options[key] += "\n" + value
            else:
                options[key] = value
    # 'bohr' is fundementally broken with qmmm, so we only allow it via additional options
    fermions.units = 'angstrom'
    # reorient must be false for dynamics, otherwise wierd stuff happens, so we only allow it via additional options
    fermions.reorient = False
    # it seems to be necessary to write an inpcrd file and read it to properly initialize qmmm
    if "qmmm_sys" in options:
        # For some reason we have to write and read an .inpcrd file
        amb_inpcrd = amber.AmberAsciiRestart("tmp.inpcrd", mode="w")
        amb_inpcrd.coordinates = [[i[1], i[2], i[3]] for i in mol]
        amb_inpcrd.close()
        with open("tmp.inpcrd", 'r') as f:
            inpcrd_in = f.read()
        fermions.set_qmmm(options["qmmm_sys"], options["mm_env"], options["prmtop"], inpcrd_in=inpcrd_in, pdb_in=None)
    fermions.set_scratch()
    fermions.init(extra_sys=options["sys"])

    if fermions.units != 'angstrom':
        print('ERROR: in Fermions, \'bohr\' seems to be fundementally broken with qmmm. Do not use it.')
        sys.exit()

    if fermions.reorient:
        print('ERROR: Dynamics must be run with \'reorient\' set to \'false\'.')
        sys.exit()

    fermions.setup()
    return fermions, options


class CisNto:

    def __init__(self, path, basis, mol, savedir):
        self.path = Path(path)
        self.basis = Path(basis)
        self.savedir = Path(savedir)
        self.basis_name, self.basis_info = read_tmol_basis(self.basis)
        self.elements = elements_from_mol(mol)

    def _get_dirname(self, i):
        return self.savedir.joinpath(f"cis_nto_{i}")

    def make_directory(self, i):
        mydir = self._get_dirname(i)
        mydir.mkdir(parents=True, exist_ok=True)
        copyfile(self.basis, mydir.joinpath('basis'))

    def save_coord(self, mol, i):
        elements = elements_from_mol(mol)
        if elements != self.elements:
            print("Error, CisNto: incompatible molecule.")
            sys.exit(648263)
        string = "$coord\n"
        a2au = 1.e0 / 0.52917726e0
        for line in mol:
            string += ("%.14f" % (float(line[1]) * a2au)).rjust(20)
            string += ("%.14f" % (float(line[2]) * a2au)).rjust(24)
            string += ("%.14f" % (float(line[3]) * a2au)).rjust(24)
            string += line[0].lower().rjust(5)
            string += "\n"
        string += "$end"
        mydir = self._get_dirname(i)
        with open(mydir.joinpath('coord'), "w") as f:
            f.write(string)

    def save_mo(self, mo, i):
        mydir = self._get_dirname(i)
        with open(mydir.joinpath('mos'), "w") as f:
            f.write(format_mo_tmol(mo, self.elements, self.basis_info))

    def save_dets(self, dets, i, exc_energies):
        mydir = self.savedir.joinpath(f"cis_nto_{i}")
        with open(mydir.joinpath('ciss_a'), "w") as f:
            f.write(format_dets_tmol(dets, exc_energies))
        with open(mydir.joinpath('control'), "w") as f:
            f.write(format_tmol_control(self.elements, len(dets), self.basis_name, self.basis_info,
                                        dets[0].shape[1], dets[0].shape[0]))

    def get_overlap(self, i, j):
        cisovl_out = run(
            str(self.path) + f" {self.savedir.joinpath(f'cis_nto_{i}')} {self.savedir.joinpath(f'cis_nto_{j}')}")
        print(cisovl_out)
        m = re.search(r'Raw overlap matrix:(.*)Writing WF overlap matrix', cisovl_out, re.DOTALL)
        ovl_strings = m.groups(1)[0].split('\n')[1:-1]
        ovl_matrix = np.float_([x.split() for x in ovl_strings])
        return ovl_matrix

    def delete_old_directories(self, i, nr_of_steps_to_keep=5):
        if i > nr_of_steps_to_keep:
            mydir = self._get_dirname(i - nr_of_steps_to_keep)
            if os.path.exists(mydir):
                rmtree(mydir)

    def prepare_input(self, i, coord, mo, dets, exc_energies):
        self.delete_old_directories(i)
        self.make_directory(i)
        self.save_coord(coord, i)
        self.save_mo(mo, i)
        self.save_dets(dets, i, exc_energies)


class SharcFermions(SHARC_INTERFACE):
    """
    Class for SHARC FERMIONS
    """
    # Name of the interface
    interface = 'FERMIONS'
    # store atom ids
    save_atids = True
    # store atom names
    save_atnames = True
    # accepted units:  0 : Bohr, 1 : Angstrom
    iunit = 0
    # not supported keys
    not_supported = ['nacdt', 'dmdr']

    @override
    def __init__(self, *args, **kwargs):
        # Internal variables for Fermions, these are set by self.setup
        self.fermions = None
        self.fermions_options = {}
        self.cisnto = {}
        self.mults = None
        self.qm_region = slice(None, None, None)

        # Additional variables for communication in file-based mode
        self.file_based = False
        self.parentpid = None
        self.has_crashed = False
        self.restart_step = -1

        self.n_scf_failed = 0

        # Currently we can only do tda, with singlet reference and excitations up to triplet
        # TODO: enforce this correctly
        self.method = 'tda'  # Read this from file once there is more than one possibility
        self.mult_ref = 'singlet'  # Read this from file once there is more than one possibility

    @override
    def final_print(self):
        """
        Shuts down fermions gracefully
        if in file based mode, sends also wake-up signal to runQM.sh
        and terminate successfully/unseccessfully
        """
        if self.restart_step > 0 and not self.has_crashed:
            copy_restart()
        print("pysharc_fermions.py: **** Shutting down FermiONs++ ****")
        self.fermions.finish()
        sys.stdout.flush()
        if self.file_based:
            if self.has_crashed:
                os.kill(self.parentpid, signal.SIGUSR2)
                sys.exit(1)
            else:
                os.kill(self.parentpid, signal.SIGUSR1)
                sys.exit(0)

    @override
    def crash_function(self):
        """
        If something creshes, print the traceback and shut down fermions gracefully
        """
        super(SharcFermions, self).crash_function()
        self.has_crashed = True
        traceback.print_exc()
        self.final_print()

    @override
    def readParameter(self, *args, **kwargs):
        """
        If we get killed somehow we still try to shut down fermions
        therefore, initialize an appropriate trap for SIGTERM Signal

        If we are in file_based mode, create a trap so that we can be
        "woken up" by SIGUSR1. In this case, we do not continue pysharc execution.
        Instead, we are now governed by the wake-up signals received from runQM.sh
        and continue directly with the qm_calculation
        """
        signal.signal(signal.SIGTERM, lambda sig, frame: self.crash_function())
        self.restart_step = kwargs['restart_step']
        if kwargs['file_based']:
            self.file_based = True
            signal.signal(signal.SIGUSR1, lambda sig, frame: None)
            with open("python.pid", "w") as f:
                f.write(str(os.getpid()))
            self.main_loop()

    def pre_qm_calculation(self, qmin_filename):
        """
        reads QMin and does some pre-processing, should only be called in file_based mode
        """
        with open("run.sh.pid", "r") as f:
            self.parentpid = int(f.readlines()[0])
        qm_in = sharc.readQMin(qmin_filename)
        if 'grad' in qm_in:
            qm_in['gradmap'] = create_gradmap(qm_in['grad'], qm_in['statemap'])
        else:
            qm_in['gradmap'] = {}
        return qm_in

    def main_loop(self):
        """
        Our propagation loop in file_based mode
        """
        with open("runSHARC.sh.pid", "r") as f:
            startpid = int(f.readlines()[0])
        os.kill(startpid, signal.SIGUSR1)

        while True:
            # Wait for signal from SHARC
            signal.pause()

            # Do the calculation
            qm_in = self.pre_qm_calculation("QM/QM.in")
            self.step = int(qm_in['step'][0])
            QMout = self.sharc_qm_failure_handle(qm_in, [i[1:] for i in qm_in['geo']])
            sharc.writeQMout(qm_in, QMout, "QM/QM.in")
            if self.step == self.nsteps:
                self.final_print()

            # Send signal to SHARC to contnue
            os.kill(self.parentpid, signal.SIGUSR1)
            sys.stdout.flush()

    def crd_to_mol(self, coords):
        """
        helper function to convert coordinates
        """
        return [[atname.lower(), self.constants['au2a'] * crd[0], self.constants['au2a'] * crd[1],
                 self.constants['au2a'] * crd[2]] for (atname, crd) in zip(self.AtNames, coords)]

    @override
    def do_qm_job(self, tasks, Crd):
        """
        Here we perform the qm calculations depending on the tasks, that were asked
        """

        tstart = perf_counter()
        mol = self.crd_to_mol(Crd)

        # GET TASKS FROM QM_IN,
        if self.file_based:
            qm_in = tasks
        else:
            qm_in = self.parse_tasks(tasks)
            qm_in['states'] = self.states['states'] #no idea if this breaks when states are not incuded in the dynamics

        # GET ALL MULTIPLICITIES
        if not self.mults:
            self.mults = set()
            for i, state in enumerate(qm_in['states'], 1):
                if state != 0:
                    self.mults.add(IToMult[i])

        # INITIALIZE FERMIONS
        if not self.fermions:
            print("pysharc_fermions.py: **** Starting FermiONs++ ****")
            sys.stdout.flush()
            self.fermions, self.fermions_options = setup_fermions(mol, additional_options={"sys" : "die_in_scf false"})
            if self.fermions.qmmm:
                # TODO: this does not work for non-continuous QM regions or definitions via residues
                m = re.search(r'qm\s*=\s*\{a(\d+)\s*-\s*(\d+)}', self.fermions.qmmm_sys, re.IGNORECASE)
                if not m:
                    sys.exit("Sorry, Could not read QM-System Definition, Definition either wrong, "
                             "or is more complicated than i implemented in SHARC_FERMIONS...")
                self.qm_region = slice(int(m.group(1)) - 1, int(m.group(2)))
        else:
            # Some funny behaviour: reinit want the coordinates in a differnt format and in bohrs
            if 'samestep' not in qm_in:
                self.fermions.reinit(np.array(Crd).flatten())

        # INITIALIZE CISNTO FOR OVERLAP CALCULATION, FOL QMMM OVERLAP SHOULD ONLY BE CALCULATED FOR QM REGION
        if not self.cisnto:
            for mult in self.mults:
                self.cisnto[mult] = CisNto("$CIS_NTO/cis_overlap.exe", basis="basis", mol=mol[self.qm_region],
                                           savedir=Path(self.savedir).joinpath(mult))

        # INITILIZE THE MATRIZES
        if 'samestep' not in qm_in:
            self._h = np.zeros([qm_in['nmstates'], qm_in['nmstates']], dtype=complex)
            self._dipole = np.zeros([3, qm_in['nmstates'], qm_in['nmstates']], dtype=complex)
            self._overlap = np.eye(qm_in['nmstates'])
            self._grad = [np.zeros([len(self.fermions.mol), 3]).tolist() for _ in range(qm_in['nmstates'])]

            # GROUND STATE CALCULATION
            # We currently assume that the reference state is in position 0
            # This might break, for example, with SF-TDDFT
            # TODO: Once something like this is possible in fermions, fix...
            self._h[0, 0], gradient_0, dipole_0 = self.calc_groundstate(not bool(qm_in['gradmap']))
            if gradient_0.size != 0:
                self._grad[0] = gradient_0.tolist()
                self._dipole[:, 0, 0] = dipole_0

        # EXCITED STATE CALCULATION
        if qm_in['nmstates'] > 1:

            # EXCITATION ENERGIES
            if 'samestep' not in qm_in:
                self._exc_state, exc_energies, tda_amplitudes = self.calc_exc_states()
                for i, mult, index, _ in self.iter_exc_states(qm_in['statemap']):
                    self._h[i, i] = self._h[0, 0] + exc_energies[mult][index]

            # EXCITED STATE GRADIENTS AND EXCITED STATE DIPOLE MOMENTS
            for _, mult, index, _ in self.iter_exc_states(qm_in['gradmap']):
                forces_ex = self._exc_state.tdscf_forces_nacs(do_grad=True, nacv_flag=False, method=self.method,
                                                        spin=mult, trg_state=index + 1,
                                                        py_string=self.fermions_options['tdscf_deriv'])
                state_dipole = np.array(self._exc_state.state_mm(mult, index, 1)[1:]) / self.constants['au2debye']
                if self.fermions.qmmm:
                    forces_ex = self.fermions.globals.get_FILES().read_double_sub(len(self.fermions.mol) * 3, 0,
                                                                                  'qmmm_exc_forces', 0)
                for ml in ml_from_n(IToMult[mult]):
                    i = key_from_value(qm_in['statemap'], [IToMult[mult], index + 1 + (mult == self.mult_ref), ml]) - 1
                    self._grad[i] = np.array(forces_ex).reshape(len(self.fermions.mol), 3).tolist()
                    self._dipole[:, i, i] = state_dipole

            # TRANSITION DIPOLE MOMENTS
            if 'dm' in qm_in and 'samestep' not in qm_in:
                tdm_0n = np.array(self._exc_state.get_transition_dipoles_0n(method=self.method)) \
                         / self.constants['au2debye']
                tdm = {}
                nstates = {}
                for mult in self.mults:
                    tdm[mult] = np.array(self._exc_state.get_transition_dipoles_mn(method=self.method, st=IToMult[mult])) \
                                / self.constants['au2debye']
                    nstates[mult] = matrix_size_from_number_of_elements_in_triangular(int(len(tdm[mult]) / 3))

                for i, mult, index, ms in self.iter_exc_states(qm_in['statemap']):
                    if mult == self.mult_ref:
                        self._dipole[:, 0, i] = tdm_0n[(3 * index):(3 * index + 3)]
                        self._dipole[:, i, 0] = self._dipole[:, 0, i]
                    for j, mult2, index2, ms2 in self.iter_exc_states(qm_in['statemap']):
                        if index2 > index and mult == mult2 and ms == ms2:
                            cindex = linear_index_upper_triangular(nstates[mult], index, index2)
                            self._dipole[:, i, j] = tdm[mult][(3 * cindex):(3 * cindex + 3)]
                            self._dipole[:, j, i] = self._dipole[:, i, j]

            # SPIN ORBIT COUPLINGS
            if 'soc' in qm_in and 'samestep' not in qm_in:
                soc_0n = np.array(self._exc_state.get_soc_s02tx(self.method))
                soc_mn = np.array(self._exc_state.get_soc_sy2tx(self.method))
                size_soc = np.sqrt(len(soc_mn) / 3)
                for i, mult, index, ms_index in self.iter_exc_states(qm_in['statemap']):
                    if mult != self.mult_ref:
                        self._h[0, i] = soc_0n[3 * index + ms_index]
                        self._h[i, 0] = np.conj(self._h[0, i])
                        for j, mult2, index2, _ in self.iter_exc_states(qm_in['statemap']):
                            if mult2 != mult:
                                self._h[j, i] = soc_mn[3 * int(index2 * size_soc + index) + ms_index]
                                self._h[i, j] = np.conj(self._h[j, i])

            # OVERLAP CALCULATION
            if 'samestep' not in qm_in:
                for mult in self.mults:
                    self.cisnto[mult].prepare_input(self.step, mol[self.qm_region], self.fermions.load("mo"),
                                                tda_amplitudes[mult], exc_energies[mult])
                if 'overlap' in qm_in:
                    ovl = {}
                    for mult in self.mults:
                        ovl[mult] = self.cisnto[mult].get_overlap(self.step, self.step - 1)
                    # !!! The iteration works different for overlap,
                    # TODO: this is very ugly, create a proper iterator and dont calc the overlap of the reference for triplets
                    for i, state1 in qm_in['statemap'].items():
                        for j, state2 in qm_in['statemap'].items():
                            if state1[0] == state2[0] and state1[2] == state2[2]:
                                if IToMult[state1[0]] == "triplet":
                                    self._overlap[i - 1, j - 1] = ovl[IToMult[state1[0]]][state1[1], state2[1]]
                                else:
                                    self._overlap[i - 1, j - 1] = ovl[IToMult[state1[0]]][state1[1] - 1, state2[1] - 1]

        # ASSIGN EVERYTHING TO QM_OUT
        runtime = perf_counter() - tstart
        qm_out = {'h': self._h.tolist(), 'dm': self._dipole.tolist(), 'overlap': self._overlap.tolist(), 'grad': self._grad, 'runtime': runtime}

        # Phases from overlaps
        if 'phases' in qm_in:
            qm_out['phases'] = [complex(1., 0.) for _ in range(qm_in['nmstates'])]
            if 'overlap' in qm_out:
                for i in range(qm_in['nmstates']):
                    if qm_out['overlap'][i][i].real < 0.:
                        qm_out['phases'][i] = complex(-1., 0.)
            else:
                print("ERROR: no phases without overlap")
                sys.exit(25982)

        # Copy restart and output (this is kind of shitty)
        if self.restart_step > 0:
            if (self.step-1) % self.restart_step == 0:
                print(f"STEP {self.step}, copying restart directory")
                copy_restart()
            else:
                print(f"STEP {self.step}, NOT copying restart directory")
        else:
            print(f"STEP {self.step}, restart_step not set for copying")

        return qm_out

    def calc_groundstate(self, only_energy):
        energy_gs, forces_gs = self.fermions.calc_energy_forces_MD(mute=0, timeit=False, only_energy=only_energy)
        if self.fermions.md_scf.get_error() > self.fermions.globals.get_Qvals().get_double("convergence_com"):
            print(f"ERROR: problems in SCF... {self.fermions.md_scf.get_error()}")
            self.n_scf_failed += 1
            #After 10 failed SCF we raise an exception
            if self.n_scf_failed > 10:
                raise Exception("SCF Problems.")
            else:
                print(f"Ignoring Convergence Problems and continuing... Ignored already {self.n_scf_failed} times.")

        else:
            self.n_scf_failed = 0

        if only_energy:
            return np.array(energy_gs), np.array([]), np.array([])
        else:
            return np.array(energy_gs), np.array(forces_gs).reshape(len(self.fermions.mol),
                                                                    3), self.fermions.calc_dipole_MD()

    def calc_exc_states(self):
        exc_state = self.fermions.get_excited_states(self.fermions_options['tdscf'])
        exc_state.evaluate()

        # get excitation energies
        exc_energies = {}
        for mult in self.mults:
            exc_energies[mult] = exc_state.get_exc_energies(method=self.method, st=mult)

        # save amplitudes for overlap calculation
        tda_amplitudes = {}
        for mult in self.mults:
            tda_amplitudes[mult] = []
            for index in range(len(exc_energies[mult])):
                tda_amplitude, _ = self.fermions.load_td_amplitudes(td_method=self.method, td_spin=mult,
                                                                    td_state=index + 1)
                tda_amplitudes[mult].append(tda_amplitude)
        return exc_state, exc_energies, tda_amplitudes

    def iter_exc_states(self, statemap):
        for state_index, state in statemap.items():
            mult = IToMult[state[0]]
            fermions_index = state[1] - 1
            if mult == self.mult_ref:
                if fermions_index == 0:
                    continue  # reference state is treated differently from all other states
                else:
                    fermions_index = fermions_index - 1  # because the reference state is treated differently, adjust the numbering
            yield state_index - 1, mult, fermions_index, int(state[2] + 1)

    def parse_tasks(self, tasks):
        """
        these things should be interface dependent

        so write what you love, it covers basically everything
        after savedir information in QMin

        """

        # find init, samestep, restart
        qm_in = dict((key, value) for key, value in self.QMin.items())
        qm_in['natom'] = self.NAtoms

        key_tasks = tasks['tasks'].lower().split()

        if any([self.not_supported in key_tasks]):
            print("not supported keys: ", self.not_supported)
            sys.exit(16)

        for key in key_tasks:
            qm_in[key] = []

        for key in ['grad', 'nacdr']:
            if tasks[key].strip() != "":
                qm_in[key] = []

        for key in self.states:
            qm_in[key] = self.states[key]

        if 'init' in qm_in:
            checkscratch(qm_in['savedir'])

        # Process the gradient requests
        if 'grad' in tasks:
            if len(tasks['grad']) == 0 or tasks['grad'][0] == 'all':
                qm_in['grad'] = [i + 1 for i in range(self.states['nmstates'])]
            else:
                for i in tasks['grad'].split():
                    try:
                        k = int(i)
                    except ValueError:
                        print('Arguments to keyword "grad" must be "all" or a list of integers!')
                        sys.exit(53)
                    qm_in['grad'].append(k)
                    if k > self.states['nmstates']:
                        print(
                            'State for requested gradient does not correspond to any state in QM input file state list!')
                        sys.exit(54)

        # get the set of states for which gradients actually need to be calculated
        qm_in['gradmap'] = create_gradmap(qm_in['grad'], qm_in['statemap'])

        return qm_in


def get_commandline():
    """
        Get Commando line option with argpase

    """
    parser = argparse.ArgumentParser("Perform SHARC FERMIONS++ calculations")
    parser.add_argument("input", metavar="FILE", type=str, default="input", nargs='?', help="input file")
    parser.add_argument('--file_based', action=argparse.BooleanOptionalAction, help='poduce QM.out?')
    parser.add_argument('--restart_step', type=int, required=False, default=-1, help='how often to copy restart and output to the primary directory.')
    parser.add_argument('--singlepoint', action=argparse.BooleanOptionalAction, help='singlepoint calculation?')
    args = parser.parse_args()

    return args.input, args.file_based, args.restart_step, args.singlepoint


def main():
    """
        Main Function if program is called as standalone

    """
    inp_file, file_based, restart_step, singlepoint = get_commandline()
    # init SHARC_FERMIONS class
    interface = SharcFermions()
    if singlepoint:
        interface.file_based = True
        qm_in = interface.pre_qm_calculation("QM.in")
        interface.step = 0
        interface.AtNames = [i[0] for i in qm_in['geo']]
        interface.constants = { 'au2a': 0.529177, 'au2debye': 2.5417482144 } #its kind of stupid to have the constants in two places...
        primary_dir = os.getenv('PRIMARY_DIR')
        interface.savedir = primary_dir #we might want to change this at some point
        QMout = interface.do_qm_job(qm_in, [i[1:] for i in qm_in['geo']])
        sharc.writeQMout(qm_in, QMout, "QM.in")
        interface.final_print()
    else:
        interface.run_sharc(inp_file, file_based=file_based, restart_step=restart_step)


if __name__ == "__main__":
    main()