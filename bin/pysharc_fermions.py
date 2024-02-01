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

import sys
import os
import re
from time import perf_counter
import argparse

import numpy as np
from overrides import override

from sharc.pysharc.interface import SHARC_INTERFACE
from Fermions_wfoverlap import CisNto, setup

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


def ml_from_n(n):
    return np.arange(-(n - 1) / 2, (n - 1) / 2 + 1, 1)


def matrix_size_from_number_of_elements_in_upper_triangular(n):
    return int(1 / 2 + np.sqrt(1 / 4 + 2 * n))


def linear_index_upper_triangular(size, index1, index2):
    return int((size * (size - 1) / 2) - (size - index1) * (
            (size - index1) - 1) / 2 + index2 - index1 - 1)


def key_from_value(mydict, value):
    return list(mydict.keys())[list(mydict.values()).index(value)]


def checkscratch(scratchdir):
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


def run_cisnto(fermions, exc_energies, tda_amplitudes, geo_old, geo, step_old: int, step: int, savedir=''):
    # if we do qmmm we need to only give the qm region to calc the overlap
    if fermions.qmmm:
        # TODO: this does not work for non-continuous QM regions or definitions via residues
        m = re.search(r'qm\s*=\s*\{a(\d+)\s*-\s*(\d+)}', fermions.qmmm_sys, re.IGNORECASE)
        if not m:
            sys.exit("Sorry, Could not read QM-System Definition, Definition either wrong, "
                     "or is more complicated than i implemented in SHARC_FERMIONS...")
        qm_slice = slice(int(m.group(1)) - 1, int(m.group(2)))
        program = CisNto("$CIS_NTO/cis_overlap.exe", geo_old[qm_slice], geo[qm_slice], step_old, step,
                         basis="basis", savedir=savedir)
    else:
        program = CisNto("$CIS_NTO/cis_overlap.exe", geo_old, geo, step_old, step, basis="basis", savedir=savedir)
    program.save_mo(fermions.load("mo"), step)
    program.save_dets(tda_amplitudes, step, exc_energies)
    return program.get_overlap(step_old, step)


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
        self.tdscf_options = None
        self.tdscf_deriv_options = None

        # Currently we can only do tda, with singlet reference and excitations up to triplet
        # TODO: enforce this correctly
        self.method = 'tda'  # Read this from file once there is more than one possibility
        self.mult_ref = 'singlet'  # Read this from file once there is more than one possibility
        self.mults = None

        # Internal variables used for convenience
        self.geo_step = {}  # here, we save all the geometries --> might be unnecessary

    @override
    def final_print(self):
        print("pysharc_fermions.py: **** Shutting down FermiONs++ ****")
        sys.stdout.flush()
        self.fermions.finish()

    @override
    def crash_function(self):
        super(SharcFermions, self).crash_function()
        self.final_print()

    def crd_to_mol(self, coords):
        return [[atname.lower(), self.constants['au2a'] * crd[0], self.constants['au2a'] * crd[1],
                 self.constants['au2a'] * crd[2]] for (atname, crd) in zip(self.AtNames, coords)]

    @override
    def do_qm_job(self, tasks, Crd):
        """

        Here you should perform all your qm calculations

        depending on the tasks, that were asked

        """
        qm_in = self.parse_tasks(tasks)
        mol = self.crd_to_mol(Crd)

        # Initialize Fermions with the current geometry
        if not self.fermions:
            print("pysharc_fermions.py: **** Starting FermiONs++ ****")
            sys.stdout.flush()
            self.fermions, self.tdscf_options, self.tdscf_deriv_options = setup(mol)
        else:
            # Some funny behaviour: reinit want the coordinates in a differnt format and in bohrs
            self.fermions.reinit(np.array(Crd).flatten())

        self.mults = set()
        for i, state in enumerate(self.states['states'], 1):
            if state != 0:
                self.mults.add(IToMult[i])

        # Store the current geometry
        self.geo_step[self.step] = mol

        # Check if we have the geometry of the last step, if not, try reading it from the restart directory
        if ((self.step - 1) not in self.geo_step) and ('init' not in qm_in):
            print("Reading the geometry is not yet implemented (and should be unnecessary)."
                  " But for now, we produce an Error.")
            sys.exit(1244)

        # Run the calculation
        qm_out = self.get_qm_out(qm_in)
        return qm_out

    def calc_groundstate(self, only_energy):
        energy_gs, forces_gs = self.fermions.calc_energy_forces_MD(mute=0, timeit=False, only_energy=only_energy)
        if only_energy:
            return np.array(energy_gs), np.array([]), np.array([])
        else:
            return np.array(energy_gs), np.array(forces_gs).reshape(len(self.fermions.mol),
                                                                    3), self.fermions.calc_dipole_MD()

    def calc_exc_states(self):
        exc_state = self.fermions.get_excited_states(self.tdscf_options)
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

    def get_qm_out(self, qm_in):

        """Calculates the MCH Hamiltonian, SOC matrix, overlap matrix, gradients, DM"""

        tstart = perf_counter()

        # INITILIZE THE MATRIZES
        h = np.zeros([qm_in['nmstates'], qm_in['nmstates']], dtype=complex)
        dipole = np.zeros([3, qm_in['nmstates'], qm_in['nmstates']], dtype=complex)
        overlap = np.eye(qm_in['nmstates'])

        # GROUND STATE CALCULATION
        # We currently assume that the reference state is in position 0
        # This might break, for example, with SF-TDDFT
        # TODO: Once something like this is possible in fermions, fix...
        h[0, 0], gradient_0, dipole_0 = self.calc_groundstate(not bool(qm_in['gradmap']))
        if gradient_0.size != 0:
            grad = [[] for _ in range(qm_in['nmstates'])]
            grad[0] = gradient_0.tolist()
            dipole[:, 0, 0] = dipole_0

        # EXCITED STATE CALCULATION
        if qm_in['nmstates'] > 1:

            # EXCITATION ENERGIES
            exc_state, exc_energies, tda_amplitudes = self.calc_exc_states()
            for i, mult, index, _ in self.iter_exc_states(qm_in['statemap']):
                h[i, i] = h[0, 0] + exc_energies[mult][index]

            # EXCITED STATE GRADIENTS AND EXCITED STATE DIPOLE MOMENTS
            for _, mult, index, _ in self.iter_exc_states(qm_in['gradmap']):
                forces_ex = exc_state.tdscf_forces_nacs(do_grad=True, nacv_flag=False, method=self.method,
                                                        spin=mult, trg_state=index + 1,
                                                        py_string=self.tdscf_deriv_options)
                state_dipole = np.array(exc_state.state_mm(index, 1)[1:]) / self.constants['au2debye']
                if self.fermions.qmmm:
                    forces_ex = self.fermions.globals.get_FILES().read_double_sub(len(self.fermions.mol) * 3, 0,
                                                                                  'qmmm_exc_forces', 0)
                for ml in ml_from_n(IToMult[mult]):
                    i = key_from_value(qm_in['statemap'], [IToMult[mult], index + 1 + (mult == self.mult_ref), ml]) - 1
                    grad[i] = np.array(forces_ex).reshape(len(self.fermions.mol), 3).tolist()
                    dipole[:, i, i] = state_dipole

            # TRANSITION DIPOLE MOMENTS
            if 'dm' in qm_in:
                tdm_0n = np.array(exc_state.get_transition_dipoles_0n(method=self.method)) \
                         / self.constants['au2debye']
                tdm = {}
                nstates = {}
                for mult in self.mults:
                    tdm[mult] = np.array(exc_state.get_transition_dipoles_mn(method=self.method, st=IToMult[mult])) \
                                / self.constants['au2debye']
                    nstates[mult] = matrix_size_from_number_of_elements_in_upper_triangular(len(tdm[mult]) / 3)

                for i, mult, index, ms in self.iter_exc_states(qm_in['statemap']):
                    if mult == self.mult_ref:
                        dipole[:, 0, i] = tdm_0n[(3 * index):(3 * index + 3)]
                        dipole[:, i, 0] = dipole[:, 0, i]
                    for j, mult2, index2, ms2 in self.iter_exc_states(qm_in['statemap']):
                        if index2 < index and mult == mult2 and ms == ms2:
                            cindex = linear_index_upper_triangular(nstates[mult], index, index2)
                            dipole[:, i, j] = tdm[mult][(3 * cindex):(3 * cindex + 3)]
                            dipole[:, j, i] = dipole[:, i, j]

            # SPIN ORBIT COUPLINGS
            if 'soc' in qm_in:
                soc_0n = np.array(exc_state.get_soc_s02tx(self.method))
                soc_mn = np.array(exc_state.get_soc_sy2tx(self.method))
                size_soc = np.sqrt(len(soc_mn) / 3)
                for i, mult, index, ms_index in self.iter_exc_states(qm_in['statemap']):
                    if mult != self.mult_ref:
                        h[0, i] = soc_0n[3 * index + ms_index]
                        h[i, 0] = np.conj(h[0, i])
                        for j, mult2, index2, _ in self.iter_exc_states(qm_in['statemap']):
                            if mult2 != mult:
                                h[j, i] = soc_mn[3 * int(index2 * size_soc + index) + ms_index]
                                h[i, j] = np.conj(h[j, i])

            if 'init' in qm_in:
                for mult in self.mults:
                    _ = run_cisnto(self.fermions, exc_energies[mult], tda_amplitudes[mult], self.geo_step[0],
                                   self.geo_step[0], 0, 0, savedir=os.path.join(self.savedir, mult))

            if 'overlap' in qm_in:
                ovl = {}
                for mult in self.mults:
                    ovl[mult] = run_cisnto(self.fermions, exc_energies[mult], tda_amplitudes[mult],
                                           self.geo_step[self.step - 1],
                                           self.geo_step[self.step],
                                           self.step - 1, self.step,
                                           savedir=os.path.join(self.savedir, mult))
                for i, mult, index, ms in self.iter_exc_states(qm_in['statemap']):
                    for j, mult2, index2, ms2 in self.iter_exc_states(qm_in['statemap']):
                        if mult == mult2 and ms == ms2:
                            overlap[i, j] = ovl[mult][index, index2]

        print("H")
        print(h)
        print("DM")
        print(dipole)
        if gradient_0.size != 0:
            print("GRAD")
            print(grad)
        sys.stdout.flush()
        # derp

        # ASSIGN EVERYTHING TO QM_OUT
        qm_out = {'h': h.tolist(), 'dm': dipole.tolist(), 'overlap': overlap.tolist()}
        if gradient_0.size != 0:
            qm_out['grad'] = grad
        tstop = perf_counter()
        qm_out['runtime'] = tstop - tstart

        return qm_out

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

        for key in self.states:
            qm_in[key] = self.states[key]

        if 'init' in qm_in:
            checkscratch(qm_in['savedir'])

        for key in ['grad', 'nacdr']:
            if tasks[key].strip() != "":
                qm_in[key] = []

        # Process the gradient requests
        if 'grad' in qm_in:
            if len(qm_in['grad']) == 0 or qm_in['grad'][0] == 'all':
                qm_in['grad'] = [i + 1 for i in range(qm_in['nmstates'])]
            else:
                for i in range(len(qm_in['grad'])):
                    try:
                        qm_in['grad'][i] = int(qm_in['grad'][i])
                    except ValueError:
                        print('Arguments to keyword "grad" must be "all" or a list of integers!')
                        sys.exit(53)
                    if qm_in['grad'][i] > qm_in['nmstates']:
                        print(
                            'State for requested gradient does not correspond to any state in QM input file state list!')

        # get the set of states for which gradients actually need to be calculated
        gradmap = dict()
        if 'grad' in qm_in:
            for i in qm_in['grad']:
                gradmap[i] = qm_in['statemap'][i]
        qm_in['gradmap'] = gradmap

        return qm_in


def get_commandline():
    """
        Get Commando line option with argpase

    """

    parser = argparse.ArgumentParser("Perform SHARC LVC calculations")
    parser.add_argument("input", metavar="FILE", type=str,
                        default="input", nargs='?',
                        help="input file")
    parser.add_argument("param", metavar="FILE", type=str,
                        default="QM/LVC.template", nargs='?',
                        help="param file, LVC.template")
    args = parser.parse_args()

    return args.input, args.param


def main():
    """
        Main Function if program is called as standalone

    """

    inp_file, param = get_commandline()
    # init SHARC_FERMIONS class
    interface = SharcFermions()
    # run sharc dynamics
    interface.run_sharc(inp_file, param)


if __name__ == "__main__":
    main()
