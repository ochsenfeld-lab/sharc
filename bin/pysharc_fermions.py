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


def key_from_value(mydict, value):
    return list(mydict.keys())[list(mydict.values()).index(value)]


def get_res(res, key, index, default='Error: Value not found'):
    """
  Looks if we have the result and returns it if it's there
  otherwise returns default value

  also takes care of (anti-)hermiticity; i.e. if we have (0,1) of an
  (anti-)hermitian matrix, we also have (1,0)
  """

    # special treatment for the matrices
    if key == 'nacv':
        if (index[1], index[0], key) in res:
            x = (-1) * np.conj(res[(index[1], index[0], key)])  # anti-hermitian matrix
        elif (index[0], index[1], key) in res:
            x = res[(index[0], index[1], key)]
        else:
            return default
        additional_indices = index[2:]
    elif key == 'soc' or key == 'dm':
        if (index[1], index[0], key) in res:
            x = np.conj(res[(index[1], index[0], key)])  # hermitian matrix
        elif (index[0], index[1], key) in res:
            x = res[(index[0], index[1], key)]
        else:
            return default
        additional_indices = index[2:]
    # default case for scalars
    else:
        if (index[0], key) in res:
            x = res[(index[0], key)]
        else:
            return default
        additional_indices = index[1:]
    for i in additional_indices:
        x = x[i]
    return x


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
    # TODO: find a list of all keys and check whch one we actually cant support
    not_supported = ['nacdt', 'dmdr']

    @override
    def __init__(self, *args, **kwargs):
        """
        Init your interface, best is you

        set parameter files etc. for read

        """
        self.fermions = None
        self.tdscf_options = None
        self.tdscf_deriv_options = None
        self.method = 'tda'
        self.geo_step = {}

    @override
    def final_print(self):
        print("pysharc_fermions.py: **** Shutting down FermiONs++ ****")
        sys.stdout.flush()
        self.fermions.finish()

    def crd_to_mol(self, coords):
        return [[atname.lower(), self.constants['au2a'] * crd[0], self.constants['au2a'] * crd[1],
                 self.constants['au2a'] * crd[2]] for (atname, crd) in zip(self.AtNames, coords)]

    @override
    def do_qm_job(self, tasks, Crd):
        """

        Here you should perform all your qm calculations

        depending on the tasks, that were asked

        """
        QMin = self.parse_tasks(tasks)
        mol = self.crd_to_mol(Crd)

        # Initialize Fermions with the current geometry
        if not self.fermions:
            print("pysharc_fermions.py: **** Starting FermiONs++ ****")
            sys.stdout.flush()
            self.fermions, self.tdscf_options, self.tdscf_deriv_options = setup(mol)
        else:
            # Some funny behaviour: reinit want the coordinates in a differnt format and in bohrs
            self.fermions.reinit(np.array(Crd).flatten())

        # Store the current geometry
        self.geo_step[self.step] = mol

        # Run the calculation
        QMout = self.get_qm_out(QMin)

        return QMout

    def calc_groundstate(self, energy_only):
        energy_gs, forces_gs = self.fermions.calc_energy_forces_MD(mute=0, timeit=False, only_energy=energy_only)
        if energy_only:
            return np.array(energy_gs), None
        else:
            return np.array(energy_gs), np.array(forces_gs).reshape(len(self.fermions.mol), 3)

    def get_gradient(self, qm_in):

        energy, gradient = self.calc_groundstate(False)
        qm_out = {(1, 'energy'): energy,
                  (1, 'gradient'): gradient,
                  (1, 1, 'dm'): np.array(self.fermions.calc_dipole_MD())}

        # EXCITED STATE CALCULATION
        if qm_in['nmstates'] > 1:
            exc_state = self.fermions.get_excited_states(self.tdscf_options)
            exc_state.evaluate()

            # get excitation energies
            exc_energies_singlet = exc_state.get_exc_energies(method=self.method, st='singlet')
            exc_energies_triplet = exc_state.get_exc_energies(method=self.method, st='triplet')
            for state in range(2, qm_in['nmstates'] + 1):
                mult = IToMult[qm_in['statemap'][state][0]]
                index = qm_in['statemap'][state][1] - 1
                if mult == 'singlet':
                    index = index - 1
                    qm_out[(state, 'energy')] = qm_out[(1, 'energy')] + exc_energies_singlet[index]
                elif mult == 'triplet':
                    qm_out[(state, 'energy')] = qm_out[(1, 'energy')] + exc_energies_triplet[index]
                else:
                    print('ERROR: Not implemented for multiplicity: ', mult)
                    sys.exit()

            # save dets for wfoverlap
            tda_amplitudes = {'singlet': [], 'triplet': []}
            for index in range(len(exc_energies_singlet)):
                tda_amplitude, _ = self.fermions.load_td_amplitudes(td_method=self.method, td_spin='singlet',
                                                                    td_state=index + 1)
                tda_amplitudes['singlet'].append(tda_amplitude)
            for index in range(len(exc_energies_triplet)):
                tda_amplitude, _ = self.fermions.load_td_amplitudes(td_method=self.method, td_spin='triplet',
                                                                    td_state=index + 1)
                tda_amplitudes['triplet'].append(tda_amplitude)

            # calculate excited state gradients and excited state dipole moments
            for state in qm_in['gradmap']:

                # We have already calculated the groundstate + gradient beforehand
                if state == (1, 1):
                    continue

                mult = IToMult[state[0]]
                index = state[1]
                if mult == 'singlet':
                    index = index - 1
                forces_ex = exc_state.tdscf_forces_nacs(do_grad=True, nacv_flag=False, method=self.method,
                                                        spin=mult, trg_state=index,
                                                        py_string=self.tdscf_deriv_options)
                # if we do qmmm we need to read a different set of forces
                if self.fermions.qmmm:
                    forces_ex = self.fermions.globals.get_FILES().read_double_sub(len(self.fermions.mol) * 3, 0,
                                                                                  'qmmm_exc_forces', 0)
                for ml in ml_from_n(state[0]):
                    snr = key_from_value(qm_in['statemap'], [state[0], state[1], ml])
                    qm_out[(snr, 'gradient')] = np.array(forces_ex).reshape(len(self.fermions.mol), 3)
                    # we only get state dipoles for the states where we calc gradients
                    qm_out[(snr, snr, 'dm')] = np.array(exc_state.state_mm(index - 1, 1)[1:]) * 1 / self.constants[
                        'au2debye']

            # calculate transition dipole moments
            if 'dm' in qm_in:
                tdm_0n = np.array(exc_state.get_transition_dipoles_0n(method=self.method)) * 1 / self.constants[
                    'au2debye']
                tdm_singlet = np.array(exc_state.get_transition_dipoles_mn(method=self.method, st=1)) * 1 / \
                              self.constants[
                                  'au2debye']
                tdm_triplet = np.array(exc_state.get_transition_dipoles_mn(method=self.method, st=3)) * 1 / \
                              self.constants[
                                  'au2debye']
                size_singlet = 1 / 2 + np.sqrt(1 / 4 + 2 / 3 * len(tdm_singlet))
                size_triplet = 1 / 2 + np.sqrt(1 / 4 + 2 / 3 * len(tdm_triplet))

                for n in range(2, qm_in['nmstates'] + 1):
                    # TDMs with ground state
                    mult_n = IToMult[qm_in['statemap'][n][0]]
                    if mult_n == 'singlet':
                        index = qm_in['statemap'][n][1] - 2
                        qm_out[(1, n, 'dm')] = tdm_0n[3 * index:3 * index + 3]
                    else:
                        # The lowest state should always be a singlet --> tdm's to states of other multiplicity are 0
                        # qm_out[(1, n, 'dm')] = 0.0
                        pass

                    # TDMs between excited states
                    for m in range(n + 1, qm_in['nmstates'] + 1):
                        mult_m = IToMult[qm_in['statemap'][m][0]]
                        if mult_m == 'singlet' and mult_n == 'singlet':
                            index1 = qm_in['statemap'][n][1] - 2
                            index2 = qm_in['statemap'][m][1] - 2
                            cindex = int((size_singlet * (size_singlet - 1) / 2) - (size_singlet - index1) * (
                                    (size_singlet - index1) - 1) / 2 + index2 - index1 - 1)
                            qm_out[(m, n, 'dm')] = tdm_singlet[3 * cindex:3 * cindex + 3]
                        elif mult_m == 'triplet' and mult_n == 'triplet':
                            index1 = qm_in['statemap'][n][1] - 1
                            index2 = qm_in['statemap'][m][1] - 1
                            if index1 != index2:
                                cindex = int((size_triplet * (size_triplet - 1) / 2) - (size_triplet - index1) * (
                                        (size_triplet - index1) - 1) / 2 + index2 - index1 - 1)
                                qm_out[(m, n, 'dm')] = tdm_triplet[3 * cindex:3 * cindex + 3]
                            else:
                                # tdm's between triplets with the same n are 0
                                # qm_out[(m, n, 'dm')] = 0.0
                                pass
                        else:
                            # tdm's between states of differing multiplicity are 0
                            # qm_out[(m, n, 'dm')] = 0.0
                            pass

            if 'soc' in qm_in:

                soc_0n = np.array(exc_state.get_soc_s02tx(self.method))
                soc_mn = np.array(exc_state.get_soc_sy2tx(self.method))

                # TODO: This is wrong for non-qual number of singlets and triplets (?currently not possible in fermions?)
                size_soc = np.sqrt(len(soc_mn) / 3)

                for n in range(2, qm_in['nmstates'] + 1):
                    # SOCs with ground state
                    mult_n = IToMult[qm_in['statemap'][n][0]]
                    if mult_n == 'triplet':
                        index = qm_in['statemap'][n][1] - 1
                        ms_index = int(qm_in['statemap'][n][2] + 1)
                        qm_out[(1, n, 'soc')] = soc_0n[3 * index + ms_index]
                    else:
                        pass

                    # SOCs between excited states
                    for m in range(2, qm_in['nmstates'] + 1):
                        mult_m = IToMult[qm_in['statemap'][m][0]]
                        index1 = qm_in['statemap'][n][1] - 1
                        index2 = qm_in['statemap'][m][1] - 2
                        if mult_m == 'singlet' and mult_n == 'triplet':
                            ms_index = int(qm_in['statemap'][n][2] + 1)
                            cindex = int(index2 * size_soc + index1)
                            qm_out[(m, n, 'soc')] = soc_mn[3 * cindex + ms_index]
                        else:
                            pass

            if 'init' in qm_in:
                _ = run_cisnto(self.fermions, exc_energies_singlet, tda_amplitudes['singlet'], self.geo_step[0],
                               self.geo_step[0], 0, 0, savedir=self.savedir + "/singlet")
                _ = run_cisnto(self.fermions, exc_energies_triplet, tda_amplitudes['triplet'], self.geo_step[0],
                               self.geo_step[0], 0, 0, savedir=self.savedir + "/triplet")

            if 'overlap' in qm_in:
                overlap_singlet = run_cisnto(self.fermions, exc_energies_singlet, tda_amplitudes['singlet'],
                                             self.geo_step[self.step - 1],
                                             self.geo_step[self.step],
                                             self.step - 1, self.step,
                                             savedir=self.savedir + "/singlet")
                overlap_triplet = run_cisnto(self.fermions, exc_energies_triplet, tda_amplitudes['triplet'],
                                             self.geo_step[self.step - 1],
                                             self.geo_step[self.step],
                                             self.step - 1, self.step,
                                             savedir=self.savedir + "/triplet")
                qm_out['overlap'] = np.zeros([qm_in['nmstates'], qm_in['nmstates']])
                for n in range(1, qm_in['nmstates'] + 1):
                    mult_n = IToMult[qm_in['statemap'][n][0]]
                    for m in range(1, qm_in['nmstates'] + 1):
                        mult_m = IToMult[qm_in['statemap'][m][0]]
                        if mult_n == 'singlet' and mult_m == 'singlet':
                            index1 = qm_in['statemap'][m][1] - 1
                            index2 = qm_in['statemap'][n][1] - 1
                            qm_out['overlap'][m - 1][n - 1] = overlap_singlet[index1][index2]
                        if mult_n == 'triplet' and mult_m == 'triplet':
                            ms1 = qm_in['statemap'][m][2]
                            ms2 = qm_in['statemap'][n][2]
                            if ms1 == ms2:
                                index1 = qm_in['statemap'][m][1]
                                index2 = qm_in['statemap'][n][1]
                                qm_out['overlap'][m - 1][n - 1] = overlap_triplet[index1][index2]
                            else:
                                pass
                        else:
                            pass

        return qm_out

    def get_qm_out(self, QMin):

        """Calculates the MCH Hamiltonian, SOC matrix ,overlap matrix, gradients, DM"""

        QMout = {}
        tstart = perf_counter()

        fermions_grad = self.get_gradient(QMin)

        grad = []
        for istate in range(1, QMin['nmstates'] + 1):
            grad.append([])
            for iat in range(QMin['natom']):
                x = get_res(fermions_grad, 'gradient', [istate, iat, 0], default=0.0)
                y = get_res(fermions_grad, 'gradient', [istate, iat, 1], default=0.0)
                z = get_res(fermions_grad, 'gradient', [istate, iat, 2], default=0.0)
                grad[-1].append([x, y, z])

        dipole = np.zeros([3, QMin['nmstates'], QMin['nmstates']], dtype=complex)
        for xyz in range(3):
            for i in range(1, QMin['nmstates'] + 1):
                for j in range(1, QMin['nmstates'] + 1):
                    dipole[xyz, i - 1, j - 1] = get_res(fermions_grad, 'dm', [i, j, xyz], default=0)

        Hfull = np.zeros([QMin['nmstates'], QMin['nmstates']], dtype=complex)
        for istate in range(1, QMin['nmstates'] + 1):
            for jstate in range(1, QMin['nmstates'] + 1):
                Hfull[istate - 1][jstate - 1] = get_res(fermions_grad, 'soc', [istate, jstate], default=0)
            Hfull[istate - 1][istate - 1] = get_res(fermions_grad, 'energy', [istate])

        # assign QMout elements
        QMout['h'] = Hfull.tolist()
        QMout['dm'] = dipole.tolist()
        QMout['grad'] = grad

        tstop = perf_counter()
        QMout['runtime'] = tstop-tstart

        if 'overlap' in QMin:
            QMout['overlap'] = fermions_grad['overlap'].tolist()

        return QMout

    def parse_tasks(self, tasks):
        """
        these things should be interface dependent

        so write what you love, it covers basically everything
        after savedir information in QMin

        """

        # find init, samestep, restart
        QMin = dict((key, value) for key, value in self.QMin.items())
        QMin['natom'] = self.NAtoms

        key_tasks = tasks['tasks'].lower().split()

        if any([self.not_supported in key_tasks]):
            print("not supported keys: ", self.not_supported)
            sys.exit(16)

        for key in key_tasks:
            QMin[key] = []

        for key in self.states:
            QMin[key] = self.states[key]

        if 'init' in QMin:
            checkscratch(QMin['savedir'])

        for key in ['grad', 'nacdr']:
            if tasks[key].strip() != "":
                QMin[key] = []

        QMin['pwd'] = os.getcwd()

        # Process the gradient requests
        if 'grad' in QMin:
            if len(QMin['grad']) == 0 or QMin['grad'][0] == 'all':
                QMin['grad'] = [i + 1 for i in range(QMin['nmstates'])]
            else:
                for i in range(len(QMin['grad'])):
                    try:
                        QMin['grad'][i] = int(QMin['grad'][i])
                    except ValueError:
                        print('Arguments to keyword "grad" must be "all" or a list of integers!')
                        sys.exit(53)
                    if QMin['grad'][i] > QMin['nmstates']:
                        print(
                            'State for requested gradient does not correspond to any state in QM input file state list!')

        # get the set of states for which gradients actually need to be calculated
        gradmap = set()
        if 'grad' in QMin:
            for i in QMin['grad']:
                gradmap.add(tuple(QMin['statemap'][i][0:2]))
        gradmap = list(gradmap)
        gradmap.sort()
        QMin['gradmap'] = gradmap

        return QMin


def get_commandline():
    """
        Get Commando line option with argpase

    """

    import argparse

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
