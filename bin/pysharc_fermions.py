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

import shutil
import sys
import os
from overrides import EnforceOverrides, final, override
import time

import numpy as np

from sharc.pysharc.interface import SHARC_INTERFACE

from Fermions_wfoverlap import CisNto, setup


class SHARC_FERMIONS(SHARC_INTERFACE):
    """
    Class for SHARC LVC
    """
    # Name of the interface
    interface = 'LVC'
    # store atom ids
    save_atids = True
    # store atom names
    save_atnames = False
    # accepted units:  0 : Bohr, 1 : Angstrom
    iunit = 0
    # TODO: where can i find a list of all possibilities??
    not_supported = ['nacdt', 'dmdr']

    @override
    def __init__(self, *args, **kwargs):
        """
        Init your interface, best is you

        set parameter files etc. for read

        """

        # Set up the fermions instance and put it in storage
        self.storage['method'] = 'tda'
        self.storage['geo_step'] = {}
        self.storage['geo_step'][0] = kwargs['geo']
        self.storage['Fermions'], self.storage['tdscf_options'], self.storage['tdscf_deriv_options'] = setup(
            kwargs['geo'])

    def do_qm_job(self, tasks, Crd):
        """

        Here you should perform all your qm calculations

        depending on the tasks, that were asked

        """
        t = time.time()

        QMin = self.parseTasks(tasks)
        step = int(QMin['step'][0])

        # TODO: Update the geometry
        print(Crd)
        self.storage['geo_step'][0] = Crd

        QMout = dict()

        Fermions = self.storage['Fermions']
        tdscf_options = self.storage['tdscf_options']
        tdscf_deriv_options = self.storage['tdscf_deriv_options']

        # ground state #TODO: only singlet ground state works like this
        # QMout[(1, 'energy')], grad_gs = calc_groundstate(Fermions, 'grad' not in QMin)
        # Always calculate groundstate gradient since its cheap and needed for other stuff
        QMout[(1, 'energy')], QMout[(1, 'gradient')] = self.calc_groundstate(Fermions, False)
        QMout[(1, 1, 'dm')] = np.array(Fermions.calc_dipole_MD())

        # excited states
        if QMin['nmstates'] > 1:
            exc_state = Fermions.get_excited_states(tdscf_options)
            exc_state.evaluate()

            # save dets for wfoverlap (TODO: probably does not work for triplets yet, check!)
            tda_amplitudes = []
            for state in range(2, QMin['nmstates'] + 1):
                mult = sharc.IToMult[QMin['statemap'][state][0]]
                index = QMin['statemap'][state][1] - 1
                tda_amplitude, _ = Fermions.load_td_amplitudes(td_method=method, td_spin=mult, td_state=index)
                tda_amplitudes.append(tda_amplitude)

            # get excitation energies
            # TODO: implement for non singlet ground states
            mult = sharc.IToMult[QMin['statemap'][2][0]]
            exc_energies = exc_state.get_exc_energies(method=method, st=mult)
            for state in range(2, QMin['nmstates'] + 1):
                index = QMin['statemap'][state][1] - 2
                QMout[(state, 'energy')] = QMout[(1, 'energy')] + exc_energies[index]

            # calculate gradients
            for state in QMin['gradmap']:
                if state == (1, 1):
                    continue
                mult = sharc.IToMult[state[0]]
                index = state[1] - 1
                forces_ex = exc_state.tdscf_forces_nacs(do_grad=True, nacv_flag=False, method=method,
                                                        spin=mult, trg_state=index,
                                                        py_string=tdscf_deriv_options)
                # if we do qmmm we need to read a different set of forces
                if Fermions.qmmm:
                    forces_ex = Fermions.globals.get_FILES().read_double_sub(len(Fermions.mol) * 3, 0,
                                                                             'qmmm_exc_forces', 0)
                for ml in sharc.ml_from_n(state[0]):
                    snr = sharc.key_from_value(QMin['statemap'], [state[0], state[1], ml])
                    QMout[(snr, 'gradient')] = \
                        np.array(forces_ex).reshape(len(Fermions.mol), 3)
                    # we only get state dipoles for the states where we calc gradients
                    QMout[(snr, snr, 'dm')] = \
                        np.array(exc_state.state_mm(index - 1, 1)[1:])

            if 'nacdr' in QMin:
                print("nacdr not yet implemented.")
                sys.exit()

            if 'soc' in QMin:
                print("soc not yet implemented.")
                sys.exit()

            # Currently only implemented for singlet, TODO: other multiplicities
            if 'dm' in QMin:
                n = QMin['nmstates']
                dm_mn = np.concatenate([exc_state.get_transition_dipoles_0n(method=method),
                                        np.array(exc_state.get_transition_dipoles_mn(method=method, st=1))]) \
                        * 0.393456  # Convert Debye to au
                dm = [np.zeros([n, n]), np.zeros([n, n]), np.zeros([n, n])]
                nelem_upper = int(0.5 * (n * (n - 1)) * 3)  # we have to do this since fermions add states sometimes
                dm[0][np.triu_indices(n, 1)] = dm_mn[0:nelem_upper:3]
                dm[1][np.triu_indices(n, 1)] = dm_mn[1:nelem_upper:3]
                dm[2][np.triu_indices(n, 1)] = dm_mn[2:nelem_upper:3]
                for i in range(1, QMin['nmstates'] + 1):
                    for j in range(1, QMin['nmstates'] + 1):
                        state1 = QMin['statemap'][i]
                        state2 = QMin['statemap'][j]
                        istate1 = state1[1] - 1
                        istate2 = state2[1] - 1
                        if (istate1 < istate2) and (state1[0] == state2[0]) and (state1[2] == state2[2]):
                            QMout[(i, j, 'dm')] = np.array(
                                [dm[0][i - 1][j - 1], dm[1][i - 1][j - 1], dm[2][i - 1][j - 1]])

            # in the first step we still need to save mos and dets for wfoverlap
            if 'step' in QMin:
                if int(QMin['step'][0]) == 0:
                    _ = self.run_cisnto(Fermions, exc_energies, tda_amplitudes, QMin['geo'], QMin['geo'], 0, 0)

            # if 'overlap' in QMin:
            #     QMout['overlap'] = self.run_cisnto(Fermions, exc_energies, tda_amplitudes, self.storage['geo_step'][step-1], self.storage['geo_step'][step], step-1, step))
            #
            # # Phases from overlaps
            # if 'phases' in QMin:
            #     if 'phases' not in QMout:
            #         QMout['phases'] = [complex(1., 0.) for _ in range(QMin['nmstates'])]
            #     if 'overlap' in QMout:
            #         for i in range(QMin['nmstates']):
            #             if QMout['overlap'][i][i].real < 0.:
            #                 QMout['phases'][i] = complex(-1., 0.)

        QMout['runtime'] = time.time() - t  # rough estimation of runtime

        return QMout

    def parseTasks(self, tasks):
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
        if 'init' not in QMin and 'samestep' not in QMin and 'restart' not in QMin:
            fromfile = os.path.join(QMin['savedir'], 'U.out')
            if not os.path.isfile(fromfile):
                print('ERROR: savedir does not contain U.out! Maybe you need to add "init" to QM.in.')
                sys.exit(1)
            tofile = os.path.join(QMin['savedir'], 'Uold.out')
            shutil.copy(fromfile, tofile)

        for key in ['grad', 'nacdr']:
            if tasks[key].strip() != "":
                QMin[key] = []

        QMin['pwd'] = os.getcwd()
        return QMin

    def run_cisnto(self, fermions, exc_energies, tda_amplitudes, geo_old, geo, step_old: int, step: int) -> CisNto:
        # if we do qmmm we need to only give the qm region to calc the overlap
        if fermions.qmmm:
            # TODO: this does not work for non-continuous QM regions or definitions via residues
            m = re.search(r'qm\s*=\s*\{a(\d+)\s*-\s*(\d+)}', fermions.qmmm_sys, re.IGNORECASE)
            if not m:
                sys.exit("Sorry, Could not read QM-System Definition, Definition either wrong, "
                         "or is more complicated than i implemented in SHARC_FERMIONS...")
            qm_slice = slice(int(m.group(1)) - 1, int(m.group(2)))
            program = CisNto("$CIS_NTO/cis_overlap.exe", geo_old[qm_slice], geo[qm_slice], step_old, step,
                             basis="basis")
        else:
            program = CisNto("$CIS_NTO/cis_overlap.exe", geo_old, geo, step_old, step, basis="basis")
        program.save_mo(fermions.load("mo"), step)
        program.save_dets(tda_amplitudes, step, exc_energies)
        return program.get_overlap(step_old, step)

    def calc_groundstate(self, fermions, energy_only):
        energy_gs, forces_gs = fermions.calc_energy_forces_MD(mute=0, timeit=False, only_energy=energy_only)
        if energy_only:
            return np.array(energy_gs), None
        else:
            return np.array(energy_gs), np.array(forces_gs).reshape(len(fermions.mol), 3)


def getCommandoLine():
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

    inp_file, param = getCommandoLine()
    # init SHARC_LVC class
    lvc = SHARC_LVC()
    # run sharc dynamics
    lvc.run_sharc(inp_file, param)


if __name__ == "__main__":
    main()
