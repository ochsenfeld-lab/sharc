#!/usr/bin/env python3
import re
import argparse
from io import StringIO
from typing import List, Union, Any
import numpy as np

U_TO_AMU = 1. / 5.4857990943e-4  # conversion from g/mol to electron masses
ANG_TO_BOHR = 1. / 0.529177211   # conversion from Angstrom to bohr

# Atomic Weights of the most common isotopes
# From https://chemistry.sciences.ncsu.edu/msf/pdf/IsotopicMass_NaturalAbundance.pdf
MASSES = {'H': 1.007825,
          'He': 4.002603,
          'Li': 7.016004,
          'Be': 9.012182,
          'B': 11.009305,
          'C': 12.000000,
          'N': 14.003074,
          'O': 15.994915,
          'F': 18.998403,
          'Ne': 19.992440,
          'Na': 22.989770,
          'Mg': 23.985042,
          'Al': 26.981538,
          'Si': 27.976927,
          'P': 30.973762,
          'S': 31.972071,
          'Cl': 34.968853,
          'Ar': 39.962383,
          'K': 38.963707,
          'Ca': 39.962591,
          'Sc': 44.955910,
          'Ti': 47.947947,
          'V': 50.943964,
          'Cr': 51.940512,
          'Mn': 54.938050,
          'Fe': 55.934942,
          'Co': 58.933200,
          'Ni': 57.935348,
          'Cu': 62.929601,
          'Zn': 63.929147,
          'Ga': 68.925581,
          'Ge': 73.921178,
          'As': 74.921596,
          'Se': 79.916522,
          'Br': 78.918338,
          'Kr': 83.911507,
          'Rb': 84.911789,
          'Sr': 87.905614,
          'Y': 88.905848,
          'Zr': 89.904704,
          'Nb': 92.906378,
          'Mo': 97.905408,
          'Tc': 98.907216,
          'Ru': 101.904350,
          'Rh': 102.905504,
          'Pd': 105.903483,
          'Ag': 106.905093,
          'Cd': 113.903358,
          'In': 114.903878,
          'Sn': 119.902197,
          'Sb': 120.903818,
          'Te': 129.906223,
          'I': 126.904468,
          'Xe': 131.904154,
          'Cs': 132.905447,
          'Ba': 137.905241,
          'La': 138.906348,
          'Ce': 139.905435,
          'Pr': 140.907648,
          'Nd': 141.907719,
          'Pm': 144.912744,
          'Sm': 151.919729,
          'Eu': 152.921227,
          'Gd': 157.924101,
          'Tb': 158.925343,
          'Dy': 163.929171,
          'Ho': 164.930319,
          'Er': 165.930290,
          'Tm': 168.934211,
          'Yb': 173.938858,
          'Lu': 174.940768,
          'Hf': 179.946549,
          'Ta': 180.947996,
          'W': 183.950933,
          'Re': 186.955751,
          'Os': 191.961479,
          'Ir': 192.962924,
          'Pt': 194.964774,
          'Au': 196.966552,
          'Hg': 201.970626,
          'Tl': 204.974412,
          'Pb': 207.976636,
          'Bi': 208.980383,
          'Po': 208.982416,
          'At': 209.987131,
          'Rn': 222.017570,
          'Fr': 223.019731,
          'Ra': 226.025403,
          'Ac': 227.027747,
          'Th': 232.038050,
          'Pa': 231.035879,
          'U': 238.050783,
          'Np': 237.048167,
          'Pu': 244.064198,
          'Am': 243.061373,
          'Cm': 247.070347,
          'Bk': 247.070299,
          'Cf': 251.079580,
          'Es': 252.082972,
          'Fm': 257.095099,
          'Md': 258.098425,
          'No': 259.101024,
          'Lr': 262.109692,
          'Rf': 267.,
          'Db': 268.,
          'Sg': 269.,
          'Bh': 270.,
          'Hs': 270.,
          'Mt': 278.,
          'Ds': 281.,
          'Rg': 282.,
          'Cn': 285.,
          'Nh': 286.,
          'Fl': 289.,
          'Mc': 290.,
          'Lv': 293.,
          'Ts': 294.,
          'Og': 294.
          }

NUMBERS = {'H': 1, 'He': 2,
           'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
           'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
           'K': 19, 'Ca': 20,
           'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
           'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
           'Rb': 37, 'Sr': 38,
           'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
           'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
           'Cs': 55, 'Ba': 56,
           'La': 57,
           'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68,
           'Tm': 69, 'Yb': 70, 'Lu': 71,
           'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
           'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
           'Fr': 87, 'Ra': 88,
           'Ac': 89,
           'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
           'Md': 101, 'No': 102, 'Lr': 103,
           'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
           'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
           }


def extract_timestep(geometry_filename: str, velocity_filename: str, time: str):
    rfloat = r'([+-]?\d+\.\d+)'
    geom_pattern = r'(\w?\w)' + r'\s+' + rfloat + r'\s+' + rfloat + r'\s+' + rfloat
    vel_pattern = r'(\d+):' + r'\s+' + rfloat + r'\s+' + rfloat + r'\s+' + rfloat
    geom: list[list[float]] = []
    vel: list[list[float]] = []
    elem: list[str] = []
    with open(geometry_filename, 'r') as geometries:
        natoms = geometries.readline()
        is_geom = False
        for line in geometries:
            if re.search(r'TIME:\s+' + time, line):
                is_geom = True
                continue
            if is_geom:
                if line == natoms:
                    break
                else:
                    m2 = re.search(geom_pattern, line)
                    if m2:
                        geom.append([float(m2.group(2)), float(m2.group(3)), float(m2.group(4))])
                        elem.append(m2.group(1))

    natoms = int(natoms)
    with open(velocity_filename, 'r') as velocities:
        is_geom = False
        for line in velocities:
            if re.search(r'TIME:\s+' + time, line):
                is_geom = True
                continue
            if is_geom:
                m2 = re.search(vel_pattern, line)
                if m2:
                    vel.append([float(m2.group(2)), float(m2.group(3)), float(m2.group(4))])
                    if int(m2.group(1)) == natoms:
                        break

    if len(vel) != len(geom):
        raise "Number of atoms in velocity and geometry file don't match."

    return elem, geom, vel


def print_initconds(filename, mass, elem, atomic_number, geoms, vels):
    with open(filename, "w") as f:
        # print header
        print("SHARC Initial conditions file, version 2.1\n"
              f"Ninit     {len(geoms)}\n"
              f"Natom     {len(mass)}\n"
              "Repr      None\n"
              "Eref            0.0000000000\n"
              "Eharm           0.0000000000\n\n"
              "Equilibrium",
              file=f)
        print(single_initcond_to_string(mass, elem, atomic_number, geoms[0], vels[0])+"\n", file=f)
        for i, _ in enumerate(geoms):
            print("Index" + str(i+1).rjust(7) + "\nAtoms\n"
                  + single_initcond_to_string(mass, elem, atomic_number, geoms[i], vels[i])
                  + single_initcond_add_info_to_string(mass, vels[i])+"\n\n", file=f)


def print_tasklist(filename, directories):
    with open(filename, "w") as f:
        for i, fdir in enumerate(directories):
            print(f"##### {i+1} #####", file=f)
            print(fdir, file=f)


def single_initcond_to_string(mass, elem, atomic_number, geom, vel):
    s = ""
    for i, _ in enumerate(geom):
        s += str(elem[i]).rjust(2)
        s += ("%.1f" % atomic_number[i]).rjust(6)
        for xyz in range(3):
            s += str("%.8f" % geom[i][xyz]).rjust(13)
        s += ("%.8f" % mass[i]).rjust(13)
        for xyz in range(3):
            s += ("%.8f" % vel[i][xyz]).rjust(13)
        s += "\n"
    return s


def single_initcond_add_info_to_string(mass, vel):
    ekin = 0.5 * np.sum(np.sum(np.tile(U_TO_AMU * mass, (3, 1)) * np.square(vel.T)))
    ekin_str = str(ekin).rjust(15)
    s = ("States\n"
         f"Ekin        {ekin_str} a.u.\n"
         "Epot_harm   0.000000000000 a.u.\n"
         "Epot        0.000000000000 a.u.\n"
         f"Etot_harm   {ekin_str} a.u.\n"
         f"Etot        {ekin_str} a.u.")
    return s


def print_keystrokes(filename, args):
    arguments_str = " "
    for key, value in args.__dict__.items():
        arguments_str += "--" + key + " "
        if np.isscalar(value):
            arguments_str += str(value) + " "
        else:
            arguments_str += " ".join([str(i) for i in value]) + " "
    with open(filename, "w") as f:
        print("python3 " + __file__ + arguments_str, file=f)


def main():
    """
    Convert a set of Fermions Trajectories to SHARC initial conditions file
    """

    parser = argparse.ArgumentParser(description='Convert a set of Fermions Trajectories to SHARC initial conditions '
                                                 'file')

    parser.add_argument('--geometry_filename', type=str,
                        help='fermions::aimd_print::geometry. Default: traj_geom',
                        default='traj_geom')
    parser.add_argument('--velocity_filename', type=str,
                        help='fermions::aimd_print::velocity. Default: traj_vel',
                        default='traj_vel')
    parser.add_argument('--time', type=str,
                        help='timestep to extract. Default: 5000.0',
                        default='5000.0')
    parser.add_argument('--md_subdir_name', type=str,
                        help='names of the subdirectories with the groundstate MD trajectories. Default: frame_',
                        default='frame_')
    parser.add_argument('--md_subdir_range', nargs=2, type=int,
                        help='frames for which to extract a geometry. Default: 0 1',
                        default=['0'])
    parser.add_argument('--groundstate_md_directory', type=str,
                        help='directory with subdirectories for the groundstate MD trajectories. Default: '
                             '/home/mpeschel/Enones/solvent_eq/conf1',
                        default='/home/mpeschel/Enones/solvent_eq/conf1')
    args = parser.parse_args()
    print_keystrokes("KEYSTROKES.fermions_to_initconds", args)

    geoms = []
    vels = []
    failed = []
    for i in range(args.md_subdir_range[0], args.md_subdir_range[1]):
        directory = args.groundstate_md_directory + "/" + args.md_subdir_name + str(i) + "/"
        print(directory)
        elem, geom, vel = extract_timestep(directory + args.geometry_filename, directory + args.velocity_filename, args.time)
        if not geoms or (np.shape(np.array(vel)) == np.shape(vels[-1]) and np.shape(np.array(geom)) == np.shape(geoms[-1])):
            vels.append(np.array(vel) / np.sqrt(U_TO_AMU))  # this conversion is necessary because of the units in fermions
            geoms.append(np.array(geom) * ANG_TO_BOHR)
        else:
            failed.append(directory)
            print(f"PROBLEM in {directory}. Trajectory does not match size of previous trajectories.")

        # THIS PART IS FOR DEBUGGING UNITS, VELOCITY UNITS IN FERMIONS ARE SH*T
        # someone forgot to convert the masses to au before printing the velocities
        # calculate temperature to check velocity conversion factor
        # mass = mass * U_TO_AMU
        # ekin = 0.5 * np.sum(np.sum(np.tile(mass, (3, 1)) * np.square(vel.T)))
        # ekin2 = np.sum(np.sum((np.tile(mass, (3, 1)) * vel.T) ** 2 / (2 * np.tile(mass, (3, 1)))))
        # print(mass)
        # print(vel)
        # print(ekin)
        # print(ekin2)
        # KB = 3.167e-6  # Boltzmann constant in Hartree/K
        # print(2 * ekin / (3 * len(mass) * KB))
        # raise ("Don't go further when debugging units")

    elem = [i.lower().capitalize() for i in elem]
    mass = np.array([MASSES[i] for i in elem])
    atomic_number = np.array([NUMBERS[i] for i in elem])
    print_initconds("initconds", mass, elem, atomic_number, geoms, vels)
    print_tasklist("failed.tasklist", failed)


if __name__ == "__main__":
    main()
