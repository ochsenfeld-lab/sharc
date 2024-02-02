#!/usr/bin/env python3
import sys
import re
import numpy as np
import os

au2a = 0.529177211

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

IToMult2 = {
    1: 'Singlets',
    2: 'Doublets',
    3: 'Triplets',
    4: 'Quartets',
    5: 'Quintets',
    6: 'Sextets',
    7: 'Septets',
    8: 'Octets',
    'Singlets': 1,
    'Doublets': 2,
    'Triplets': 3,
    'Quartets': 4,
    'Quintets': 5,
    'Sextets': 6,
    'Septets': 7,
    'Octets': 8
}

IToMult3 = {
    1: 'S',
    2: 'D',
    3: 'T',
    4: 'Q',
    'S': 1,
    'D': 2,
    'T': 3,
    'Q': 4,
}

PRINT = True


def run(command):
    return os.popen(command).read().rstrip()


def readf(myfile):
    with open(myfile, 'r') as file:
        data = file.read()
    return data


def iterable_to_converted_array(iterable, converter=float):
    myarray = []
    for group in iterable:
        myarray.append(converter(group))
    return myarray


def key_from_value(mydict, value):
    return list(mydict.keys())[list(mydict.values()).index(value)]


def ml_from_n(n):
    return np.arange(-(n - 1) / 2, (n - 1) / 2 + 1, 1)


def find_info(ferm_out, patterns):
    nfound_begin = dict.fromkeys(patterns, 0)
    nfound_end = dict.fromkeys(patterns, 0)
    res = {k: [] for k in patterns.keys()}  # empty list for each key
    for i, line in enumerate(ferm_out):
        for key, ml_pattern in patterns.items():
            if nfound_begin[key] < len(ml_pattern['begin']):
                if re.search(ml_pattern['begin'][nfound_begin[key]], line):
                    nfound_begin[key] += 1
            if nfound_begin[key] == len(
                    ml_pattern['begin']):  # <<< no else here because 'begin' and 'match' can be in the same line
                if re.search(ml_pattern['end'][nfound_end[key]], line):
                    nfound_end[key] += 1
                if nfound_end[key] == len(ml_pattern['end']):
                    res[key] = np.squeeze(np.array(res[key]))
                    nfound_begin[key] += 1  # <<< if we have reached the last end we are done with this pattern
                else:
                    m = re.search(ml_pattern['match'], line)
                    if m:
                        res[key].append(iterable_to_converted_array(m.groups(), converter=ml_pattern['convert']))
    tmp = {k: v for k, v in res.items() if
           np.size(v) != 0}  # return dict with only the keys where we have found a result, i.e. delete empty lists
    return tmp


def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % filename)
        sys.exit(12)
    return out


def containsstring(string, line):
    '''Takes a string (regular expression) and another string. Returns True if the first string is contained in the
    second string.

    Arguments:
    1 string: Look for this string
    2 string: within this string

    Returns:
    1 boolean'''

    a = re.search(string, line)
    if a:
        return True
    else:
        return False


def get_pairs(QMinlines, i):
    nacpairs = []
    while True:
        i += 1
        try:
            line = QMinlines[i].lower()
        except IndexError:
            print('"keyword select" has to be completed with an "end" on another line!')
            sys.exit(39)
        if 'end' in line:
            break
        fields = line.split()
        try:
            nacpairs.append([int(fields[0]), int(fields[1])])
        except ValueError:
            print('"nacdr select" is followed by pairs of state indices, each pair on a new line!')
            sys.exit(40)
    return nacpairs, i


def writefile(filename, content):
    # content can be either a string or a list of strings
    try:
        f = open(filename, 'w')
        if isinstance(content, list):
            for line in content:
                f.write(line)
        elif isinstance(content, str):
            f.write(content)
        else:
            print('Content %s cannot be written to file!' % content)
        f.close()
    except IOError:
        print('Could not write to file %s!' % (filename))
        sys.exit(13)


def eformat(f, prec, exp_digits):
    '''Formats a float f into scientific notation with prec number of decimals and exp_digits number of exponent digits.

  String looks like:
  [ -][0-9]\.[0-9]*E[+-][0-9]*

  Arguments:
  1 float: Number to format
  2 integer: Number of decimals
  3 integer: Number of exponent digits

  Returns:
  1 string: formatted number'''

    s = "% .*e" % (prec, f)
    mantissa, exp = s.split('e')
    return "%sE%+0*d" % (mantissa, exp_digits + 1, int(exp))


def get_res(res, key, index, default='ThrowError: Value not found'):
    '''
  Looks if we have the result and returns it if it's there
  otherwise returns default value

  also takes care of (anti-)hermiticity; i.e. if we have (0,1) of an
  (anti-)hermitian matrix, we also have (1,0)
  '''

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


def format_res(res, key, index, default='Error: Value not found', iscomplex=True):
    x = get_res(res, key, index, default)
    if iscomplex:
        return '%s %s ' % (eformat(np.real(x), 12, 3), eformat(np.imag(x), 12, 3))
    else:
        return '%s ' % (eformat(np.real(x), 12, 3))


def getsmate(out):
    print(out)
    ilines = -1
    while True:
        ilines += 1
        if ilines == len(out):
            print('Overlap of states %i - %i not found!' % (s1, s2))
            sys.exit(103)
        if containsstring('Overlap matrix <PsiA_i|PsiB_j>', out[ilines]):
            break
    ilines += 1 + s1
    f = out[ilines].split()
    return float(f[s2 + 1])


def removekey(d,key):
    '''Removes an entry from a dictionary and returns the dictionary.

    Arguments:
    1 dictionary
    2 anything which can be a dictionary keyword

    Returns:
    1 dictionary'''

    if key in d:
        r = dict(d)
        del r[key]
        return r
    return d

# =============================================================================================== #
# =============================================================================================== #
# ============================= iterator routines  ============================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def itmult(states):
    for i in range(len(states)):
        if states[i] < 1:
            continue
        yield i + 1
    return


# ======================================================================= #
def itnmstates(states):
    for i in range(len(states)):
        if states[i] < 1:
            continue
        for k in range(i + 1):
            for j in range(states[i]):
                yield i + 1, j + 1, k - i / 2.
    return


# =============================================================================================== #
# =============================================================================================== #
# ============================= readQMin  ======================================================= #
# =============================================================================================== #
# =============================================================================================== #

def readQMin(QMinfilename):
    '''Reads the time-step dependent information from QMinfilename. This file contains all information from the
    current SHARC job: geometry, velocity, number of states, requested quantities along with additional information.
    The routine also checks this input and obtains a number of environment variables necessary to run MOLPRO.

  Steps are:
  - open and read QMinfilename
  - Obtain natom, comment, geometry (, velocity)
  - parse remaining keywords from QMinfile
  - check keywords for consistency, calculate nstates, nmstates
  - obtain environment variables for path to MOLPRO and scratch directory, and for error handling

  Arguments:
  1 string: name of the QMin file

  Returns:
  1 dictionary: QMin'''

    # read QMinfile
    QMinlines = readfile(QMinfilename)
    QMin = {}

    # Get natom
    try:
        natom = int(QMinlines[0].split()[0])
    except ValueError:
        print('first line must contain the number of atoms!')
        sys.exit(41)
    QMin['natom'] = natom
    if len(QMinlines) < natom + 4:
        print('Input file must contain at least:\nnatom\ncomment\ngeometry\nkeyword "states"\nat least one task')
        sys.exit(42)

    # Save Comment line
    QMin['comment'] = QMinlines[1]

    # Get geometry and possibly velocity
    QMin['geo'] = []
    QMin['veloc'] = []
    hasveloc = True
    for i in range(2, natom + 2):
        if not containsstring('[a-zA-Z][a-zA-Z]?[0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*.*[-]?[0-9]+[.][0-9]*',
                              QMinlines[i]):
            print('Input file does not comply to xyz file format! Maybe natom is just wrong.')
            sys.exit(43)
        fields = QMinlines[i].split()
        for j in range(1, 4):
            fields[j] = float(fields[j])
        QMin['geo'].append(fields[0:4])
        if len(fields) >= 7:
            for j in range(4, 7):
                fields[j] = float(fields[j])
            QMin['veloc'].append(fields[4:7])
        else:
            hasveloc = False
    if not hasveloc:
        QMin = removekey(QMin, 'veloc')

    # Parse remaining file
    i = natom + 1
    while i + 1 < len(QMinlines):
        i += 1
        line = QMinlines[i]
        line = re.sub('#.*$', '', line)
        if len(line.split()) == 0:
            continue
        key = line.lower().split()[0]
        if 'savedir' in key:
            args = line.split()[1:]
        else:
            args = line.lower().split()[1:]
        if key in QMin:
            print('Repeated keyword %s in line %i in input file! Check your input!' % (linekeyword(line), i + 1))
            continue  # only first instance of key in QM.in takes effect
        if len(args) >= 1 and 'select' in args[0]:
            pairs, i = get_pairs(QMinlines, i)
            QMin[key] = pairs
        else:
            QMin[key] = args

    # Units conversion
    if 'unit' in QMin:
        if QMin['unit'][0] == 'angstrom':
            factor = 1. / au2a
        elif QMin['unit'][0] == 'bohr':
            factor = 1.
        else:
            print('Dont know input unit %s!' % (QMin['unit'][0]))
            sys.exit(44)
    else:
        factor = 1. / au2a
    for iatom in range(len(QMin['geo'])):
        for ixyz in range(3):
            QMin['geo'][iatom][ixyz + 1] *= factor

    # State number
    if not 'states' in QMin:
        print('Keyword "states" not given!')
        sys.exit(45)
    for i in range(len(QMin['states'])):
        QMin['states'][i] = int(QMin['states'][i])
    reduc = 0
    for i in reversed(QMin['states']):
        if i == 0:
            reduc += 1
        else:
            break
    for i in range(reduc):
        del QMin['states'][-1]
    nstates = 0
    nmstates = 0
    for i in range(len(QMin['states'])):
        nstates += QMin['states'][i]
        nmstates += QMin['states'][i] * (i + 1)
    QMin['nstates'] = nstates
    QMin['nmstates'] = nmstates

    # Various logical checks
    possibletasks = ['h', 'soc', 'dm', 'grad', 'nacdr', 'overlap', 'ion', 'molden', 'phases']
    if not any([i in QMin for i in possibletasks]):
        print('No tasks found! Tasks are "h", "soc", "dm", "grad", "nacdr", "overlap", "ion", and "molden".')
        sys.exit(46)

    if 'samestep' in QMin and 'init' in QMin:
        print('"Init" and "Samestep" cannot be both present in QM.in!')
        sys.exit(47)

    if 'phases' in QMin:
        QMin['overlap'] = []

    if 'overlap' in QMin and 'init' in QMin:
        print('"overlap" and "phases" cannot be calculated in the first timestep! Delete either "overlap" or "init"')
        sys.exit(48)

    if not 'init' in QMin and not 'samestep' in QMin and not 'restart' in QMin:
        QMin['newstep'] = []

    if len(QMin['states']) > 8:
        print('Higher multiplicities than octets are not supported!')
        sys.exit(49)

    if 'h' in QMin and 'soc' in QMin:
        QMin = removekey(QMin, 'h')

    if 'nacdt' in QMin:
        print('Within the SHARC-XXX interface, couplings can only be calculated via "nacdr" or "overlap".')
        sys.exit(50)

    if 'dmdr' in QMin:
        print('Dipole moment gradients not available!')
        sys.exit(51)

    if 'socdr' in QMin:
        print('Spin-orbit gradients not available!')
        sys.exit(52)

    if 'nacdr' in QMin:
        QMin['docicas'] = True

    # Process the gradient requests
    if 'grad' in QMin:
        if len(QMin['grad']) == 0 or QMin['grad'][0] == 'all':
            QMin['grad'] = [i + 1 for i in range(nmstates)]
        else:
            for i in range(len(QMin['grad'])):
                try:
                    QMin['grad'][i] = int(QMin['grad'][i])
                except ValueError:
                    print('Arguments to keyword "grad" must be "all" or a list of integers!')
                    sys.exit(53)
                if QMin['grad'][i] > nmstates:
                    print('State for requested gradient does not correspond to any state in QM input file state list!')
                    sys.exit(54)

    # Process the non-adiabatic coupling requests
    # type conversion has already been done
    if 'nacdr' in QMin:
        if len(QMin['nacdr']) >= 1:
            nacpairs = QMin['nacdr']
            for i in range(len(nacpairs)):
                if nacpairs[i][0] > nmstates or nacpairs[i][1] > nmstates:
                    print(
                        'State for requested non-adiabatic couplings does not correspond to any state in QM input file state list!')
                    sys.exit(55)
        else:
            QMin['nacdr'] = [[j + 1, i + 1] for i in range(nmstates) for j in range(i)]

    # obtain the statemap
    statemap = {}
    i = 1
    for imult, istate, ims in itnmstates(QMin['states']):
        statemap[i] = [imult, istate, ims]
        i += 1
    QMin['statemap'] = statemap
    QMin['maxmult'] = max([i[0] for i in QMin['statemap'].values()])

    # get the set of states for which gradients actually need to be calculated
    gradmap = set()
    if 'grad' in QMin:
        for i in QMin['grad']:
            gradmap.add(tuple(statemap[i][0:2]))
    gradmap = list(gradmap)
    gradmap.sort()
    QMin['gradmap'] = gradmap

    # get the list of statepairs for NACdr calculation
    nacmap = set()
    if 'nacdr' in QMin:
        for i in QMin['nacdr']:
            s1 = statemap[i[0]][0:2]
            s2 = statemap[i[1]][0:2]
            if s1[0] != s2[0] or s1 == s2:
                continue
            if s1[1] > s2[1]:
                continue
            nacmap.add(tuple(s1 + s2))
    nacmap = list(nacmap)
    nacmap.sort()
    QMin['nacmap'] = nacmap

    if PRINT:
        print("\n===> Successfully read QM.in\n")
        print(QMin)

    return QMin


# =============================================================================================== #
# =============================================================================================== #
# =========================================== QMout writing ===================================== #
# =============================================================================================== #
# =============================================================================================== #

# ======================================================================= #
def writeQMout(QMin, QMout, QMinfilename):
    '''Writes the requested quantities to the file which SHARC reads in. The filename is QMinfilename with everything after the first dot replaced by "out".

  Arguments:
  1 dictionary: QMin
  2 dictionary: QMout
  3 string: QMinfilename'''

    k = QMinfilename.find('.')
    if k == -1:
        outfilename = QMinfilename + '.out'
    else:
        outfilename = QMinfilename[:k] + '.out'
    if PRINT:
        print('\n===> Writing output to file %s in SHARC Format\n' % outfilename)
    string = ''
    if 'h' in QMin or 'soc' in QMin:
        string += writeQMoutsoc(QMin, QMout)
    if 'dm' in QMin:
        string += writeQMoutdm(QMin, QMout)
    if 'grad' in QMin:
        string += writeQMoutgrad(QMin, QMout)
    if 'nacdr' in QMin:
        string += writeQMoutnacana(QMin, QMout)
    if 'overlap' in QMin:
        string += writeQMoutnacsmat(QMin, QMout)
    if 'ion' in QMin:
        string += writeQMoutprop(QMin, QMout)
    if 'phases' in QMin:
        string += writeQMoutPhases(QMin, QMout)
    string += writeQMouttime(QMin, QMout)
    writefile(outfilename, string)
    if PRINT:
        print(string)
    return


# ======================================================================= #
def writeQMoutsoc(QMin, QMout):
    '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

  The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

  Arguments:
  1 dictionary: QMin
  2 dictionary: QMout

  Returns:
  1 string: multiline string with the SOC matrix'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Hamiltonian Matrix (%ix%i, complex)\n' % (1, nmstates, nmstates)
    string += '%i %i\n' % (nmstates, nmstates)
    for i in range(1, nmstates + 1):
        for j in range(1, nmstates + 1):
            if i == j:
                string += format_res(QMout, 'energy', [i])
            else:
                string += format_res(QMout, 'soc', [i, j], default=0)
        string += '\n'
    string += '\n'
    return string


# ======================================================================= #
def writeQMoutdm(QMin, QMout):
    '''Generates a string with the Dipole moment matrices in SHARC format.

  The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the
  matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line. The
  string contains three such matrices.

  Arguments:
  1 dictionary: QMin
  2 dictionary: QMout

  Returns:
  1 string: multiline string with the DM matrices'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Dipole Moment Matrices (3x%ix%i, complex)\n' % (2, nmstates, nmstates)
    for xyz in range(3):
        string += '%i %i\n' % (nmstates, nmstates)
        for i in range(1, nmstates + 1):
            for j in range(1, nmstates + 1):
                string += format_res(QMout, 'dm', [i, j, xyz], default=0)
            string += '\n'
        string += ''
    return string


# ======================================================================= #
def writeQMoutgrad(QMin, QMout):
    '''Generates a string with the Gradient vectors in SHARC format.

  The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are
  written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (
  nmstates gradients are written).

  Arguments:
  1 dictionary: QMin
  2 dictionary: QMout

  Returns:
  1 string: multiline string with the Gradient vectors'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Gradient Vectors (%ix%ix3, real)\n' % (3, nmstates, natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        string += '%i %i ! %i %i %i\n' % (natom, 3, imult, istate, ims)
        for atom in range(natom):
            for xyz in range(3):
                string += format_res(QMout, 'gradient',
                                     [key_from_value(QMin['statemap'], [imult, istate, ims]), atom, xyz], default=0,
                                     iscomplex=False)
            string += '\n'
        string += ''
        i += 1
    return string


# ======================================================================= #
def writeQMoutnacana(QMin, QMout):
    '''Generates a string with the NAC vectors in SHARC format.

  The string starts with a ! followed by a flag specifying the type of data. On the next line, natom and 3 are
  written, followed by the gradient, with one line per atom and a blank line at the end. Each MS component shows up (
  nmstates x nmstates vectors are written).

  Arguments:
  1 dictionary: QMin
  2 dictionary: QMout

  Returns:
  1 string: multiline string with the NAC vectors'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Non-adiabatic couplings (ddr) (%ix%ix%ix3, real)\n' % (5, nmstates, nmstates, natom)
    i = 0
    for imult, istate, ims in itnmstates(states):
        j = 0
        for jmult, jstate, jms in itnmstates(states):
            string += '%i %i ! %i %i %i %i %i %i\n' % (natom, 3, imult, istate, ims, jmult, jstate, jms)
            for atom in range(natom):
                for xyz in range(3):
                    string += format_res(QMout, 'nacv', [key_from_value(QMin['statemap'], [imult, istate, ims]),
                                                         key_from_value(QMin['statemap'], [jmult, jstate, jms]), atom,
                                                         xyz], default=0, iscomplex=False)
                string += '\n'
            string += ''
            j += 1
        i += 1
    return string


# ======================================================================= #
def writeQMoutnacsmat(QMin, QMout):
    '''Generates a string with the adiabatic-diabatic transformation matrix in SHARC format.

  The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the
  matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

  Arguments:
  1 dictionary: QMin
  2 dictionary: QMout

  Returns:
  1 string: multiline string with the transformation matrix'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']

    string = ''
    string += '! %i Overlap matrix (%ix%i, complex)\n' % (6, nmstates, nmstates)
    string += '%i %i\n' % (nmstates, nmstates)
    for j in range(nmstates):
        for i in range(nmstates):
            string += '%s %s ' % (
                eformat(QMout['overlap'][j][i].real, 12, 3), eformat(QMout['overlap'][j][i].imag, 12, 3))
        string += '\n'
    string += '\n'
    return string


# ======================================================================= #
def writeQMouttime(QMin, QMout):
    '''Generates a string with the quantum mechanics total runtime in SHARC format.

  The string starts with a ! followed by a flag specifying the type of data. In the next line, the runtime is given

  Arguments:
  1 dictionary: QMin
  2 dictionary: QMout

  Returns:
  1 string: multiline string with the runtime'''

    string = '! 8 Runtime\n%s\n' % (eformat(QMout['runtime'], 12, 3))
    return string


# ======================================================================= #
def writeQMoutprop(QMin, QMout):
    '''Generates a string with the Spin-Orbit Hamiltonian in SHARC format.

  The string starts with a ! followed by a flag specifying the type of data. In the next line, the dimensions of the
  matrix are given, followed by nmstates blocks of nmstates elements. Blocks are separated by a blank line.

  Arguments:
  1 dictionary: QMin
  2 dictionary: QMout

  Returns:
  1 string: multiline string with the SOC matrix'''

    states = QMin['states']
    nstates = QMin['nstates']
    nmstates = QMin['nmstates']
    natom = QMin['natom']
    string = ''
    string += '! %i Property Matrix (%ix%i, complex)\n' % (11, nmstates, nmstates)
    string += '%i %i\n' % (nmstates, nmstates)
    for i in range(nmstates):
        for j in range(nmstates):
            string += '%s %s ' % (eformat(QMout['prop'][i][j].real, 12, 3), eformat(QMout['prop'][i][j].imag, 12, 3))
        string += '\n'
    string += '\n'
    return string


# ======================================================================= #
def writeQMoutPhases(QMin, QMout):
    string = '! 7 Phases\n%i ! for all nmstates\n' % (QMin['nmstates'])
    for i in range(QMin['nmstates']):
        string += '%s %s\n' % (eformat(QMout['phases'][i].real, 9, 3), eformat(QMout['phases'][i].imag, 9, 3))
    return string
