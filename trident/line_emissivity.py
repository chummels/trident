"""
Ion fraction fields using Cloudy data.

"""

#-----------------------------------------------------------------------------
# Copyright (c) 2016, Trident Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
#-----------------------------------------------------------------------------

from yt.fields.field_detector import \
    FieldDetector
from yt.utilities.linear_interpolators import \
    TrilinearFieldInterpolator, \
    UnilinearFieldInterpolator
from yt.utilities.physical_constants import mh
from yt.funcs import mylog
import numpy as np
import h5py
import copy
import os
from trident.config import \
    ion_table_filepath, \
    emissivity_table_filepath
from trident.line_database import \
    LineDatabase, \
    uniquify
from trident.roman import \
    from_roman
from scipy.interpolate import \
    interpn

H_mass_fraction = 0.76
to_nH = H_mass_fraction / mh

# set fractions to 0 for values lower than 1e-9,
# which is what is used in Sutherland & Dopita (1993).
fraction_zero_point = 1.e-9
zero_out_value = -30.

table_store = {}

class LineEmissivityTable(object):
    def __init__(self, filename=None, line=None):
        """
        Base class for building additional ion fields

        Used to load in an HDF5 file that contains the values for the
        elemental ionization state of the gas as a function of density,
        temperature, metallcity, and redshift (for metagalactic photoionizing
        radiation field).

        **Parameters**

        :filename: string, optional

            Name of the HDF5 file that contains the ionization table data.

            Default: it uses the table specified in ~/.trident/config

        :atom: string, optional

            The atomic species for which you want to create an IonBalanceTable

            Default: None
        """
        if filename is None:
            filename = emissivity_table_filepath
        self.filename = filename
        self.parameters = []
        self.line_emissivity = []
        self._load_hdf5_table(line)

    def _load_hdf5_table(self, line):
        """
        Read in the HDF5 ion balance table
        """
        input = h5py.File(self.filename, 'r')
        self.line_emissivity = np.log10(input[line][()])
        self.parameters.append(input.attrs['density_index'])
        self.parameters.append(input.attrs['temperature_index'])
        self.parameters.append(input.attrs['metallicity_index'])
        input.close()


def _log_nH(field, data):
    """
    One index of ion balance table is in log of density, so this translates
    dataset's density values into the same format for indexing the table

    N.B. All datasets *should* have an H_nuclei_density field defined if
    created in the standard way in yt.  Ray objects will also include
    H_nuclei_density from the parent dataset to assure identical behavior
    when ions are added to a ray as when they are added to an original dataset
    before then being included in a ray.
    """
    if ("gas", "H_nuclei_density") in data.ds.derived_field_list:
        log_nH_field = np.log10(data["gas", "H_nuclei_density"])
    else:
        log_nH_field = np.log10(data["gas", "density"] * to_nH)
    return log_nH_field

def _log_T(field, data):
    """
    One index of ion balance table is in log of temperature, so this translates
    dataset's temperature values into the same format for indexing the table
    """
    return np.log10(data["gas", "temperature"])

def _log_Z(field, data):
    """
    One index of ion balance table is in log of metallicity, so this translates
    dataset's metallicity  values into the same format for indexing the table
    """
    return np.log10(data["gas", "metallicity"].to('Zsun'))

def _emission_volume(field, data):
    """
    """
    return data["gas", "line"] * data["gas", "mass"] / data["gas", "density"]


def _add_field(ds, name, function, units, sampling_type):
    """
    Private function for adding fields that wraps the yt add_field function.
    First it checks to see if field exists and if so, it does not attempt to
    add it and gives a warning message to user.  This avoids a misleading
    warning message from yt about using force_override=True, which no longer
    applies in Trident.
    """
    if name in ds.derived_field_list:
        mylog.warning("Field %s already exists. Not clobbering." % str(name))
        return
    else:
        return ds.add_field(name, function=function, units=units,
                            sampling_type=sampling_type)


def calculate_line_emissivity(line, density, temperature, metallicity, emissivity_table=None):
    """
    Calculate line emissivity at specified density, temperature, and metallicity

    **Parameters**

    :ion: string

            The desired ion following the form: "<element> <ionization_state>"
            Examples: "H I", "C II", "Mg II"

    :density: float or array of floats

            Gas density in the form of hydrogen number density cm$**{-3}$

    :temperature: float or array of floats

            Gas temperature in kelvin

    :redshift: float or array of floats

            Redshift at which to assume the metagalactic UV background

    :ionization_table: string, optional

        Path to an appropriately formatted HDF5 table that can be used to
        compute the ion fraction as a function of density, temperature,
        metallicity, and redshift.  When set to None, it uses the table
        specified in ~/.trident/config
        Default: None

    **Example**

    To calculate the ion fraction of Mg II for gas with hydrogen number
    density 1e-2 cm**-3 and redshift 0.25 and temperatures 1e4K:

    >>> import trident
    >>> trident.calculate_line_emissivity('Mg II', 1e-2, 1e4, 0.25, ')
"""
    if emissivity_table is None:
        emissivity_table = emissivity_table_filepath

    #ionn = ion.split()
    #atom = ionn[0]
    #ion_state = from_roman(ionn[1])

    field = "%s_%s_%s" % (line[0], line[1], line[2])
    if field not in table_store:
        emissivityTable = LineEmissivityTable(emissivity_table, line)
        table_store[field] = {'emissivity': copy.deepcopy(emissivityTable.line_emissivity),
                              'parameters': copy.deepcopy(emissivityTable.parameters)}
        del emissivityTable

    line = table_store[field]['emissivity']
    n_param = table_store[field]['parameters'][0]
    t_param = table_store[field]['parameters'][1]
    Z_param = table_store[field]['parameters'][2]

    # x,y,z coordinates for all ion_fractions from table
    coords = (n_param, t_param, Z_param)

    # Convert values from user into log space to match table
    dens = np.log10(density)
    temp = np.log10(temperature)
    metals = np.array(metallicity)

    if not dens.shape == temp.shape == metals.shape:
        raise RuntimeError("density, temperature, and redshift array sizes must be equal.")

    # user supplied values
    arr = (dens, temp, metals)

    # Actually interpolate the ion fraction from the user-supplied values
    line_emissivity = interpn(coords, line, arr, method='linear', bounds_error=False)
    return 10**line_emissivity


# Taken from Cloudy documentation.

solar_abundance = {
    'H' : 1.00e+00, 'He': 1.00e-01, 'Li': 2.04e-09,
    'Be': 2.63e-11, 'B' : 6.17e-10, 'C' : 2.45e-04,
    'N' : 8.51e-05, 'O' : 4.90e-04, 'F' : 3.02e-08,
    'Ne': 1.00e-04, 'Na': 2.14e-06, 'Mg': 3.47e-05,
    'Al': 2.95e-06, 'Si': 3.47e-05, 'P' : 3.20e-07,
    'S' : 1.84e-05, 'Cl': 1.91e-07, 'Ar': 2.51e-06,
    'K' : 1.32e-07, 'Ca': 2.29e-06, 'Sc': 1.48e-09,
    'Ti': 1.05e-07, 'V' : 1.00e-08, 'Cr': 4.68e-07,
    'Mn': 2.88e-07, 'Fe': 2.82e-05, 'Co': 8.32e-08,
    'Ni': 1.78e-06, 'Cu': 1.62e-08, 'Zn': 3.98e-08}

atomic_mass = {
    'H' : 1.00794,   'He': 4.002602,  'Li': 6.941,
    'Be': 9.012182,  'B' : 10.811,    'C' : 12.0107,
    'N' : 14.0067,   'O' : 15.9994,   'F' : 18.9984032,
    'Ne': 20.1797,   'Na': 22.989770, 'Mg': 24.3050,
    'Al': 26.981538, 'Si': 28.0855,   'P' : 30.973761,
    'S' : 32.065,    'Cl': 35.453,    'Ar': 39.948,
    'K' : 39.0983,   'Ca': 40.078,    'Sc': 44.955910,
    'Ti': 47.867,    'V' : 50.9415,   'Cr': 51.9961,
    'Mn': 54.938049, 'Fe': 55.845,    'Co': 58.933200,
    'Ni': 58.6934,   'Cu': 63.546,    'Zn': 65.409}

atomic_number = {
    'H' : 1,  'He': 2,  'Li': 3,
    'Be': 4,  'B' : 5,  'C' : 6,
    'N' : 7,  'O' : 8,  'F' : 9,
    'Ne': 10, 'Na': 11, 'Mg': 12,
    'Al': 13, 'Si': 14, 'P' : 15,
    'S' : 16, 'Cl': 17, 'Ar': 18,
    'K' : 19, 'Ca': 20, 'Sc': 21,
    'Ti': 22, 'V' : 23, 'Cr': 24,
    'Mn': 25, 'Fe': 26, 'Co': 27,
    'Ni': 28, 'Cu': 29, 'Zn': 30}
