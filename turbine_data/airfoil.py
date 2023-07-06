# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, attribute-defined-outside-init

"""
Airfoil utilities
-----------------

"""

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.interpolate as sint
import yaml

from . import cst

class AirfoilTableDB:
    """Airfoil Polar lookup table database for multiple airfoils

    Load multiple airfoil tables from a yaml file and use AirfoilTable
    to provide an interpolation utility to lookup polars for a given
    range of angles of attack.
    """
    def __init__(self, yaml_file):
        self.af_data = yaml.load(open(yaml_file),Loader=yaml.BaseLoader)

    def get_airfoils(self):
        """Get list of available airfoils"""
        return self.af_data.keys()

    def get_airfoil_data(self, airfoil):
        """Get data for given airfoil"""
        return pd.DataFrame(self.af_data[airfoil]).astype(float)

    def get_aftable(self, airfoil):
        """Return AirfoilTable object for a given airfoil"""
        if (airfoil in self.af_data.keys()):
            aftb = AirfoilTable(
                aoa = np.array(self.af_data[airfoil]['aoa'],dtype=float),
                cl = np.array(self.af_data[airfoil]['cl'],dtype=float),
                cd = np.array(self.af_data[airfoil]['cd'],dtype=float),
                cm = np.array(self.af_data[airfoil]['cm'],dtype=float))
            return aftb
        else:
            return None

    def stall_angle(self, airfoil):
        """Placeholder for default stall angle

        Args:
            airfoil (string): Airfoil name
        """

        return self.lift_stall_angle(airfoil)

    def lift_stall_angle(self, airfoil):
        """Return stall angle for airfoil

        Args:
            airfoil (string): Airfoil name
        """
        aftb = self.get_aftable(airfoil)
        return aftb.lift_stall_angle()

    def moment_stall_angle(self, airfoil):
        """"Return moment stall angle for airfoil

        Args:
            airfoil (string): Airfoil name
        """
        aftb = self.get_aftable(airfoil)
        return aftb.moment_stall_angle()

    def __call__(self, airfoil, alpha, key):
        """Return interpolated values for a given airfoil and key at angles of attack

        Example:

        .. code-block:: python

           # Load the airfoil data
           aftable = AirfoilTableDB(yaml_file)
           # Show available columns
           print(aftable.columns)
           cl, cd, cm = aftable('NACA64_A17',aoa_new, ['cl', 'cd', 'cm'])

        Args:
            airfoil (string): Airfoil name
            alpha (np.ndarray): Angles of attack where value is desired
            key: string or list
        """
        aftb = self.get_aftable(airfoil)
        return aftb(alpha, key)

    def to_csv(self, filename):

        af_data_csv = pd.DataFrame(columns=['aoa','cl','cd','label'])
        for af in self.get_airfoils():
            af_data = self.get_airfoil_data(af)
            af_data['stall_margin'] = self.stall_angle(af) - af_data['aoa']
            af_data['label'] = af
            af_data_csv = af_data_csv.append(af_data,ignore_index=True)

        af_data_csv.to_csv(filename)


class AirfoilTable:
    """Airfoil lookup table

    Load an airfoil polar table from file and provide an interpolation utility
    to lookup polars for a given range of angles of attack.
    """

    @classmethod
    def read_csv(cls, csv_file):
        """Load lookup table from a CSV file"""
        obj = cls.__new__(cls)
        obj.data = pd.read_csv(csv_file)
        return obj

    def __init__(self, aoa, **kwargs):
        """
        Args:
            aoa (np.ndarray): Array of angles of attack
        """
        dmap = dict(aoa=aoa)
        assert (kwargs), "Need at least one attribute for AirfoilTable"
        for k, v in kwargs.items():
            assert (v.shape == aoa.shape), "Shape mismatch for AoA and %s"%k
        dmap.update(kwargs)
        #: A pandas dataframe containing the polar data
        self.data = pd.DataFrame(dmap)

    def _get_interpolator(self, key):
        """Helper to make an interpolator function"""
        int_attr = f"_{key}int"
        if not hasattr(self, int_attr):
            data = self.data
            aoa = data['aoa'].to_numpy()
            yvals = data[key].to_numpy()
            interpfn = sint.interp1d(
                aoa, yvals, bounds_error=False,
                fill_value=(yvals[0], yvals[-1]))
            setattr(self, int_attr, interpfn)
        return getattr(self, int_attr)

    def __call__(self, aoa, key):
        """Return interpolated values for a given key at angles of attack

        Example:

        .. code-block:: python

           # Load the airfoil data
           aftable = AirfoilTable.read_csv("S809.TXT")
           # Show available columns
           print(aftable.data.columns)
           cl, cd, cm = aftable(aoa_new, ['cl', 'cd', 'cm'])

        Args:
            aoa (np.ndarray): Angles of attack where value is desired
            key: string or list
        """
        if isinstance(key, str):
            interpfn = self._get_interpolator(key)
            return interpfn(aoa)

        dmap = {}
        for k in key:
            interpfn = self._get_interpolator(k)
            dmap[k] = interpfn(aoa)
        return pd.DataFrame(dmap)

    def op_pt(self, stall_margin):
        """Get operating point given a stall margin
        Args:
            stall_margin (double): Stall margin
        Return:
            op_aoa (double): Operating angle of attack
        """
        return np.minimum(self.lbyd_opt_angle,
                          self.lift_stall_angle - stall_margin)

    @property
    def lbyd_opt_angle(self):
        """Return optimal L/D angle"""
        aoa_filter = (self.data['aoa'] > -5.0) & (self.data['aoa'] < 24.0)
        lbyd = self.data['cl'].values[aoa_filter]/self.data['cd'].values[aoa_filter]
        aoa_f = np.array(self.data['aoa'][aoa_filter])
        return aoa_f[np.argmax(lbyd)]

    @property
    def lift_stall_angle(self):
        """Return lift stall angle"""
        data = self.data
        dcl = data['cl'].values[1:] - data['cl'].values[:-1]
        aoa = (data['aoa'].values[1:] + data['aoa'].values[:-1]) * 0.5
        dcl = dcl[np.where(aoa > 5)]
        aoa = aoa[np.where(aoa > 5)]
        try:
          if (np.min(dcl) < 0):
            stall_idx =  np.where( dcl < 0)[0][0]-1
            return aoa[stall_idx] - dcl[stall_idx]/(dcl[stall_idx+1] - dcl[stall_idx])
          else:
            data['dsqcl'] = np.gradient(np.gradient(data['cl']))
            t_data = data.loc[data['aoa'] > 5]
            return t_data.iloc[t_data['dsqcl'].argmin()]['aoa']
        except:
          t_data = data.loc[data['aoa'] > 5]
          return t_data.iloc[t_data['cl'].argmax()]['aoa']

    def moment_stall_angle(self):
        """Return moment stall angle"""
        data = self.data
        dcm = data['cm'].values[1:] - data['cm'].values[:-1]
        aoa = (data['aoa'].values[1:] + data['aoa'].values[:-1]) * 0.5
        dcm = dcm[np.where(aoa > 5)]
        aoa = aoa[np.where(aoa > 5)]
        try:
          if (np.min(dcm) < 0):
            stall_idx =  np.where( dcm > 0)[0][0]-1
            return aoa[stall_idx] - dcm[stall_idx]/(dcm[stall_idx+1] - dcm[stall_idx])
          else:
            data['dsqcm'] = np.gradient(np.gradient(data['cm']))
            t_data = data.loc[data['aoa'] < 10]
            return t_data.iloc[t_data['dsqcm'].argmax()]['aoa']
        except:
          t_data = data.loc[data['aoa'] < 10]
          return t_data.iloc[t_data['cm'].argmin()]['aoa']


    def cl(self, aoa):
        """Return interpolated cl values at desired angles of attack"""
        return self(aoa, "cl")

    def cd(self, aoa):
        """Return interpolated cd values at desired angles of attack"""
        return self(aoa, "cd")

    def cm(self, aoa):
        """Return interpolated cm values at desired angles of attack"""
        return self(aoa, "cm")

    def stall_margin(self, aoa):
        """Return stall margin at desired angles of attack"""
        return self.lift_stall_angle - aoa


class AirfoilTableInterp(AirfoilTable):
    """Interpolate between two AirfoilTable objects"""

    def __init__(self, aft1, aft2, wt):
        """Interpolate two AirfoilTable objects
        Args:
            aft1 (AirfoilTable): Left AirfoilTable
            aft2 (AirfoilTable): Right AirfoilTable
            wt (double): Interpolation weight
        """
        aoa_comb = aft1.data['aoa'].append(aft2.data['aoa'], ignore_index=True)
        aoa = np.array(sorted(aoa_comb.unique()))
        interp_data = {
            'cl': (1.0 - wt) * aft1.cl(aoa) + wt * aft2.cl(aoa),
            'cd': (1.0 - wt) * aft1.cd(aoa) + wt * aft2.cd(aoa),
            'cm': (1.0 - wt) * aft1.cm(aoa) + wt * aft2.cm(aoa)  }
        super().__init__(aoa = aoa, **interp_data)

class AirfoilTableCST(AirfoilTable):
    """Airfoil Polar lookup table database for multiple airfoils

    Load multiple airfoil tables from a yaml file and use AirfoilTable
    to provide an interpolation utility to lookup polars for a given set of CST
    parameters.

    """
    def __init__(self, cfg, cst_params):
        af_data = yaml.load(open(cfg.yaml_file),Loader=yaml.BaseLoader)

        afs = list(af_data.keys())
        cst_size = len(af_data[afs[0]]['cst'])
        assert(cst_size == np.size(cst_params))

        BPpsi = int(cst_size/2 - 3)
        af_shape = cst.AirfoilShape.from_cst_parameters(cst_params[0:BPpsi+1],
                                                    cst_params[2*(BPpsi+1)],
                                                    cst_params[BPpsi+1:2*(BPpsi+1)],
                                                    cst_params[2*(BPpsi+1)+1],
                                                    cst_params[-2], cst_params[-1]).yco

        shape_diffs = np.empty(( len(afs) ))
        for i,af in enumerate(afs):
            cst2d = np.array(af_data[af]['cst'],dtype=float)
            ref_shape = cst.AirfoilShape.from_cst_parameters(cst2d[0:BPpsi+1], cst2d[2*(BPpsi+1)], cst2d[BPpsi+1:2*(BPpsi+1)],  cst2d[2*(BPpsi+1)+1], cst2d[-2], cst2d[-1]).yco
            shape_diffs[i] = np.linalg.norm(ref_shape - af_shape)

        #Get the closest airfoil in the database
        c_af = afs[np.argmin(shape_diffs)]
        super().__init__(
                aoa = np.array(af_data[c_af]['aoa'],dtype=float),
                cl = np.array(af_data[c_af]['cl'],dtype=float),
                cd = np.array(af_data[c_af]['cd'],dtype=float),
                cm = np.array(af_data[c_af]['cm'],dtype=float))
