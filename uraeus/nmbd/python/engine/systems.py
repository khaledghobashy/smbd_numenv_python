# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 08:49:16 2019

@author: khaled.ghobashy
"""
# Standard library imports
import os
import json
import itertools
import functools
import importlib.util
from collections import namedtuple

# 3rd party library imports
import numpy as np
import pandas as pd
from numpy.linalg import multi_dot

# Local applicataion imports
from .numerics.solvers import kds_solver, dds_solver
from .numerics.math_funcs import G
from .utilities.decoders import JSON_Decoder

# =============================================================================

# Helper function to import the generated numerical python module
# from the generated script file
def import_source(source_dir, model_name):
    source_file = os.path.join(source_dir, '%s.py'%model_name)
    # importing the topology source file 
    spec  = importlib.util.spec_from_file_location(model_name, source_file)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model

###############################################################################

class multibody_system(object):
    def __init__(self, system):
        try:
            self.topology = system.topology()
        except AttributeError:
            self.topology = system
        try:
            self.Subsystems = system.subsystems
        except AttributeError:
            pass
        
###############################################################################
###############################################################################

class simulation(object):
    def __init__(self, name, model, typ='kds'):
        self.name = name
        self.assembly = model.topology
        if typ == 'kds':
            self.soln = kds_solver(self.assembly)
        elif typ == 'dds':
            self.soln = dds_solver(self.assembly)
        else:
            raise ValueError('Bad simulation type argument : %r'%typ)
    
    def set_time_array(self, duration, spacing):
        self.soln.set_time_array(duration, spacing)
        
    def solve(self, run_id=None):
        run_id = '%s_temp'%self.name if run_id is None else run_id
        self.soln.solve(run_id)
    
    def save_as_csv(self, path, filename, key='pos'):
        assert key in {'pos', 'vel', 'acc'}
        path = os.path.abspath(path)
        filepath = os.path.join(path, filename)
        data = getattr(self.soln, '%s_dataframe'%key)
        data.to_csv('%s.csv'%filepath, index=True)
        print('Position level simulation results saved as %s.csv at %s'%(filename, path))
    
    def save_as_npz(self, path, filename):
        path = os.path.abspath(path)
        filepath = os.path.join(path, filename)
        coordinates = list(self.soln.pos_dataframe.keys())
        constraints = list(self.soln._reactions_indicies)
        pos = self.soln.pos_dataframe.values
        vel = self.soln.vel_dataframe.values
        acc = self.soln.acc_dataframe.values
        lgr = self.soln.lgr_dataframe.values
        data = {'coordinates':coordinates, 
                'constraints':constraints,
                'pos':pos, 
                'vel':vel, 
                'acc':acc,
                'lgr':lgr}
        np.savez_compressed(filepath, **data)
        print('System States results saved as %s.npz at %s'%(filename, path))

    def set_initial_states(self, npz_file):
        data = np.load(npz_file)
        pos = data['pos'][-1,:-1][:,np.newaxis]
        vel = data['vel'][-1,:-1][:,np.newaxis]
        self.soln.set_initial_states(pos, vel)
    
    def load_simulation_results(self, npz_file):
        data = np.load(npz_file)
        pos = data['pos']
        vel = data['vel']
        acc = data['acc']
        lgr = data['lgr']
        columns = data['coordinates']

        self.soln.pos_dataframe = pd.DataFrame(
                data = pos, columns = columns)
        self.soln.vel_dataframe = pd.DataFrame(
                data = vel, columns = columns)
        self.soln.acc_dataframe = pd.DataFrame(
                data = acc, columns = columns)
        self.soln.lgr_dataframe = pd.DataFrame(
                data = lgr, columns = range(self.soln.model.nc + 1))

        self.soln._pos_history = {i: pos[i][:,None] for i in range(pos.shape[0])}
        self.soln._vel_history = {i: vel[i][:,None] for i in range(pos.shape[0])}
        self.soln._acc_history = {i: acc[i][:,None] for i in range(pos.shape[0])}
        self.soln._lgr_history = {i: lgr[i][:,None] for i in range(pos.shape[0])}

        
    def eval_reactions(self):
        self.soln.eval_reactions()
    
###############################################################################
###############################################################################

class configuration(object):
    def __init__(self, name):
        self.name = name

        self.R_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)

        self.Rd_ground = np.array([[0], [0], [0]], dtype=np.float64)
        self.Pd_ground = np.array([[0], [0], [0], [0]], dtype=np.float64)
    
    def construct_from_json(self, json_file, assemble=False):

        self.decoded_data = JSON_Decoder(json_file, self)

        if not assemble:
            _attributes = self.decoded_data.user_inputs.keys()
            for key in _attributes:
                value = getattr(self.decoded_data, key)
                setattr(self, key, value)
        else:
            self.assemble()
        
    def assemble(self):
        self.decoded_data.assemble()
        _attributes = itertools.chain(self.decoded_data.evaluations.keys(),
                                      self.decoded_data.outputs.keys())
        
        for key in _attributes:
            value = getattr(self, key)
            setattr(self, key, value)

    def export_json(self, file_path='', subsystem_initial=''):
        new_data = self.decoded_data.data_dict.copy()
        new_data['information']['subsystem_name'] = subsystem_initial
        user_inputs = new_data['user_inputs']
        
        for key, value in user_inputs.items():
            if isinstance(value, (dict,)):
                if value['constructor'] == 'array':
                    user_inputs[key]['args'] = list(getattr(self, key).flat[:])
                else:
                    pass
            else:
                user_inputs[key] = getattr(self, key)
        
        file_name = os.path.join(file_path, '%s.json'%self.name)
        json_text = json.dumps(new_data, indent=4)
        with open(file_name, 'w') as f:
            f.write(json_text)
    

###############################################################################
###############################################################################

# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def subsystems_creator(mapping, templates_dir):
    container = namedtuple('Subsystems', mapping.keys())
    data = {}
    for subsystem, template in mapping.items():
        model = import_source(templates_dir, template)
        #spec = importlib.util.spec_from_file_location(template, f"{templates_dir}/{template}.py")
        #foo = importlib.util.module_from_spec(spec)
        #spec.loader.exec_module(foo)
        data[subsystem] = model.topology(subsystem)
    return container(**data)

class assembly(object):
    def __init__(self, json_file, templates_dir):
        assembly_data = self.construct_from_json(json_file)

        self.subsystems = subsystems_creator(assembly_data['subsystems'], templates_dir)
        self.interface_map = assembly_data['interface_map']
        self.indicies_map  = assembly_data['nodes_indicies']
        self.mapped_vir_coordinates  = assembly_data['mapped_vir_coordinates']
        self.mapped_vir_velocities   = assembly_data['mapped_vir_velocities']
        self.mapped_vir_accelerations  = assembly_data['mapped_vir_accelerations']

        self.nrows = sum([sub.nrows for sub in self.subsystems]) + 2
        self.ncols = sum([sub.ncols for sub in self.subsystems]) + 2
        self.n  = sum([sub.n for sub in self.subsystems]) + 7
        self.nc = sum([sub.nc for sub in self.subsystems]) + 7

        self.R_ground  = np.array([[0], [0], [0]], dtype=np.float64)
        self.P_ground  = np.array([[1], [0], [0], [0]], dtype=np.float64)
        self.Pg_ground = np.array([[1], [0], [0], [0]], dtype=np.float64)

        self.m_ground = np.eye(3, dtype=np.float64)
        self.Jbar_ground = np.eye(3, dtype=np.float64)

        self.gr_rows = np.array([0, 1], dtype=np.intc)
        self.gr_jac_rows = np.array([0, 0, 1, 1], dtype=np.intc)
        self.gr_jac_cols = np.array([0, 1, 0, 1], dtype=np.intc)
        self._t = 0
        self._i = 0

    @property
    def t(self):
        return self._t
    @t.setter
    def t(self,t):
        self._t = t
        for sub in self.subsystems:
            sub.t = t
    
    @property
    def i(self):
        return self._i
    @i.setter
    def i(self, i):
        self._i = i
        for sub in self.subsystems:
            sub.i = i

    @staticmethod
    def construct_from_json(json_file):
        with open(json_file, 'r') as f:
            json_text = f.read()
        data_dict = json.loads(json_text)
        return data_dict
    
    def initialize(self, q, qd, qdd, lgr):
        self.t = 0
        self.assemble()
        self._set_states_arrays(q, qd, qdd, lgr)
        self._map_states_arrays()
        self.set_initial_states()
        self.eval_constants()

    def _set_states_arrays(self, q, qd, qdd, lgr):
        self._q = q
        self._qd = qd
        self._qdd = qdd
        self._lgr = lgr
    
    def _map_states_arrays(self):
        self._map_gen_coordinates()
        self._map_gen_velocities()
        self._map_gen_accelerations()
        self._map_lagrange_multipliers()

    def set_initial_states(self):
        for sub in self.subsystems:
            sub.set_initial_states()
        coordinates = [sub._q for sub in self.subsystems if len(sub._q)!=0]
        np.concatenate([self.R_ground, self.P_ground, *coordinates], out=self._q)

        veolicties = [sub._qd for sub in self.subsystems if len(sub._qd)!=0]
        np.concatenate([self.Rd_ground, self.Pd_ground, *veolicties], out=self._qd)


    def assemble(self):
        offset = 2
        for sub in self.subsystems:
            sub.assemble(self.indicies_map, self.interface_map, offset)
            offset += sub.nrows

        self.rows = np.concatenate([s.rows for s in self.subsystems])
        self.jac_rows = np.concatenate([s.jac_rows for s in self.subsystems])
        self.jac_cols = np.concatenate([s.jac_cols for s in self.subsystems])

        self.rows = np.concatenate([self.gr_rows, self.rows])
        self.jac_rows = np.concatenate([self.gr_jac_rows, self.jac_rows])
        self.jac_cols = np.concatenate([self.gr_jac_cols, self.jac_cols])

        self.reactions_indicies = sum([sub.reactions_indicies for sub in self.subsystems],[])

    def _map_coordinates(self, mapping, config=False):
        for virtual, actual in mapping.items():
            if config:
                virtual = '%s.config.%s'%tuple(virtual.split('.'))
                try:
                    actual  = '%s.config.%s'%tuple(actual.split('.'))
                except TypeError:
                    pass
            try:
                rsetattr(self.subsystems, virtual, rgetattr(self.subsystems, actual))
            except AttributeError:
                rsetattr(self.subsystems, virtual, rgetattr(self, actual))
        
    def eval_constants(self):
        self._map_coordinates(self.mapped_vir_coordinates, config=True)
        for sub in self.subsystems:
            sub.eval_constants()

    def _map_gen_coordinates(self):
        q = self._q
        self.R_ground = q[0:3]
        self.P_ground = q[3:7]
        offset = 7
        for sub in self.subsystems:
            qs = q[offset: sub.n+offset]
            sub._q = qs
            sub._map_gen_coordinates()
            offset += sub.n
        self._map_coordinates(self.mapped_vir_coordinates)

    def _map_gen_velocities(self):
        qd = self._qd
        self.Rd_ground = qd[0:3]
        self.Pd_ground = qd[3:7]
        offset = 7
        for sub in self.subsystems:
            qs = qd[offset: sub.n+offset]
            sub._qd = qs
            sub._map_gen_velocities()
            offset += sub.n
        self._map_coordinates(self.mapped_vir_velocities)

    def _map_gen_accelerations(self):
        qdd = self._qdd
        self.Rdd_ground = qdd[0:3]
        self.Pdd_ground = qdd[3:7]
        offset = 7
        for sub in self.subsystems:
            qs = qdd[offset: sub.n + offset]
            sub._qdd = qs
            sub._map_gen_accelerations()
            offset += sub.n
        self._map_coordinates(self.mapped_vir_accelerations)

    def _map_lagrange_multipliers(self):
        Lambda = self._lgr
        offset = 7
        for sub in self.subsystems:
            l = Lambda[offset: sub.nc + offset]
            sub._lgr = l
            sub._map_lagrange_multipliers()
            offset += sub.nc

    def eval_pos_eq(self):
        pos_ground_eq_blocks = [self.R_ground,(-1*self.Pg_ground + self.P_ground)]
        for sub in self.subsystems:
            sub.eval_pos_eq()
        eq_blocks = (s.pos_eq_blocks for s in self.subsystems)
        self.pos_eq_blocks = (*pos_ground_eq_blocks, *itertools.chain(*eq_blocks))
    
    def eval_vel_eq(self):
        vel_ground_eq_blocks = [np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64)]
        for sub in self.subsystems:
            sub.eval_vel_eq()
        eq_blocks = (s.vel_eq_blocks for s in self.subsystems)
        self.vel_eq_blocks = (*vel_ground_eq_blocks, *itertools.chain(*eq_blocks))

    def eval_acc_eq(self):
        acc_ground_eq_blocks = [np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64)]
        for sub in self.subsystems:
            sub.eval_acc_eq()
        eq_blocks = (s.acc_eq_blocks for s in self.subsystems)
        self.acc_eq_blocks = (*acc_ground_eq_blocks, *itertools.chain(*eq_blocks))

    def eval_jac_eq(self):
        jac_ground_eq_blocks = [np.eye(3, dtype=np.float64),np.zeros((3,4),dtype=np.float64),np.zeros((4,3),dtype=np.float64),np.eye(4, dtype=np.float64)]
        for sub in self.subsystems:
            sub.eval_jac_eq()
        eq_blocks = (s.jac_eq_blocks for s in self.subsystems)
        self.jac_eq_blocks = (*jac_ground_eq_blocks, *itertools.chain(*eq_blocks))
    
    def eval_mass_eq(self):
        mass_ground_eq_blocks = [self.m_ground*np.eye(3, dtype=np.float64),4*multi_dot([G(self.P_ground).T,self.Jbar_ground,G(self.P_ground)])]
        for sub in self.subsystems:
            sub.eval_mass_eq()
        eq_blocks = (s.mass_eq_blocks for s in self.subsystems)
        self.mass_eq_blocks = (*mass_ground_eq_blocks, *itertools.chain(*eq_blocks))

    def eval_frc_eq(self):
        frc_ground_eq_blocks = [np.zeros((3,1),dtype=np.float64),np.zeros((4,1),dtype=np.float64)]
        for sub in self.subsystems:
            sub.eval_frc_eq()
        eq_blocks = (s.frc_eq_blocks for s in self.subsystems)
        self.frc_eq_blocks = (*frc_ground_eq_blocks, *itertools.chain(*eq_blocks))

    def eval_reactions_eq(self):
        self.reactions = {}
        for sub in self.subsystems:
            sub.eval_reactions_eq()
            for k,v in sub.reactions.items():
                self.reactions['%s%s'%(sub.prefix,k)] = v

