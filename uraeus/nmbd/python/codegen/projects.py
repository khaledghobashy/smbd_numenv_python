#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:36:49 2019

@author: khaledghobashy
"""

# Standard library imports
import os
import shutil
import textwrap

# Third party imports
import cloudpickle

# Local application imports
from . import generators

##########################################################################

def load_pickled_data(file):
    with open(file, 'rb') as f:
        instance = cloudpickle.load(f)
    return instance

##########################################################################


class standalone_project(object):
    
    def __init__(self, stpl_file, parent_dir=''):
        
        self.parent_dir = parent_dir
        self.code_dir = os.path.join(self.parent_dir, 'numenv', 'python')

        self._load_symbolic_topology(stpl_file)
        
    def create_dirs(self, clean=False):
        if os.path.exists(self.code_dir):
            if clean:
                shutil.rmtree(self.code_dir)
                self._create_subdirs()
                self._create_common_dirs()
        self._create_subdirs()
        self._write_init_file()
            
        
    def write_topology_code(self):
        topology = self.topology
        src_path = os.path.join(self.code_dir, 'src')
        codegen = generators.template_codegen(topology)
        codegen.write_code_file(src_path)
    
    def _create_subdirs(self):
        for d in ['src']:
            subdir = os.path.join(self.code_dir, d)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
    
    def _load_symbolic_topology(self, stpl_file):
        instance = load_pickled_data(stpl_file)
        instance.assemble()
        self.topology = instance.topology
            
    
    def write_mainfile(self):
        text = '''
                import numpy as np
                import pandas as pd
                
                
                from uraeus.numenv.python import multibody_system, simulation, configuration
                #from src import topology, configuration
                
                
                num_model  = multibody_system(topology)
                
                num_model.topology.config = num_config
                
                inputs_df = pd.read_csv('../../config_inputs/config.csv', index_col=0)
                # input the configuration data here ...
                inputs_df.loc['P_ground'] = [1, 0, 0, 0]
                
                
                # Saving the configuration as a .csv file.
                inputs_df.to_csv('../../config_inputs/new.csv')
                
                num_config.load_from_dataframe(inputs_df)
                
                # Setting actuation data
                #num_config.UF_mcs_act_1 = lambda t :  np.deg2rad(360)*t
           
                sim = simulation('sim', num_model, 'kds')
                sim.set_time_array(1, 100)
                sim.solve()
                sim.save_results('../../results', 'sim')
            
        '''        
        text = text.expandtabs()
        text = textwrap.dedent(text)        
        
        file_path = os.path.join(self.code_dir, 'main')
        file_name = '%s.py'%file_path
        with open(file_name, 'w') as file:
            file.write(text)
        print('File full path : %s'%file_name)
        
    
    def _write_init_file(self):
        file_path = os.path.join(self.code_dir, '__init__.py')
        file_name = file_path
        with open(file_name, 'w') as file:
            file.write('#')
        
        src_path = os.path.join(self.code_dir, 'src',' __init__.py')
        file_name = src_path
        with open(file_name, 'w') as file:
            file.write('#')
        


class templatebased_project(object):
    def __init__(self, database_dir):
        self._parent_dir = os.path.abspath(database_dir)
        self._code_dir = os.path.join(self._parent_dir, 'numenv', 'python')
        self._templates_dir  = os.path.join(self._code_dir, 'templates')
        #self._assemblies_dir = os.path.join(self._code_dir, 'assemblies')
        
    def create_dirs(self, clean=False):
        if not os.path.exists(self._templates_dir):
            os.makedirs(self._templates_dir)
            
    def write_topology_code(self, model):
        src_path = self._templates_dir
        if type(model) is str:
            stpl_file = model
            instance = load_pickled_data(stpl_file)
            instance.assemble()
        else:
            instance = model
        codegen = generators.template_codegen(instance.topology)
        codegen.write_code_file(src_path)
            
    def _write_init_file(self):
        pass
    
