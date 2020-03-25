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
    
    def __init__(self, parent_dir=''):
        
        self.parent_dir = parent_dir
        self.code_dir = os.path.join(self.parent_dir, 'numenv', 'python')
        
    def create_dirs(self):
        if not os.path.exists(self.code_dir):
            self._create_subdirs()
            self._write_init_file()
            
        
    def write_topology_code(self, model):
        src_path = os.path.join(self.code_dir, 'src')
        if type(model) is str:
            stpl_file = model
            instance = load_pickled_data(stpl_file)
            instance.assemble()
        else:
            instance = model
        codegen = generators.template_codegen(instance.topology)
        codegen.write_code_file(src_path)
    
    def _create_subdirs(self):
        for d in ['src']:
            subdir = os.path.join(self.code_dir, d)
            if not os.path.exists(subdir):
                os.makedirs(subdir)            
    
    def _write_mainfile(self):
        pass        
    
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
        
    def create_dirs(self):
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
    
