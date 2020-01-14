#
#__import__('pkg_resources').declare_namespace(__name__)

from .numerics.core.systems import multibody_system, simulation, configuration
from .codegen.projects import standalone_project, templatebased_project

__all__ = ['multibody_system', 'simulation', 'configuration', 
           'standalone_project', 'standalone_project']
