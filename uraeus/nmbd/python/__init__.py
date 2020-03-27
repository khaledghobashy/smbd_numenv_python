#
#__import__('pkg_resources').declare_namespace(__name__)

from .engine.systems import multibody_system, simulation, configuration, assembly, import_source
from .codegen.projects import standalone_project, templatebased_project

__all__ = ['multibody_system', 'simulation', 'configuration', 'assembly',
           'standalone_project', 'standalone_project', 'import_source']
