# pylint: disable-all
import os
import sys
from datetime import datetime

import sphinx_bootstrap_theme

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('.'))

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints', 'sphinx.ext.autosummary', 'sphinx.ext.doctest',
    'sphinx_click.ext', 'sphinx_git', 'sphinxcontrib.katex', 'exec_directive',
    'sphinx.ext.todo'
]

# General information about the project.
project = 'Prodigy'
copyright = '{}, Lutz Klinkenberg'.format(datetime.now().year)
author = 'Lutz Klinkenberg'
master_doc = 'index'
pygments_style = 'sphinx'
todo_include_todos = True
add_module_names = False

html_theme_path = ["themes"] + sphinx_bootstrap_theme.get_html_theme_path()
html_theme = 'fixedbootstrap'
_navbar_links = [
    ('GitHub Repository', 'https://github.com/Philipp15b/probably', True)
]
html_theme_options = {
    'navbar_sidebarrel': False,
    'globaltoc_depth': -1,
    'navbar_site_name': "Contents",
    'source_link_position': "footer",
    'bootswatch_theme': "united",
    'navbar_pagenav': False,
    'navbar_links': _navbar_links
}
html_static_path = ['_static']

template_path = ['_templates']
html_sidebars = {'**': ['simpletoctree.html']}
html_extra_path = [".nojekyll"]

autodoc_default_options = {'members': True, 'undoc-members': True}
autodoc_member_order = "bysource"
autodoc_typehints_format = 'short'
autodoc_type_aliases = {'Instr': 'Union[SkipInstr, WhileInstr, IfInstr, AsgnInstr, ChoiceInstr, LoopInstr, TickInstr, ObserveInstr, ProbabilityQueryInstr, ExpectationInstr, PlotInstr, PrintInstr, OptimizationQuery, Sequence[Union[SkipInstr, WhileInstr, IfInstr, AsgnInstr, ChoiceInstr, LoopInstr, TickInstr, ObserveInstr, ProbabilityQueryInstr, ExpectationInstr, PlotInstr, PrintInstr, OptimizationQuery]]]'}
#napoleon_use_ivar = True

# et typing.TYPE_CHECKING to True to enable “expensive” typing imports
set_type_checking_flag = False  # TODO
always_document_param_types = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'graphviz': ('https://graphviz.readthedocs.io/en/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None)
}

# nitpicky emits warnings for all broken links
# nitpicky = True
