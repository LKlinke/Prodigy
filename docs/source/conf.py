# pylint: disable-all
# Sphinx configuration. Build with `make docs`; the output lands in docs/build/html.
# The API reference is generated from the docstrings (autosummary, see api.rst),
# the command-line reference from the click commands (sphinx-click).
import os
import re
import sys
from datetime import datetime
from importlib.metadata import version as _version

sys.path.insert(0, os.path.abspath('../..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    # Before sphinx_autodoc_typehints: that one imports the module sphinx.ext.autodoc.mock, which
    # then shadows the function mock that sphinx_click imports from sphinx.ext.autodoc (Sphinx 9).
    'sphinx_click',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

project = 'Prodigy'
author = 'Lutz Klinkenberg'
copyright = f'{datetime.now().year}, {author}'
release = _version('prodigy-ppl')
version = release

master_doc = 'index'
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
exclude_patterns = ['_build']
templates_path = ['_templates']

# Furo follows the system's light or dark mode.
html_theme = 'furo'
html_logo = '../../icon.svg'
html_favicon = '../../prodigy/web/static/app/favicon.svg'
html_title = f'Prodigy {release}'
html_theme_options = {
    'source_repository': 'https://github.com/LKlinke/Prodigy/',
    'source_branch': 'main',
    'source_directory': 'docs/source/',
}

autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {'members': True, 'undoc-members': True, 'show-inheritance': True}
autodoc_member_order = 'bysource'
add_module_names = False
always_document_param_types = True

myst_heading_anchors = 3

# frozendict's pure-Python fallback has annotations that cannot be resolved.
suppress_warnings = ['sphinx_autodoc_typehints.forward_reference']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sympy': ('https://docs.sympy.org/latest/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
}

# Many module docstrings list their members with autodoc directives (or `.. click::`) for the
# hand-written docs of the past. The generated pages document every member anyway, so these
# directives are dropped from module docstrings instead of documenting the members twice.
_DIRECTIVE = re.compile(r'\s*\.\. (auto\w+|click)::')


def _drop_autodoc_directives(app, what, name, obj, options, lines):
    if what != 'module':
        return
    kept, skip_indent = [], None
    for line in lines:
        indent = len(line) - len(line.lstrip())
        if skip_indent is not None and (not line.strip() or indent > skip_indent):
            continue
        skip_indent = None
        if _DIRECTIVE.match(line):
            skip_indent = indent
            continue
        kept.append(line)
    lines[:] = kept


def _escape_click_help(app, ctx, lines):
    # Option help is plain text for the terminal; a lone * would start emphasis in reST.
    lines[:] = [re.sub(r'(?<![*\\])\*(?!\*)', r'\\*', line) for line in lines]


def _write_grammar(app):
    # grammar.rst includes the grammar the parser is built from.
    from prodigy.pgcl.parser import _PGCL_GRAMMAR
    target = os.path.join(app.srcdir, '_generated', 'pgcl.lark')
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, 'w', encoding='utf-8') as file:
        file.write(_PGCL_GRAMMAR.strip() + '\n')


def setup(app):
    app.connect('builder-inited', _write_grammar)
    app.connect('autodoc-process-docstring', _drop_autodoc_directives)
    app.connect('sphinx-click-process-options', _escape_click_help)
