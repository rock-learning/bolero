# -*- coding: utf-8 -*-
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# See http://sphinx-doc.org/latest/config.html for a detailed explanation

import sys
import os


# Directory 'bolero'
root_path = '../..'

# Paths are relative from this file
sys.path.insert(0, os.path.abspath(root_path))
sys.path.insert(0, os.path.abspath(root_path + '/doc/source'))
sys.path.insert(0, os.path.abspath(root_path + '/doc/source/modules/generated'))
sys.path.insert(0, os.path.abspath(root_path + '/doc/source/sphinxext'))


needs_sphinx = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    #'sphinx.ext.pngmath',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_gallery.gen_gallery',
    'breathe',
    'numpydoc',
]

import sphinx_bootstrap_theme
import sphinx_gallery
import bolero

# Breathe options, see http://michaeljones.github.io/breathe/
breathe_projects = {
    "bolero": "../build/_doxygen/xml/",
}
breathe_default_project = "bolero"
templates_path = ['_templates']
source_suffix = '.rst'
exclude_patterns = ["sphinxext"]
exclude_trees = ["_templates"]
source_encoding = 'utf-8-sig'

master_doc = 'index'
project = u'bolero'
copyright = u'2014-2017, DFKI GmbH / Robotics Innovation Center, University of Bremen / Robotics Group'
version = bolero.__version__
release = bolero.__version__
language = 'en'
today_fmt = '%B %d, %Y'
add_function_parentheses = True
show_authors = True
pygments_style = 'sphinx'
html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_theme_options = {
    'bootswatch_theme': "flatly",
    'navbar_fixed_top': "true",
    'navbar_sidebarrel': False,
}

html_static_path = ["_static"]
html_last_updated_fmt = '%b %d, %Y'
html_use_smartypants = True
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("bolero",),
    "download_section_examples": False
}

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

autosummary_generate = True
autodoc_default_flags = ['members', 'inherited-members']
# The problem is that the numpydoc doc scraper is building
# autosummary documentation after autosummary has already run.
# To avoid warnings, we have to set this to variable.
# Source: http://stackoverflow.com/questions/12206334
numpydoc_show_class_members = False

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'http://docs.python.org/': None}
