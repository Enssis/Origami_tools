# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../../../src'))  # Path to src/ directory  

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'origami_tools'
copyright = '2025, Druart Mateo'
author = 'Druart Mateo'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [  
    'sphinx.ext.autodoc',        # Auto-generate docs from docstrings  
    'sphinx.ext.napoleon',       # Support Google/NumPy-style docstrings  
    'sphinx_autodoc_typehints',  # Include type hints in docs  
    'sphinx_rtd_theme',          # Use Read the Docs theme  
]  

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'origami_tools Documentation'  # Title in browser tabs 