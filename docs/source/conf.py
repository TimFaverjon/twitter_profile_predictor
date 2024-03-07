# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# docs/source/conf.py

import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.path.abspath('../..'))
print(f"Updated sys.path: {sys.path}")

sys.path.insert(0, os.path.abspath('src/twitter_profile_predictor'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Twitter Profile Predictor'
copyright = '2023, Tim Faverjon'
author = 'Tim Faverjon'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
