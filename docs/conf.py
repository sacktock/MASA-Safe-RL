# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MASA'
copyright = '2025, Alexander Goodall, Edwin Hamel De Le Court, Omar Adalat, Francesco Belardinelli'
author = 'Alexander Goodall, Edwin Hamel De Le Court, Omar Adalat, Francesco Belardinelli'
release = 'v0'

html_title = "Multi and Single Agent Safe Reinforcement Learning"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

import os
import sys

sys.path.insert(0, os.path.abspath("."))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


# html_logo = "_static/logo.png"  # single logo for both modes
# html_static_path = ["_static"]
html_theme_options = {
    # "light_logo": "img/masa_black.svg",
    # "dark_logo": "img/masa_white.svg",

    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/sacktock/MASA-Safe-RL",
    "source_branch": "main",
    "source_directory": "docs/",
    
}