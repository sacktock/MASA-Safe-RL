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
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
]

myst_enable_extensions = [
    "eval-rst",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Napoleon settings
napoleon_use_ivar = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_attr_annotations = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
# napoleon_custom_sections = [("Returns", "params_style")]

# Autodoc
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "signature"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ["custom.css"]

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