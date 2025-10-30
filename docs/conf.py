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

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


# html_logo = "_static/logo.png"  # single logo for both modes
# html_static_path = ["_static"]
# html_theme_options = {
#     "top_of_page_buttons": ["view", "edit"],  # toolbar buttons
#     "sidebar_hide_name": True,
# }