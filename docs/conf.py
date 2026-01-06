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
napoleon_attr_annotations = True
napoleon_use_admonition_for_references = True
# See https://github.com/sphinx-doc/sphinx/issues/9119
napoleon_custom_sections = [("Returns", "params_style")]

# Autodoc
autoclass_content = "both"
autodoc_preserve_defaults = True

autodoc_typehints = "description"

# This function removes the content before the parameters in the __init__ function.
# This content is often not useful for the website documentation as it replicates
# the class docstring.
def remove_lines_before_parameters(app, what, name, obj, options, lines):
    if what == "class":
        # ":param" represents args values
        first_idx_to_keep = next(
            (i for i, line in enumerate(lines) if line.startswith(":param")), 0
        )
        lines[:] = lines[first_idx_to_keep:]


def setup(app):
    app.connect("autodoc-process-docstring", remove_lines_before_parameters)

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