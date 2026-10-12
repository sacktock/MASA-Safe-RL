"""Documentation-only tests; no machine-learning dependencies are required."""
import unittest

try:
    from scripts.docs_docstrings import normalize_docstring
except ModuleNotFoundError as error:
    if error.name != 'griffe':
        raise
    normalize_docstring = None


@unittest.skipIf(normalize_docstring is None, 'Install the docs dependency group to test rendering.')
class DocstringTests(unittest.TestCase):
    def test_inline_math_and_roles(self):
        self.assertEqual(normalize_docstring(r'Cost :math:`x^2` in :class:`~masa.Constraint` and ``info``.'),
                         r'Cost $x^2$ in `Constraint` and `info`.')

    def test_display_math(self):
        result = normalize_docstring('Loss:\n\n.. math::\n\n    L = x^2\n\nReturns:\n    A scalar.')
        self.assertIn('$$\nL = x^2\n$$', result)
        self.assertIn('Returns:\n    A scalar.', result)

    def test_explicit_role_title(self):
        self.assertEqual(normalize_docstring(':class:`a constraint <masa.Constraint>`'), '`a constraint`')

    def test_google_sections_unchanged(self):
        text = 'Args:\n    value (float): An input.\n\nReturns:\n    float: A result.'
        self.assertEqual(normalize_docstring(text), text)

    def test_fenced_example_unchanged(self):
        text = 'Example:\n\n```python\nprint(":math:`x`")\n```\n\nAfter.'
        self.assertEqual(normalize_docstring(text), text)

    def test_idempotent(self):
        text = ':math:`x` and ``code``.\n\n.. math::\n\n    x = 1\n'
        once = normalize_docstring(text)
        self.assertEqual(normalize_docstring(once), once)


if __name__ == '__main__':
    unittest.main()
