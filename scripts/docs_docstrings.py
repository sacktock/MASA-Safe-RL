"""Adapt legacy reStructuredText docstrings for Markdown, without importing MASA.

Only rendered docstrings change; Python source and runnable examples are untouched.
"""
from __future__ import annotations
import re
import textwrap
from griffe import Extension

_ROLE = re.compile(r':(?:(?:py|rst):)?(math|class|meth|func|attr|mod|obj|data|const|exc|type):`([^`]+)`')
_MATH = re.compile(r'^(?P<indent>[ \t]*)\.\. math::[ \t]*\n(?P<body>(?:[ \t]+[^\n]*\n|\n)+)', re.M)


def normalize_docstring(value: str) -> str:
    """Translate prose markup while preserving fenced code and Google sections."""
    def role(match: re.Match[str]) -> str:
        kind, target = match.groups()
        if kind == 'math':
            return '$' + target + '$'
        explicit = re.fullmatch(r'(.+?)\s*<([^>]+)>', target)
        label = explicit[1] if explicit else target.lstrip('~')
        if target.startswith('~') and not explicit:
            label = label.rsplit('.', 1)[-1]
        return '`' + label + '`'

    def math(match: re.Match[str]) -> str:
        indent = match['indent']
        body = textwrap.dedent(match['body']).strip()
        return indent + '$$\n' + textwrap.indent(body, indent) + '\n' + indent + '$$\n\n'

    chunks = re.split(r'(^\s*`{3,}[^\n]*\n.*?^\s*`{3,}[ \t]*$)', value, flags=re.M | re.S)
    for i in range(0, len(chunks), 2):
        trailing = len(chunks[i]) - len(chunks[i].rstrip('\n'))
        text = _MATH.sub(math, chunks[i] + '\n').rstrip('\n')
        text = _ROLE.sub(role, text)
        chunks[i] = re.sub(r'``([^`]+)``', r'`\1`', text) + '\n' * trailing
    return ''.join(chunks)


class MarkdownDocstrings(Extension):
    """Normalize each statically loaded Griffe object's docstring once."""
    def on_instance(self, *, obj, **kwargs) -> None:
        if obj.docstring is not None:
            obj.docstring.value = normalize_docstring(obj.docstring.value)
