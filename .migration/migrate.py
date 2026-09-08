"""One-time, loss-checked conversion of MASA's MyST documentation to Zensical.

Normal builds read the resulting Markdown directly, without a compatibility layer.
"""
from __future__ import annotations
import html
import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

ROOT = Path.cwd()
DOCS = ROOT / 'docs'
BLOCK = re.compile(r'^(`{3,})\{([^}]+)\}([^\n]*)\n(.*?)^\1[ \t]*$', re.M | re.S)
ROLE = re.compile(r'(?:\{(doc|ref|class|meth|func|attr|mod|math)\}|:(doc|ref|class|meth|func|attr|mod|math):)`([^`]+)`')
API = re.compile(r'^\.\. (auto(?:class|function|method|module))::\s*(\S+)\s*\n((?:[ \t]+[^\n]*\n|\n)*)', re.M)
LINK = re.compile(r'(!?\[[^\]\n]*\]\()((?:[^()\n]|\([^()\n]*\))*)(\))')
original = {p: p.read_text(encoding='utf-8') for p in DOCS.rglob('*.md')}
excluded = {DOCS / 'README.md', DOCS / 'Tutorials/Shielding/Shielding.md'}


def split_options(body: str) -> tuple[dict[str, str], str]:
    options = {}
    lines = body.splitlines()
    while lines and (not lines[0].strip() or re.match(r'^\s*:[\w-]+:', lines[0])):
        line = lines.pop(0).strip()
        if line:
            key, value = line[1:].split(':', 1)
            options[key] = value.strip()
    return options, '\n'.join(lines).strip()


def link_target(value: str, page: Path) -> str:
    value = value.strip().strip('<>')
    parts = urlsplit(value)
    if parts.scheme or parts.netloc or not parts.path:
        return value
    candidate = (page.parent / unquote(parts.path)).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError(f'{page}: link escapes repository: {value}')
    if candidate.suffix in ('', '.html') and candidate.with_suffix('.md').is_file():
        candidate = candidate.with_suffix('.md')
    if candidate.is_relative_to(ROOT / 'images'):
        candidate = DOCS / 'assets/images' / candidate.relative_to(ROOT / 'images')
    if candidate.is_relative_to(DOCS):
        import os
        target = Path(os.path.relpath(candidate, page.parent)).as_posix()
        return urlunsplit(('', '', quote(target, safe='/._-'), parts.query, parts.fragment))
    return 'https://github.com/nightly/MASA-Safe-RL/blob/main/' + quote(candidate.relative_to(ROOT).as_posix(), safe='/._-') + (('#' + parts.fragment) if parts.fragment else '')


def roles(text: str, page: Path) -> str:
    def replace(m):
        role, value = m[1] or m[2], m[3]
        explicit = re.fullmatch(r'(.+?)\s*<([^>]+)>', value)
        target = explicit[2] if explicit else value
        label = explicit[1] if explicit else target.lstrip('~')
        if target.startswith('~') and not explicit:
            label = target.rsplit('.', 1)[-1]
        target = target.lstrip('~')
        if role == 'math':
            return '$' + value + '$'
        if role == 'doc':
            if not explicit:
                resolved = (page.parent / unquote(target)).with_suffix('.md').resolve()
                title = re.search(r'^# (.+)$', original.get(resolved, ''), re.M)
                label = title[1] if title else Path(target).name
            return f'[{label}]({link_target(target, page)})'
        if target.startswith('masa.'):
            return f'[`{label}`][{target}]'
        return f'`{label}`'
    return ROLE.sub(replace, text)


def rst(text: str, page: Path) -> str:
    def api(m):
        kind, target, tail = m.groups()
        options, remaining = split_options(tail)
        if remaining:
            raise ValueError(f'Unexpected API content for {target}: {remaining}')
        allowed = {'members', 'show-inheritance', 'special-members', 'private-members', 'inherited-members', 'undoc-members', 'no-index'}
        if options.keys() - allowed:
            raise ValueError(f'Unsupported API options: {options.keys() - allowed}')
        result = [f'::: {target}']
        settings = {}
        if options.get('members'):
            settings['members'] = [x.strip() for x in options['members'].split(',') if x.strip()]
        elif kind != 'automodule':
            settings['members'] = 'members' in options
        if 'inherited-members' in options:
            settings['inherited_members'] = True
        if 'undoc-members' in options:
            settings['show_if_no_docstring'] = True
        if 'private-members' in options:
            settings['filters'] = ['!^__']
        elif options.get('special-members'):
            special = [re.escape(x.strip()) for x in options['special-members'].split(',') if x.strip()]
            settings['filters'] = ['!^_', '^(' + '|'.join(special) + ')$']
        if 'no-index' in options:
            settings['skip_local_inventory'] = True
        if settings:
            result.append('    options:')
            for key, value in settings.items():
                result.append(f'      {key}: {json.dumps(value)}')
        return '\n'.join(result) + '\n\n'
    text = API.sub(api, text)
    text = re.sub(r'^\.\. math::\s*\n((?:[ \t]+[^\n]*\n|\n)+)', lambda m: '$$\n' + '\n'.join(x.strip() for x in m[1].strip().splitlines()) + '\n$$\n\n', text + '\n', flags=re.M)
    text = re.sub(r'^([^\n]+)\n[~]{3,}\s*$', r'### \1', text, flags=re.M)
    text = re.sub(r'``([^`]+)``', r'`\1`', text)
    text = roles(text, page)
    if re.search(r'^\.\. \w+::', text, re.M):
        raise ValueError(f'Unconverted RST in {page}: {text}')
    return text.strip()


def convert(text: str, page: Path) -> str:
    def block(m):
        _, kind, arg, body = m.groups()
        arg = arg.strip()
        if kind == 'toctree':
            return ''
        if kind == 'eval-rst':
            return rst(body, page)
        options, body = split_options(body)
        if kind in {'important', 'seealso'}:
            title = 'Important' if kind == 'important' else 'See also'
            return f'!!! {"warning" if kind == "important" else "info"} "{title}"\n\n' + '\n'.join('    ' + line if line else '' for line in roles(body, page).splitlines())
        if kind == 'figure':
            if options.keys() - {'alt', 'width', 'align', 'name'}:
                raise ValueError(f'Unsupported figure options in {page}: {options}')
            alt = options.get('alt', body.replace('`', ''))
            attrs = (' width="' + html.escape(options['width'].removesuffix('px')) + '"') if 'width' in options else ''
            name = (' id="' + html.escape(options['name']) + '"') if 'name' in options else ''
            return f'<figure markdown="1"{name}>\n\n![{alt}]({link_target(arg, page)}){{ loading=lazy{attrs} }}\n\n<figcaption markdown="1">\n{body}\n</figcaption>\n</figure>'
        if kind == 'list-table':
            rows = []
            for line in body.splitlines():
                if line.startswith('* - '):
                    rows.append([line[4:]])
                elif line.startswith('  - '):
                    rows[-1].append(line[4:])
                elif line.strip():
                    rows[-1][-1] += ' ' + line.strip()
            if options.get('header-rows', '1') != '1' or not rows or len({len(r) for r in rows}) != 1:
                raise ValueError(f'Unsupported table in {page}')
            rows = [[roles(re.sub(r'``([^`]+)``', r'`\1`', c), page).replace('|', r'\|') for c in row] for row in rows]
            lines = ['| ' + ' | '.join(row) + ' |' for row in rows]
            lines.insert(1, '| ' + ' | '.join('---' for _ in rows[0]) + ' |')
            return (('**' + arg + '**\n\n') if arg else '') + '\n'.join(lines)
        raise ValueError(f'Unknown MyST directive {kind!r} in {page}')
    text = BLOCK.sub(block, text)
    chunks = re.split(r'(^`{3,}[^\n]*\n.*?^`{3,}[ \t]*$)', text, flags=re.M | re.S)
    for index in range(0, len(chunks), 2):
        chunks[index] = roles(chunks[index], page)
        chunks[index] = LINK.sub(lambda m: m[1] + link_target(m[2], page) + m[3], chunks[index])
    return re.sub(r'\n{4,}', '\n\n\n', ''.join(chunks)).rstrip() + '\n'


def toml(value):
    if isinstance(value, dict):
        return '{' + ', '.join(json.dumps(k) + ' = ' + toml(v) for k, v in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ', '.join(toml(v) for v in value) + ']'
    return json.dumps(value, ensure_ascii=False)


visited = set()

def nav_page(page: Path, explicit=None):
    if not page.is_file():
        raise ValueError(f'Missing navigation page: {page}')
    text = original[page]
    title = re.search(r'^# (.+)$', text, re.M)
    label = explicit or (title[1].strip() if title else page.stem)
    rel = page.relative_to(DOCS).as_posix()
    visited.add(page)
    children = []
    for m in BLOCK.finditer(text):
        if m[2] != 'toctree':
            continue
        _, body = split_options(m[4])
        for line in body.splitlines():
            target = line.strip()
            if not target:
                continue
            match = re.fullmatch(r'(.+?)\s*<([^>]+)>', target)
            name = match[1] if match else None
            target = match[2] if match else target
            child = (page.parent / (target + ('' if target.endswith('.md') else '.md'))).resolve()
            if child not in visited and child not in excluded:
                children.append(nav_page(child, name))
    return {label: [{'Overview': rel}, *children] if children else rel}


def main():
    nav = [{'Home': 'index.md'}]
    visited.add(DOCS / 'index.md')
    for m in BLOCK.finditer(original[DOCS / 'index.md']):
        if m[2] != 'toctree':
            continue
        opts, body = split_options(m[4])
        section = []
        for line in body.splitlines():
            if line.strip():
                page = (DOCS / (line.strip() + '.md')).resolve()
                if page not in visited:
                    section.append(nav_page(page))
        nav.append({opts['caption']: section})
    for page in sorted(set(original) - excluded - visited):
        top = page.relative_to(DOCS).parts[0]
        section_name = {'Common': 'Common API'}.get(top, top)
        section = next(item[section_name] for item in nav if section_name in item)
        section.append(nav_page(page))
    config = ROOT / 'zensical.toml'
    text = config.read_text(encoding='utf-8')
    if '# MIGRATION_NAV' not in text:
        raise ValueError('Expected navigation placeholder')
    config.write_text(text.replace('# MIGRATION_NAV', 'nav = [\n' + ''.join('  ' + toml(item) + ',\n' for item in nav) + ']'), encoding='utf-8')
    before = sum(len(re.findall(r'^\.\. auto(?:class|function|method|module)::', t, re.M)) for p, t in original.items() if p not in excluded)
    for page, text in original.items():
        if page not in excluded:
            page.write_text(convert(text, page), encoding='utf-8')
    after = sum(len(re.findall(r'^::: masa\.', p.read_text(), re.M)) for p in original if p not in excluded)
    if before != after:
        raise ValueError(f'API references lost: {before} -> {after}')
    for name in ['logo.png', 'logo.svg', 'logo_large.png', 'logo_large.svg']:
        destination = DOCS / 'assets/images' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'images' / name, destination)
    (DOCS / 'Tutorials/Shielding/Shielding.md').unlink()
    for name in ['conf.py', 'requirements.txt']:
        (DOCS / name).unlink()
    project = ROOT / 'pyproject.toml'
    text, count = re.subn(r'^docs = \[.*?^\]', 'docs = [\n    "zensical==0.0.60",\n    "mkdocstrings-python>=1.18,<3",\n    "tomli>=2; python_version < \'3.11\'",\n]', project.read_text(), flags=re.M | re.S)
    if count != 1:
        raise ValueError('Expected one docs dependency group')
    project.write_text(text, encoding='utf-8')
    readme = ROOT / 'README.md'
    text = readme.read_text().replace('uv sync --group docs', 'uv sync --locked --only-group docs')
    text = text.replace('uv run --locked --group docs sphinx-build -W -b html docs docs/_build/html', 'uv run --locked --only-group docs zensical build --strict')
    text = text.replace('uv run --locked --group docs sphinx-autobuild docs docs/_build/html', 'uv run --locked --only-group docs zensical serve')
    readme.write_text(text.replace('docs/_build/html/index.html', 'site/index.html'), encoding='utf-8')
    quick = DOCS / 'Get Started/Quick Start.md'
    quick.write_text(quick.read_text().replace('uv sync --group docs', 'uv sync --locked --only-group docs'), encoding='utf-8')
    ignore = ROOT / '.gitignore'
    text = ignore.read_text()
    for pattern in ['/site/', '/.cache/']:
        if pattern not in text.splitlines():
            text += '\n' + pattern + '\n'
    ignore.write_text(text, encoding='utf-8')
    print(f'Converted {len(original) - len(excluded)} pages; retained {after} API directives; all pages included in navigation.')


if __name__ == '__main__':
    main()
