"""Validate navigation, native Markdown, and optionally built local URLs.

Run from any directory: python scripts/check_docs.py [--site site].
External links are deliberately not fetched, making CI deterministic and offline.
"""
from __future__ import annotations
import argparse
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def nav_paths(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from nav_paths(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from nav_paths(item)


class Page(HTMLParser):
    def __init__(self, source: str):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.feed(source)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if attrs.get('id'):
            self.ids.add(attrs['id'])
        if tag == 'a' and attrs.get('name'):
            self.ids.add(attrs['name'])
        for attr in ('href', 'src'):
            if attrs.get(attr):
                self.links.append(attrs[attr])


def validate(site: Path | None = None) -> list[str]:
    config = tomllib.loads((ROOT / 'zensical.toml').read_text())['project']
    docs = ROOT / config['docs_dir']
    errors = []
    expected = {p.relative_to(docs).as_posix() for p in docs.rglob('*.md') if p.name != 'README.md' and '_build' not in p.parts}
    linked = list(nav_paths(config['nav']))
    actual = {p for p in linked if not urlsplit(p).scheme}
    for path in sorted(expected - actual):
        errors.append(f'Missing navigation entry: {path}')
    for path in sorted(actual - expected):
        errors.append(f'Navigation target does not exist: {path}')
    if len(actual) != len(linked):
        errors.append('Duplicate or external navigation entry; review the explicit page tree.')
    api_count = 0
    for name in sorted(expected):
        text = (docs / name).read_text()
        api_count += len(re.findall(r'^::: masa\.', text, re.M))
        if re.search(r'^`{3,}\{(?:toctree|eval-rst|figure|list-table|important|seealso)\}', text, re.M):
            errors.append(f'Legacy MyST directive in {name}')
    if site is not None:
        pages = {p.resolve(): Page(p.read_text(encoding='utf-8')) for p in site.rglob('*.html')}
        if not pages or not (site / 'index.html').is_file():
            errors.append('The built site is missing index.html.')
        prefix = urlsplit(config['site_url']).path
        for path, page in pages.items():
            for link in page.links:
                parsed = urlsplit(link)
                if parsed.scheme or parsed.netloc:
                    continue
                target = unquote(parsed.path)
                if target.startswith('/'):
                    if not target.startswith(prefix):
                        errors.append(f'{path.relative_to(site)}: URL outside site prefix: {link}')
                        continue
                    destination = (site / target[len(prefix):]).resolve()
                else:
                    destination = (path.parent / target).resolve() if target else path
                if destination.is_dir():
                    destination /= 'index.html'
                if not destination.is_file():
                    errors.append(f'{path.relative_to(site)}: missing local target {link}')
                elif parsed.fragment and destination in pages:
                    anchor = unquote(parsed.fragment)
                    if anchor not in pages[destination].ids:
                        errors.append(f'{path.relative_to(site)}: missing anchor {link}')
    print(f'Checked {len(expected)} documentation pages and {api_count} API directives' + (' and built local URLs.' if site else '.'))
    return sorted(set(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path)
    args = parser.parse_args()
    errors = validate(args.site.resolve() if args.site else None)
    for error in errors:
        print('ERROR:', error)
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
