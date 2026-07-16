#!/usr/bin/env python3
"""Zero-dependency Markdown -> self-contained styled HTML (good enough for the lab handout).
Handles: # headings, ``` fenced code, | pipe tables, **bold**, `code`, > quotes, ---, lists, paras.
Usage: python3 render.py worksheet/LAB.md "Day 3 — NCCL Lab" > worksheet/LAB.html
"""
import sys, re, html

def inline(t):
    t = html.escape(t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', t)
    return t

def render(md):
    out, i, lines = [], 0, md.split('\n')
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):                       # code fence
            i += 1; buf = []
            while i < len(lines) and not lines[i].startswith('```'):
                buf.append(html.escape(lines[i])); i += 1
            i += 1; out.append('<pre><code>' + '\n'.join(buf) + '</code></pre>'); continue
        if ln.strip().startswith('|') and i+1 < len(lines) and re.match(r'^\s*\|?[-:| ]+\|', lines[i+1]):
            rows = []                                    # table
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')]); i += 1
            head, body = rows[0], rows[2:]
            t = '<table><thead><tr>' + ''.join(f'<th>{inline(c)}</th>' for c in head) + '</tr></thead><tbody>'
            for r in body: t += '<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>'
            out.append(t + '</tbody></table>'); continue
        m = re.match(r'^(#{1,4})\s+(.*)', ln)
        if m: out.append(f'<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>'); i += 1; continue
        if ln.strip() == '---': out.append('<hr>'); i += 1; continue
        if ln.startswith('>'):                           # blockquote (merge consecutive)
            buf = []
            while i < len(lines) and lines[i].startswith('>'):
                buf.append(inline(lines[i].lstrip('> ').rstrip())); i += 1
            out.append('<blockquote>' + '<br>'.join(buf) + '</blockquote>'); continue
        if re.match(r'^\s*[-*]\s+', ln) or re.match(r'^\s*\d+\.\s+', ln):   # list
            ordered = bool(re.match(r'^\s*\d+\.', ln)); tag = 'ol' if ordered else 'ul'; buf = []
            while i < len(lines) and (re.match(r'^\s*[-*]\s+', lines[i]) or re.match(r'^\s*\d+\.\s+', lines[i])):
                buf.append('<li>' + inline(re.sub(r'^\s*(?:[-*]|\d+\.)\s+', '', lines[i])) + '</li>'); i += 1
            out.append(f'<{tag}>' + ''.join(buf) + f'</{tag}>'); continue
        if ln.strip() == '': out.append(''); i += 1; continue
        out.append('<p>' + inline(ln) + '</p>'); i += 1
    return '\n'.join(out)

CSS = """
body{font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:820px;margin:2rem auto;padding:0 1.2rem;color:#1a1a2e}
h1{color:#0b5;border-bottom:3px solid #0b5;padding-bottom:.3rem}
h2{color:#087;margin-top:2rem;border-bottom:1px solid #cde}
h3{color:#0a6}
code{background:#f4f6f8;padding:.1em .35em;border-radius:4px;font-family:ui-monospace,Consolas,monospace;font-size:.92em}
pre{background:#0f1720;color:#d6e2f0;padding:.9rem 1rem;border-radius:8px;overflow-x:auto}
pre code{background:none;color:inherit;padding:0}
table{border-collapse:collapse;margin:1rem 0;width:100%}
th,td{border:1px solid #cde;padding:.45rem .7rem;text-align:left}
th{background:#eaf7f1}
blockquote{background:#fff8e1;border-left:5px solid #f5b301;margin:1rem 0;padding:.6rem 1rem;border-radius:4px}
hr{border:none;border-top:1px solid #dde;margin:2rem 0}
a{color:#087}
"""

def main():
    md = open(sys.argv[1], encoding='utf-8').read()
    title = sys.argv[2] if len(sys.argv) > 2 else 'Lab'
    print(f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head><body>')
    print(render(md))
    print('</body></html>')

if __name__ == '__main__':
    main()
