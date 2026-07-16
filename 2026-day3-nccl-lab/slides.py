#!/usr/bin/env python3
"""Zero-dependency slide-deck generator: Markdown -> self-contained HTML slides.
Two modes:
  * default: split slides on a line containing only '---'  (hand-authored decks)
  * --by-heading: start a new slide at every #/##/### heading (turn a doc into slides,
    keeping ALL content; dense slides scroll)
Arrow keys / space / click to move. Dark/Light toggle button (top-right). Prints to PDF.
Usage: python3 slides.py IN.md "Deck Title" [--by-heading] > OUT.html
"""
import sys, re, html

def inline(t):
    t = html.escape(t)
    t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
    t = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', t)
    return t

def render_block(md):
    out, i, lines = [], 0, md.split('\n')
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('```'):
            i += 1; buf = []
            while i < len(lines) and not lines[i].startswith('```'):
                buf.append(html.escape(lines[i])); i += 1
            i += 1; out.append('<pre><code>' + '\n'.join(buf) + '</code></pre>'); continue
        if ln.strip().startswith('|') and i+1 < len(lines) and re.match(r'^\s*\|?[-:| ]+\|', lines[i+1]):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')]); i += 1
            head, body = rows[0], rows[2:]
            t = '<table><thead><tr>' + ''.join(f'<th>{inline(c)}</th>' for c in head) + '</tr></thead><tbody>'
            for r in body: t += '<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>'
            out.append(t + '</tbody></table>'); continue
        m = re.match(r'^(#{1,4})\s+(.*)', ln)
        if m: out.append(f'<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>'); i += 1; continue
        if ln.startswith('>'):
            buf = []
            while i < len(lines) and lines[i].startswith('>'):
                buf.append(inline(lines[i].lstrip('> ').rstrip())); i += 1
            out.append('<blockquote>' + '<br>'.join(buf) + '</blockquote>'); continue
        if re.match(r'^\s*[-*]\s+', ln):
            buf = []
            while i < len(lines) and re.match(r'^\s*[-*]\s+', lines[i]):
                buf.append('<li>' + inline(re.sub(r'^\s*[-*]\s+', '', lines[i])) + '</li>'); i += 1
            out.append('<ul>' + ''.join(buf) + '</ul>'); continue
        if ln.strip() == '': out.append(''); i += 1; continue
        out.append('<p>' + inline(ln) + '</p>'); i += 1
    return '\n'.join(out)

CSS = """
*{box-sizing:border-box} html,body{margin:0;height:100%;background:#0b1020;color:#e8eef7;
font:28px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;overflow:hidden}
#deck{height:100vh;width:100vw}
.slide{display:none;height:100vh;width:100vw;padding:4vh 6vw;flex-direction:column;justify-content:center;overflow:auto}
.slide.on{display:flex}
.slide.title{align-items:center;text-align:center}
h1{font-size:2.0em;color:#5ee6a8;margin:.2em 0;line-height:1.1}
h2{font-size:1.5em;color:#5ee6a8;margin:.1em 0 .4em;border-bottom:2px solid #24406a;padding-bottom:.2em}
h3{font-size:1.1em;color:#7fd1ff;margin:.3em 0}
p,li{font-size:1.0em;margin:.3em 0}
strong{color:#ffd54a}
code{background:#16233d;color:#9fe0ff;padding:.05em .3em;border-radius:4px;font-family:ui-monospace,Consolas,monospace;font-size:.9em}
pre{background:#050a14;border:1px solid #24406a;border-radius:10px;padding:.8rem 1rem;overflow:auto;font-size:.62em;line-height:1.35}
pre code{background:none;color:#c9e6ff;padding:0}
table{border-collapse:collapse;margin:.5em 0;font-size:.85em}
th,td{border:1px solid #24406a;padding:.35rem .7rem} th{background:#16233d;color:#7fd1ff}
blockquote{background:#132a1e;border-left:6px solid #5ee6a8;padding:.5rem 1rem;border-radius:6px;font-size:.9em;color:#cdeede}
ul{margin:.2em 0 .2em 1em}
#bar{position:fixed;bottom:0;left:0;height:5px;background:#5ee6a8;transition:width .2s}
#num{position:fixed;bottom:12px;right:20px;font-size:16px;color:#5a7099}
.hint{position:fixed;bottom:12px;left:20px;font-size:15px;color:#3f5478}
#theme{position:fixed;top:12px;right:16px;z-index:10;background:#16233d;color:#e8eef7;border:1px solid #24406a;
border-radius:20px;padding:.3em .8em;font-size:15px;cursor:pointer}
/* dense decks (--by-heading): smaller text so more fits per slide */
body.dense{font-size:24px} body.dense .slide{justify-content:flex-start;padding-top:3vh}
body.dense pre{font-size:.66em}
/* ---- LIGHT THEME ---- */
body.light{background:#f6f8fc;color:#1a2333}
body.light h1,body.light h2{color:#0a7a4f;border-color:#cfe0d6}
body.light h3{color:#0b6bb0}
body.light strong{color:#b06a00}
body.light code{background:#e7edf6;color:#0b5a94}
body.light pre{background:#0e1626;border-color:#c3d0e2}   /* keep logs dark for contrast */
body.light th{background:#e7f2ec;color:#0b6bb0} body.light td,body.light th{border-color:#cfdae8}
body.light blockquote{background:#eefaf2;color:#215c3e;border-color:#3fae7a}
body.light #num{color:#8091a8} body.light .hint{color:#a9b6c8}
body.light #theme{background:#e7edf6;color:#1a2333;border-color:#c3d0e2}
@media print{html,body{overflow:visible;height:auto} .slide{display:flex!important;page-break-after:always;height:100vh;border-bottom:1px solid #ccc} #bar,#num,.hint,#theme{display:none}}
"""

JS = """
const S=[...document.querySelectorAll('.slide')];let c=0;
function show(n){c=Math.max(0,Math.min(S.length-1,n));S.forEach((s,i)=>s.classList.toggle('on',i===c));
document.getElementById('bar').style.width=((c+1)/S.length*100)+'%';
document.getElementById('num').textContent=(c+1)+' / '+S.length;location.hash=c+1;}
document.addEventListener('keydown',e=>{if(['ArrowRight','ArrowDown','PageDown',' '].includes(e.key)){show(c+1);e.preventDefault();}
else if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key)){show(c-1);e.preventDefault();}
else if(e.key==='Home')show(0);else if(e.key==='End')show(S.length-1);});
document.getElementById('deck').addEventListener('click',e=>{if(e.clientX>window.innerWidth*0.35)show(c+1);else show(c-1);});
// theme toggle (remembers your choice)
const tb=document.getElementById('theme');
function setTheme(l){document.body.classList.toggle('light',l);tb.textContent=l?'◑ Dark':'◐ Light';try{localStorage.setItem('deckLight',l?'1':'0')}catch(e){}}
tb.addEventListener('click',()=>setTheme(!document.body.classList.contains('light')));
setTheme((()=>{try{return localStorage.getItem('deckLight')==='1'}catch(e){return false}})());
show((parseInt(location.hash.slice(1))||1)-1);
"""

def split_by_heading(md):
    """Start a new slide at every #/##/### heading. Keeps ALL content."""
    parts, cur = [], []
    for ln in md.split('\n'):
        if re.match(r'^#{1,3}\s', ln) and any(x.strip() for x in cur):
            parts.append('\n'.join(cur)); cur = [ln]
        else:
            cur.append(ln)
    if cur: parts.append('\n'.join(cur))
    return parts

def main():
    md = open(sys.argv[1], encoding='utf-8').read()
    title = sys.argv[2] if len(sys.argv) > 2 else 'Slides'
    by_heading = '--by-heading' in sys.argv[3:]
    slides = split_by_heading(md) if by_heading else re.split(r'(?m)^---\s*$', md)
    body = []
    for s in slides:
        s = s.strip('\n')
        if not s.strip(): continue
        cls = 'slide title' if re.match(r'^#\s', s.lstrip()) else 'slide'
        body.append(f'<section class="{cls}">{render_block(s)}</section>')
    bodyclass = 'dense' if by_heading else ''
    print(f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head><body class="{bodyclass}">')
    print(f'<button id="theme">◐ Light</button>')
    print(f'<div id="deck">{"".join(body)}</div>')
    print('<div id="bar"></div><div id="num"></div><div class="hint">← → move · click sides · button = theme · Ctrl-P = PDF</div>')
    print(f'<script>{JS}</script></body></html>')

if __name__ == '__main__':
    main()
