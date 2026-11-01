"""Export the checked-in Markdown mother manuscript to editable Word.

No model execution, scientific writes or separate hard-coded report text. PDF is
rendered from this DOCX with the documented render_docx.py workflow. The optional
artifact root is accepted for compatibility; it is never written or imported.
"""
from pathlib import Path
import argparse, re, unicodedata
from urllib.parse import urlsplit
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT

R=Path(__file__).resolve().parents[1];O=R/'report'
BASE='https://github.com/Sean-xzx/Smolvlm-Compression-Study/blob/main/'

def slug(text):
    return ''.join(c for c in text.lower() if c in '-_ ' or unicodedata.category(c)[0] in 'LN').replace(' ','-')

def render_document():
    """Use one manuscript, native text/tables, and portable document hyperlinks."""
    manuscript=(O/'REPORT.zh.md').read_text();doc=Document();sec=doc.sections[0]
    sec.page_height=Cm(29.7);sec.page_width=Cm(21)
    sec.top_margin=Cm(1.7);sec.bottom_margin=Cm(1.6);sec.left_margin=Cm(1.8);sec.right_margin=Cm(1.8)
    normal=doc.styles['Normal'];normal.font.name='Noto Serif CJK SC';normal.font.size=Pt(10.5)
    normal._element.rPr.rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC')
    normal.paragraph_format.line_spacing=1.13;normal.paragraph_format.space_after=Pt(6)
    for name,size in [('Title',23),('Heading 1',16),('Heading 2',12)]:
        st=doc.styles[name];st.font.name='Noto Sans CJK SC';st._element.rPr.rFonts.set(qn('w:eastAsia'),'Noto Sans CJK SC')
        st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
        st.paragraph_format.space_before=Pt(10);st.paragraph_format.space_after=Pt(6)
    for el in list(doc.styles.element.iter(qn('w:pBdr'))):el.getparent().remove(el)
    doc.core_properties.author='Sean-xzx';doc.core_properties.title='SmolVLM无需训练压缩实验报告'
    doc.core_properties.comments='professor-report-v3; AI-authored revision. Human authorship, review and mastery not asserted.'
    headings=[re.sub(r'^#{1,3} ','',x) for x in manuscript.splitlines() if re.match(r'^#{1,3} ',x)]
    bookmarks={slug(h):f'section_{i}' for i,h in enumerate(headings)}
    def words(text):
        text=re.sub(r'`([^`]+)`',r'\1',text);text=re.sub(r'\*\*([^*]+)\*\*',r'\1',text)
        return re.sub(r'[A-Za-z0-9_./:@-]{27,}',lambda m:'\u200b'.join(m[0][i:i+24] for i in range(0,len(m[0]),24)),text)
    def url(target):
        if urlsplit(target).scheme:return target
        file,sep,anchor=target.partition('#');path=(O/file).resolve().relative_to(R).as_posix()
        return BASE+path+('#'+anchor if sep else '')
    def add_runs(par,text):
        pos=0
        for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)',text):
            if m.start()>pos:par.add_run(words(text[pos:m.start()]))
            link=OxmlElement('w:hyperlink')
            if m[2].startswith('#'):
                assert m[2][1:] in bookmarks,m[2];link.set(qn('w:anchor'),bookmarks[m[2][1:]])
            else:
                rid=par.part.relate_to(url(m[2]),'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True);link.set(qn('r:id'),rid)
            run=OxmlElement('w:r');prop=OxmlElement('w:rPr');col=OxmlElement('w:color');col.set(qn('w:val'),'246998');prop.append(col);run.append(prop)
            t=OxmlElement('w:t');t.text=words(m[1]);run.append(t);link.append(run);par._p.append(link);pos=m.end()
        if pos<len(text):par.add_run(words(text[pos:]))
    def para(text,style=None):
        pp=doc.add_paragraph(style=style);add_runs(pp,text);return pp
    def make_table(lines):
        rows=[[s.strip() for s in line.strip('|').split('|')] for line in lines if not re.match(r'^\|[-:| ]+\|$',line)]
        h=rows[0];n=len(h);widths={2:[4.1,13.3],3:[3.6,6.9,6.9],4:[3.0,5.2,5.2,4.0],5:[2.6,3.5,3.8,4.2,3.3],6:[2.6,3.0,3.0,2.4,2.6,3.8],7:[2.8]+[2.43]*6}[n]
        if h[0]=='配置' and 'Encoder比' in h:widths=[2.0,4.3,2.1,4.4,4.6]
        if h[0]=='层':widths=[.8]+[3.32]*5
        if h[0]=='配置/层/模块':widths=[3.6,3.1,6.7,2.,2.]
        if h[0]=='层与算子':widths=[2.4,11.,2.,2.]
        if h[0]=='目标层':widths=[1.6,2.4,4.3,4.,5.1]
        if h[0]=='数据/配置':widths=[2.7,2.6,2.8,2.6,2.6,4.1]
        if h[0]=='配置' and h[1]=='Text %':widths=[2.2,3.,2.6,3.2,2.6,3.8]
        t=doc.add_table(rows=0,cols=n);t.autofit=False
        for col,width in zip(t.columns,widths):col.width=Cm(width)
        pr=t._tbl.tblPr;borders=OxmlElement('w:tblBorders')
        for edge in ['top','left','bottom','right','insideH','insideV']:
            e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
        pr.append(borders)
        for i,row in enumerate(rows):
            cells=t.add_row().cells
            for j,(cell,text,width) in enumerate(zip(cells,row,widths)):
                cell.width=Cm(width);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tc=cell._tc.get_or_add_tcPr();mar=OxmlElement('w:tcMar')
                for side,value in [('top',40 if h[0]=='层' else 60),('bottom',40 if h[0]=='层' else 60),('left',70),('right',70)]:
                    e=OxmlElement('w:'+side);e.set(qn('w:w'),str(value));e.set(qn('w:type'),'dxa');mar.append(e)
                tc.append(mar);pp=cell.paragraphs[0];pp.paragraph_format.space_after=Pt(1);pp.paragraph_format.space_before=Pt(1);pp.paragraph_format.line_spacing=1.08
                if re.match(r'^[+−\-\d\[]',text) and len(text)<40:pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
                add_runs(pp,text)
                for rr in pp.runs:rr.font.size=Pt(9.2)
                if i==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'EAF0F5');tc.append(shade)
                    for rr in pp.runs:rr.bold=True
            if len(rows)<=7:
                for cell in cells:
                    for pp in cell.paragraphs:pp.paragraph_format.keep_with_next = i < len(rows)-1
            tr=t.rows[i]._tr.get_or_add_trPr();tr.append(OxmlElement('w:cantSplit'))
            if i==0:tr.append(OxmlElement('w:tblHeader'))
        gap=doc.add_paragraph();gap.paragraph_format.space_after=Pt(2);gap.paragraph_format.space_before=Pt(0);gap.paragraph_format.line_spacing=Pt(2)
    lines=manuscript.splitlines();i=0;headnum=0
    while i<len(lines):
        line=lines[i].strip()
        if not line or line.startswith('<!--'):i+=1;continue
        if line.startswith('```'):
            i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                pp=para(lines[i]);pp.paragraph_format.space_after=Pt(2)
                for rr in pp.runs:rr.font.name='DejaVu Sans Mono';rr.font.size=Pt(9)
                i+=1
            i+=1;continue
        if line.startswith('|'):
            tab=[]
            while i<len(lines) and lines[i].startswith('|'):tab.append(lines[i]);i+=1
            make_table(tab);continue
        if line.startswith('!['):
            m=re.match(r'!\[(.*?)\]\((.*?)\)',line);ip=(O/m[2]).resolve();pp=doc.add_paragraph();pp.alignment=WD_ALIGN_PARAGRAPH.CENTER
            picture=pp.add_run().add_picture(str(ip),width=Cm(16.8));picture._inline.docPr.set('descr',m[1])
            pp.paragraph_format.keep_with_next=True;pp.paragraph_format.space_after=Pt(3)
        elif line.startswith('#'):
            h=re.sub(r'^#{1,3} ','',line);style='Title' if line.startswith('# ') else 'Heading 1' if line.startswith('## ') else 'Heading 2';pp=para(h,style)
            start=OxmlElement('w:bookmarkStart');start.set(qn('w:id'),str(headnum));start.set(qn('w:name'),bookmarks[slug(h)]);pp._p.insert(0,start)
            end=OxmlElement('w:bookmarkEnd');end.set(qn('w:id'),str(headnum));pp._p.append(end);headnum+=1
        else:
            pp=para(line)
            if line.endswith('：') and len(line)<30:pp.paragraph_format.keep_with_next=True
            if re.match(r'^图\d',line):
                for rr in pp.runs:rr.font.size=Pt(9.2)
        i+=1
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER;footer.add_run('Sean-xzx · ').font.size=Pt(8)
    fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
    doc.save(O/'REPORT.zh.docx');print('DOCX from committed Markdown:',headnum,'headings; no forced section pagination')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact-root',type=Path,help='Compatibility only; science is read-only');p.parse_args();render_document()
