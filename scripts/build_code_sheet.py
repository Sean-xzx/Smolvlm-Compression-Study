from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Table,TableStyle,Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
import re
R=Path(__file__).resolve().parents[1];p=R/'talk/CODE_MAP.zh.pdf'
pdfmetrics.registerFont(TTFont('CJK','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
from xml.sax.saxutils import escape
def mixed(text):
 return '<br/>'.join(''.join('<font name="Latin">'+escape(x)+'</font>' if re.match(r'[^\u3400-\u9fff\u3000-\u303f\uff00-\uffef]',x) else escape(x) for x in re.findall(r'[^\u3400-\u9fff\u3000-\u303f\uff00-\uffef]+|[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]+',part)) for part in text.split('<br/>'))
style=ParagraphStyle('body',fontName='CJK',fontSize=8.3,leading=12)
rows=[['用途','实际文件和函数','行号']]
for line in (R/'talk/CODE_MAP.zh.md').read_text().splitlines():
 if not line.startswith('|') or line.startswith('|---') or line.startswith('|用途'):continue
 a,b,c=line.strip('|').split('|');m=re.match(r'\[(.*?)\]\([^)]*\) `(.*?)`',b)
 rows.append([a.replace('，省','<br/>省'),m[1]+'<br/>'+m[2],c])
data=[[Paragraph(mixed(x),style) for x in row] for row in rows]
t=Table(data,colWidths=[155,325,50]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7)]))
c=Canvas(str(p),pagesize=A4);c.setTitle('SmolVLM代码定位一页速查');c.setAuthor('Sean-xzx');title=Paragraph(mixed('SmolVLM 代码定位一页速查'),ParagraphStyle('title',fontName='CJK',fontSize=17,leading=22));title.wrap(530,30);title.drawOn(c,32,792)
subtitle=Paragraph(mixed('行号从实际源文件读取。完整证据及命令见离线文档。现场指定环节不用AI。'),style);subtitle.wrap(530,25);subtitle.drawOn(c,32,769)
w,h=t.wrap(530,680);assert h<680;t.drawOn(c,32,750-h);footer=Paragraph(mixed('科学文件与v3相同；新入口使用显式artifact-root。个人独立操作尚未验收。'),style);footer.wrap(530,25);footer.drawOn(c,32,65);c.showPage();c.save();print(p)
