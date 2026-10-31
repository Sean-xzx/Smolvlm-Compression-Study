// Run with the bundled Node runtime and @oai/artifact-tool available in build/node_modules.
// Usage: node build_talk.mjs /absolute/path/to/autonomous
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
const workspaceDir=path.resolve(process.argv[2]);
const repo=path.join(workspaceDir,'repo');
const out=path.join(workspaceDir,'build/talk-render');
const SKILL_DIR='/home/zx/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const RUNTIME_PYTHON='/home/zx/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
process.env.RUNTIME_NODE_MODULES ??= '/home/zx/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
await fs.mkdir(out,{recursive:true});
const content=JSON.parse(await fs.readFile(path.join(repo,'talk/talk_source.json'),'utf8'));
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
const ink='#172A3A',muted='#526373',accent='#2575B8',font='Noto Sans CJK SC';
function text(slide,txt,x,y,w,h,size=28,color=ink,bold=false){
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=txt;
  shape.text.style={fontSize:size,typeface:font,color,bold,autoFit:'none',insets:0,wrap:'word'};
  return shape;
}
async function picture(slide,name,x,y,w,h){
  const bytes=await fs.readFile(path.join(repo,'figures',name));
  slide.images.add({blob:bytes,contentType:'image/png',alt:name+'，真实来源见figure_manifest',fit:'contain',position:{left:x,top:y,width:w,height:h}});
}
for(let i=0;i<content.length;i++){
  const data=content[i],slide=presentation.slides.add();slide.background.fill='#FFFFFF';
  text(slide,data.title,60,40,1155,68,i===0?48:38,ink,true);
  text(slide,`${i+1} / 12   Sean-xzx`,1030,681,200,25,16,muted);
  slide.speakerNotes.text=`目标${data.minutes}分钟。\n来源：${data.source}\n\n${data.notes}\n\nAI辅助制作，用户审阅/独立演练未记录。`;
  if(i===0){
    text(slide,data.body[0],64,140,1110,54,34,muted);
    text(slide,data.body[1],64,216,1110,48,28,muted);
    text(slide,'3072',64,320,510,96,80,accent,true);
    text(slide,'正式预测  六配置  两集各256题',64,425,1080,55,30);
    text(slide,data.body[3],64,535,1130,74,38,ink,true);
    text(slide,'原权重保留  质量与计算量分别核验',64,620,1000,40,25,muted);
  }else if([2,3].includes(i)){
    await picture(slide,data.figure,60,116,1160,527);
    text(slide,data.body[0],65,647,950,28,22,muted);
  }else if([5,6,7,8].includes(i)){
    await picture(slide,data.figure,48,122,1184,451);
    text(slide,data.body[0],62,585,1150,40,25,ink,true);
    text(slide,data.body[1],62,632,1120,36,24,muted);
  }else if(i===10){
    await picture(slide,data.figure,60,140,730,475);
    text(slide,'baseline',840,165,350,46,30,muted);
    text(slide,'22.',840,215,350,78,64,accent,true);
    text(slide,'Multi',840,335,350,46,30,muted);
    text(slide,'2.',840,385,350,78,64,'#C74747',true);
    text(slide,'得分 1 到 0',840,490,350,50,32,ink,true);
    text(slide,'尚无Attention归因实验，不声称已定位被删文字patch',62,625,1140,42,25,muted);
  }else if(i===4){
    text(slide,'视觉传播',64,140,520,54,31,accent,true);
    text(slide,'postnorm 后零回填\n每块恢复1024格\n\npixel shuffle 4×4\n单Linear 12288到576\n\nText5仍为832视觉位置',64,208,540,407,28);
    text(slide,'LLM FFN Skip',682,140,530,54,31,'#258B7C',true);
    text(slide,'L8 / L16 / L24 同时跳过\n保留Attention与cache\n省norm2 + MLP\n\n7962624个MLP参数停用\n另norm2为1728\n原权重仍全部保存',682,208,540,407,28);
    text(slide,'Linear与pointwise MLP通常保持N，Q-Former/Resampler由query/latent数决定',64,631,1150,40,23,muted);
  }else{
    const y0=i===1?145:155,step=i===1?94:105;
    data.body.forEach((line,j)=>text(slide,line,68,y0+j*step,1140,75,j===0?32:29,j===0?accent:ink,j===0));
    if(i===9)text(slide,'证据各自证明实现、数值或执行路径，不能互相替代',68,612,1120,44,25,muted);
  }
}
for(const [i,slide] of presentation.slides.items.entries()){
  const png=await presentation.export({slide,format:'png',scale:1.5});
  await fs.writeFile(path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
  const layout=await slide.export({format:'layout'});
  await fs.writeFile(path.join(out,`slide-${String(i+1).padStart(2,'0')}.layout.json`),await layout.text());
}
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);
const stagingDir=path.join(workspaceDir,'build/.codex-finalizer');await fs.mkdir(stagingDir,{recursive:true});
const candidatePath=path.join(stagingDir,'candidate.pptx');await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const result=await finalizePresentation({
  explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  workspaceDir,candidatePath,finalPath:path.join(repo,'talk/DEFENSE.zh.pptx'),pythonExecutable:RUNTIME_PYTHON,
  integrityValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,
  receiptPath:path.join(stagingDir,'DEFENSE.validation.json')
});
console.log(JSON.stringify(result));
