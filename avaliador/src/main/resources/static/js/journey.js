(() => {
  'use strict';
  const namespace = 'http://www.w3.org/2000/svg';
  const $ = id => document.getElementById(id);
  let current = null;
  let zoom = 1;
  let dragging = null;
  function svgElement(tag, attributes = {}, text) {
    const node = document.createElementNS(namespace, tag);
    Object.entries(attributes).forEach(([name,value]) => node.setAttribute(name,String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function shorten(text, size=43) {
    text = String(text || '').replace(/\s+/g,' ').trim();
    return text.length>size ? text.slice(0,size-1)+'…' : text;
  }
  function selected(source, title, text) {
    $('journey-detail-source').textContent=source;
    $('journey-detail-title').textContent=title;
    $('journey-detail-text').textContent=text;
  }
  function compact(events, source) {
    const indexed=events.map((event,index)=>({event,index,source}));
    if ($('journey-expand').checked || indexed.length<=7) return indexed;
    return [...indexed.slice(0,2),{collapsed:indexed.length-5,source},...indexed.slice(-3)];
  }
  function applyZoom() {
    const svg=$('journey-canvas').querySelector('svg');
    if (!svg) return;
    const box=svg.viewBox.baseVal;
    svg.style.width=`${box.width*zoom}px`;
    svg.style.height=`${box.height*zoom}px`;
    $('journey-zoom-reset').textContent=`${Math.round(zoom*100)}%`;
  }
  function render(history,result,outcome) {
    current={history,result,outcome};
    const box=$('journey-canvas');
    const scroll={left:box.scrollLeft,top:box.scrollTop};
    box.replaceChildren();
    const css=getComputedStyle(document.documentElement);
    const colors={text:css.getPropertyValue('--text').trim(),muted:css.getPropertyValue('--muted').trim(),
      panel:css.getPropertyValue('--panel').trim(),line:css.getPropertyValue('--line').trim(),
      purple:css.getPropertyValue('--purple').trim(),lime:css.getPropertyValue('--lime').trim(),yellow:css.getPropertyValue('--yellow').trim()};
    const observed=compact(history,'Histórico recebido pelo modelo');
    const future=outcome ? compact(outcome.continuation || [],'Continuação real do ABCD') : [];
    const decision=result.slm;
    const center=[
      {title:'Contexto montado',text:`${result.context.input_tokens} tokens do Qwen / ${(result.context.policies||[]).length} regras recuperadas / ${result.context.dropped_turns} eventos retirados da janela.`,kind:'context'},
      {title:decision.choice||'Revisão necessária',text:decision.choice ? `Próximo passo recomendado pelo SLM. Confiança calibrada: ${(decision.confidence*100).toFixed(1)}%. Esta ação não foi executada pelo avaliador.` : 'O contexto ultrapassou a janela do modelo.',kind:'prediction'},
      {title:decision.requires_review?'Encaminhar para revisão':'Acima do limiar',text:`Limiar escolhido: ${Number.isFinite(decision.threshold)?(decision.threshold*100).toFixed(1)+'%':'indisponível'}. A revisão é indicada pela regra aplicada à confiança.`,kind:'review'}
    ];
    if(result.comparison)center.push({title:'Comparação OpenAI',text:`${result.comparison.model||'API'}: ${result.comparison.choice||result.comparison.message||'sem resposta'}`,kind:'comparison'});
    const height=250+Math.max(observed.length,future.length,center.length,1)*96;
    const svg=svgElement('svg',{viewBox:`0 0 1040 ${height}`,role:'group','aria-label':'Evolução do atendimento: histórico observado, recomendação do SLM e continuação registrada no ABCD.'});
    const defs=svgElement('defs');
    const marker=svgElement('marker',{id:'journey-arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'});
    marker.append(svgElement('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:colors.muted}));defs.append(marker);svg.append(defs);
    const connect=(x1,y1,x2,y2,dashed=false)=>svg.append(svgElement('path',{d:`M${x1},${y1} C${x1},${(y1+y2)/2} ${x2},${(y1+y2)/2} ${x2},${y2}`,fill:'none',stroke:colors.muted,'stroke-opacity':'.5','stroke-width':'1.2','stroke-dasharray':dashed?'5 5':'none','marker-end':'url(#journey-arrow)'}));
    function node(x,y,title,detail,source,color,subtitle,callback,dashed=false) {
      const group=svgElement('g',{class:'journey-node','data-node':'true',role:'button',tabindex:0,'aria-label':source+': '+title});
      group.append(svgElement('title',{},title+'\n'+detail));
      group.append(svgElement('rect',{x,y,width:300,height:74,rx:12,fill:colors.panel,stroke:color,'stroke-opacity':'.7','stroke-width':1,'stroke-dasharray':dashed?'5 4':'none'}));
      group.append(svgElement('circle',{cx:x+18,cy:y+21,r:3,fill:color}));
      group.append(svgElement('text',{x:x+31,y:y+25,fill:color,'font-size':11,'font-family':'Segoe UI, Arial, sans-serif'},shorten(title,39)));
      group.append(svgElement('text',{x:x+17,y:y+51,fill:colors.muted,'font-size':10,'font-family':'Segoe UI, Arial, sans-serif'},shorten(subtitle||detail,46)));
      const open=()=>{selected(source,title,detail);if(callback)callback();};
      group.addEventListener('click',open);
      group.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();open();}});
      svg.append(group);
    }
    node(370,20,outcome ? `Atendimento ${outcome.id.split(':')[0]}` : 'Atendimento em teste',
      'O histórico observado é a entrada do modelo. O próximo passo é uma recomendação. A continuação ABCD, quando presente, é mostrada como referência depois da inferência.',
      'Mapa do atendimento',colors.purple,'Clique em qualquer etapa para inspecionar');
    const headers=[['Histórico observado',`${history.length} eventos anteriores`,20],['Próximo passo / SLM','Recomendação desta inferência',370],['Continuação / ABCD',outcome?'Eventos registrados depois do ponto de teste':'Sem continuação registrada',720]];
    headers.forEach(([title,detail,x],index)=>{connect(520,94,x+150,126,index===1);node(x,126,title,detail,'Fase do atendimento',colors.muted,detail);});
    function column(items,x,fallback) {
      if(!items.length) {connect(x+150,200,x+150,222);node(x,222,'Sem registro posterior',fallback,'Informação do atendimento',colors.muted,fallback);return;}
      items.forEach((item,index)=>{
        const y=222+index*96;connect(x+150,index===0?200:y-22,x+150,y);
        if(item.collapsed){node(x,y,`+ ${item.collapsed} eventos`,`${item.collapsed} etapas foram recolhidas para facilitar a leitura. Clique para mostrar todos os eventos.`,item.source,colors.muted,'Clique para expandir a sequência',()=>{$('journey-expand').checked=true;render(current.history,current.result,current.outcome);});return;}
        const event=item.event;
        const role={customer:'Cliente',agent:'Atendente',action:'Ação executada'}[event.role]||event.role;
        const color=event.role==='customer'?colors.lime:event.role==='action'?colors.yellow:colors.purple;
        const stage=event.id?Number(event.id.split(':').pop())+1:item.index+1;
        const title=`${String(stage).padStart(2,'0')} / ${event.action||role}`;
        node(x,y,title,event.text,item.source+(event.id?' / '+event.id:''),color,event.text);
      });
    }
    column(observed,20,'Nenhum evento observado.');
    center.forEach((item,index)=>{const y=222+index*96;connect(520,index===0?200:y-22,520,y,true);node(370,y,item.title,item.text,'Resultado do teste',colors.purple,item.text,null,true);});
    column(future,720,'Selecione um exemplo ABCD com desfecho registrado para acompanhar a resposta entregue.');
    box.append(svg);applyZoom();box.scrollLeft=scroll.left;box.scrollTop=scroll.top;
  }
  function initialize() {
    if(!$('journey-canvas'))return;
    $('journey-expand').addEventListener('change',()=>{if(current)render(current.history,current.result,current.outcome);});
    $('journey-zoom-in').addEventListener('click',()=>{zoom=Math.min(1.6,zoom+.15);applyZoom();});
    $('journey-zoom-out').addEventListener('click',()=>{zoom=Math.max(.4,zoom-.15);applyZoom();});
    $('journey-zoom-reset').addEventListener('click',()=>{zoom=1;applyZoom();});
    window.addEventListener('slm-theme-change',()=>{if(current)render(current.history,current.result,current.outcome);});
    const canvas=$('journey-canvas');
    canvas.addEventListener('pointerdown',event=>{if(event.pointerType!=='mouse'||event.target.closest('[data-node]'))return;dragging={x:event.clientX,y:event.clientY,left:canvas.scrollLeft,top:canvas.scrollTop};canvas.setPointerCapture(event.pointerId);canvas.classList.add('dragging');});
    canvas.addEventListener('pointermove',event=>{if(!dragging)return;canvas.scrollLeft=dragging.left+dragging.x-event.clientX;canvas.scrollTop=dragging.top+dragging.y-event.clientY;});
    const stop=()=>{dragging=null;canvas.classList.remove('dragging');};
    canvas.addEventListener('pointerup',stop);canvas.addEventListener('pointercancel',stop);
  }
  window.LabJourney={render};initialize();
})();
