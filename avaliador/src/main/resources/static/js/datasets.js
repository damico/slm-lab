(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {labs:[], lab:null, dataset:null, page:1, total:0, items:[], current:null, answer:false, generation:0};
  let searchTimer, listController, detailController;
  const element = (tag,text,cls) => {const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;};
  const format = number => number.toLocaleString('pt-BR');
  const error = message => {$('datasets-error').hidden=!message;$('datasets-error').textContent=message;};
  async function get(path,signal) {
    const response=await fetch('/api/datasets'+path,{signal});
    const data=await response.json();
    if(!response.ok)throw new Error(data.message||data.detail||'Os datasets locais ainda não estão disponíveis. Consulte a documentação do avaliador.');
    return data;
  }
  function datasetParams(extra={}) {
    return new URLSearchParams({dataset:state.dataset,q:$('dataset-search').value,page:String(state.page),size:'20',eligible:$('dataset-eligible').value,...extra});
  }
  function renderLabs() {
    $('dataset-labs').replaceChildren();
    state.labs.forEach(lab=>{
      const button=element('button',undefined,'panel dataset-lab');button.type='button';button.dataset.lab=lab.id;
      button.setAttribute('aria-pressed',String(lab.id===state.lab?.id));
      button.append(element('span','LAB '+lab.id,'eyebrow'),element('strong',lab.title),element('span',lab.dataset,'small muted'),
        element('small',lab.available?`${format(lab.train_count)} exemplos de treino`:'Dados ainda não carregados','dataset-lab-count'));
      button.addEventListener('click',()=>selectLab(lab));$('dataset-labs').append(button);
    });
  }
  function resetDetail() {
    detailController?.abort();state.current=null;state.answer=false;
    $('dataset-detail').hidden=true;$('dataset-empty').hidden=false;
    $('dataset-empty').textContent='Escolha um exemplo na lista para conhecer seus dados.';
  }
  function selectLab(lab) {
    listController?.abort();state.generation++;state.lab=lab;state.page=1;resetDetail();error('');renderLabs();
    $('dataset-workspace').hidden=!lab.available;
    if(!lab.available){error('Os dados deste laboratório ainda não foram carregados nesta instalação. Consulte o roteiro do Lab '+lab.id+' para a preparação.');return;}
    $('dataset-name').textContent='LAB '+lab.id+' / '+lab.dataset;$('dataset-title').textContent=lab.title;
    $('dataset-description').textContent=lab.description;$('dataset-document').href='/docs/'+lab.id;
    $('dataset-search').value='';$('dataset-eligible').value='all';
    $('dataset-version').replaceChildren(...lab.versions.map(version=>{const option=element('option',version.label);option.value=version.id;return option;}));
    selectVersion();history.replaceState(null,'','/datasets?lab='+lab.id);
  }
  function selectVersion() {
    const version=state.lab.versions.find(version=>version.id===$('dataset-version').value);
    const order={train:0,validation:1,test:2,calibration:3,selection:4};
    const splits=[...version.splits].sort((a,b)=>order[a.id]-order[b.id]);
    $('dataset-split').replaceChildren(...splits.map(split=>{const option=element('option',`${split.label} / ${format(split.count)} exemplos`);option.value=split.id;return option;}));
    const original=version.origin==='original';
    $('dataset-provenance').className='dataset-provenance '+(original?'original':'prepared');
    $('dataset-provenance').textContent=original
      ? (state.lab.id==='03'?'Fonte original SROIE: imagem e transcrição anotada. Não é a versão com OCR e divisão de validação preparada pelo Lab 03.':'Fonte original FinQA: tabelas e textos completos. Ainda não passou pela recuperação e pelo limite de contexto do Lab 05.')
      : 'Dados preparados pelo laboratório. Você está vendo os exemplos locais, com sua divisão e configuração registradas.';
    $('dataset-eligible').disabled=original;$('dataset-eligible').value='all';selectSplit();
  }
  function selectSplit() {
    const version=state.lab.versions.find(version=>version.id===$('dataset-version').value);
    state.dataset=version.splits.find(split=>split.id===$('dataset-split').value).dataset_id;state.page=1;state.generation++;
    const helps={train:'Treino: exemplos usados para ensinar o modelo.',validation:'Validação: exemplos usados para escolher configurações, sem atualizar os pesos nesta etapa.',test:'Teste: exemplos reservados para medir o resultado final.',calibration:'Calibração: exemplos usados para ajustar a interpretação da confiança.',selection:'Seleção do limiar: exemplos usados para escolher quando indicar revisão.'};
    $('dataset-split-help').textContent=helps[$('dataset-split').value];resetDetail();loadSamples();
  }
  async function loadSamples(targetOrdinal=null) {
    listController?.abort();listController=new AbortController();const generation=++state.generation;
    ['dataset-prev-page','dataset-next-page','dataset-random'].forEach(id=>$(id).disabled=true);
    $('dataset-count').textContent='Buscando exemplos…';$('dataset-list').setAttribute('aria-busy','true');
    try {
      const data=await get('/samples?'+datasetParams(),listController.signal);if(generation!==state.generation)return;
      state.items=data.items;state.total=data.total;renderList();error('');
      if(state.items.length)openExample(targetOrdinal==='last'?state.items.at(-1).ordinal:state.items.find(item=>item.ordinal===targetOrdinal)?.ordinal??state.items[0].ordinal);else{resetDetail();$('dataset-empty').textContent='Nenhum exemplo corresponde à busca. Tente outra palavra ou remova o filtro.';}
    } catch(failure) {if(failure.name!=='AbortError')error(failure.message);}
    finally {if(generation===state.generation)$('dataset-list').setAttribute('aria-busy','false');}
  }
  function renderList() {
    $('dataset-list').replaceChildren();
    const first=state.total?(state.page-1)*20+1:0,last=Math.min(state.page*20,state.total);
    $('dataset-count').textContent=`${format(first)}–${format(last)} de ${format(state.total)} exemplos`;
    if(!state.items.length)$('dataset-list').append(element('p','Nenhum exemplo encontrado.','dataset-list-empty muted'));
    state.items.forEach(item=>{
      const button=element('button',undefined,'dataset-case');button.type='button';button.dataset.ordinal=String(item.ordinal);
      button.setAttribute('aria-pressed',String(item.ordinal===state.current?.ordinal));
      button.append(element('small',item.identifier,'eyebrow'),element('strong',item.preview||'Exemplo sem texto de prévia'));
      if(item.label)button.append(element('span',item.label,'small muted'));
      button.addEventListener('click',()=>openExample(item.ordinal));$('dataset-list').append(button);
    });
    const pages=Math.max(1,Math.ceil(state.total/20));$('dataset-page').textContent=`${state.page} / ${pages}`;
    $('dataset-prev-page').disabled=state.page<=1;$('dataset-next-page').disabled=state.page>=pages;
    $('dataset-random').disabled=!state.total;
  }
  function code(text) {return element('pre',typeof text==='string'?text:JSON.stringify(text,null,2),'dataset-code');}
  function block(title,content,open=true) {
    const section=element('details',undefined,'dataset-section');section.open=open;
    section.append(element('summary',title),content);return section;
  }
  function table(rows) {
    const wrapper=element('div',undefined,'sql-table-wrap');if(!rows?.length)return wrapper;
    const table=element('table',undefined,'table table-sm'),head=element('thead'),header=element('tr');
    rows[0].forEach(value=>header.append(element('th',String(value||''))));head.append(header);table.append(head);
    const body=element('tbody');rows.slice(1).forEach(row=>{const tr=element('tr');row.forEach(value=>tr.append(element('td',String(value??''))));body.append(tr);});
    table.append(body);wrapper.append(table);return wrapper;
  }
  function renderInput(item) {
    const container=$('dataset-input-content');container.replaceChildren();
    if(item.lab==='01') {
      const tools=element('div',undefined,'dataset-tools');
      if(Array.isArray(item.tools))item.tools.forEach(tool=>{const card=element('details',undefined,'dataset-tool');card.append(element('summary',tool.name||'Ferramenta'),element('p',tool.description||'','small muted'),code(tool.parameters||tool));tools.append(card);});
      else tools.append(code(item.tools));
      container.append(block(`Ferramentas disponíveis (${Array.isArray(item.tools)?item.tools.length:'catálogo'})`,tools));
    } else if(item.lab==='02') {
      const meta=element('p',`Banco: ${item.db_id}${item.metadata.difficulty?' / dificuldade: '+item.metadata.difficulty:''}`,'small muted');container.append(meta);
      if(item.evidence)container.append(block('Pistas para responder',element('p',item.evidence,'dataset-paragraph')));
      container.append(block('Esquema do banco',code(item.schema),false));
    } else if(item.lab==='03') {
      if(item.has_image){const image=element('img');image.src='/api/datasets/image?'+new URLSearchParams({dataset:item.dataset_id,ordinal:item.ordinal});image.alt='Imagem original do recibo '+item.id;image.className='dataset-receipt-image';image.loading='lazy';container.append(image);}
      container.append(element('p',item.text_origin,'small muted'),block('Texto do recibo',element('p',item.receipt_text,'dataset-paragraph')));
    } else if(item.lab==='04') {
      const timeline=element('div',undefined,'dataset-timeline');
      item.history.forEach(event=>{const turn=element('article',undefined,'dataset-turn '+event.role);turn.append(element('small',{customer:'Cliente',agent:'Atendente',action:'Ação observada'}[event.role]||event.role,'eyebrow'),element('p',event.text));if(event.action)turn.append(element('span',event.action,'pill subtle'));timeline.append(turn);});
      container.append(timeline);
      if(item.rules?.length){const rules=element('div');item.rules.forEach(rule=>rules.append(element('p',rule,'dataset-paragraph')));container.append(block(`Regras recuperadas (${item.rules.length})`,rules,false));}
    } else if(item.lab==='05') {
      const sources=Object.entries(item.sources||{});
      if(sources.length){const list=element('div');sources.forEach(([id,text])=>{const source=element('article',undefined,'dataset-source');source.append(element('small',id,'eyebrow'),element('p',text));list.append(source);});container.append(block('Fontes selecionadas para o prompt',list));container.append(block('Consultar tabela de origem',table(item.table),false));}
      else {container.append(block('Tabela do relatório',table(item.table)));const texts=[...(item.pre_text||[]),...(item.post_text||[])];const list=element('div');texts.forEach((text,i)=>{const source=element('article',undefined,'dataset-source');source.append(element('small','text_'+i,'eyebrow'),element('p',text));list.append(source);});container.append(block(`Contexto do relatório (${texts.length} trechos)`,list,false));}
    }
  }
  function renderReference(item) {
    const container=$('dataset-reference-content');container.replaceChildren();
    const titles={'01':'Chamadas e argumentos esperados','02':'Consulta SQL de referência','03':'Campos anotados no recibo','04':'Próxima etapa anotada','05':'Programa e resultado de referência'};
    $('dataset-reference-title').textContent=titles[item.lab];
    if(item.lab==='01' && Array.isArray(item.reference))item.reference.forEach(call=>{const card=element('article',undefined,'dataset-call');card.append(element('strong',call.name),code(call.arguments));container.append(card);});
    else if(item.lab==='03' && item.reference && typeof item.reference==='object') {
      const fields=element('div',undefined,'dataset-fields');const labels={company:'Empresa',date:'Data',address:'Endereço',total:'Total'};
      Object.entries(labels).forEach(([key,label])=>{const card=element('article');card.append(element('small',label,'eyebrow'),element('strong',item.reference[key]??'Não anotado'));fields.append(card);});container.append(fields);
    } else if(item.lab==='04')container.append(element('strong',item.reference,'dataset-class'));
    else if(item.lab==='05') {
      container.append(code(item.reference?.program||item.reference),element('p','Resultado esperado: '+String(item.expected_result),'dataset-financial-answer'));
      if(item.reference?.evidence?.length)container.append(element('p','Fontes anotadas: '+item.reference.evidence.join(', '),'small muted'));
      container.append(element('p','Escala de exe_ans no FinQA. O programa define a conversão; não acrescentamos “%” automaticamente.','small muted'));
      if(item.metadata.supervision==='teacher')container.append(element('span','Supervisão do professor aprovada','pill purple'));
    } else container.append(code(item.reference));
  }
  async function openExample(ordinal) {
    detailController?.abort();detailController=new AbortController();const controller=detailController,dataset=state.dataset;
    $('dataset-empty').hidden=false;$('dataset-empty').textContent='Abrindo exemplo…';$('dataset-detail').hidden=true;
    try {
      const item=await get('/detail?'+new URLSearchParams({dataset,ordinal}),detailController.signal);if(dataset!==state.dataset||controller!==detailController)return;
      state.current=item;state.answer=false;
      $('dataset-empty').hidden=true;$('dataset-detail').hidden=false;$('dataset-question').textContent=item.question;
      $('dataset-example-id').textContent=item.id;
      $('dataset-example-status').textContent=item.origin==='original'?'Fonte original':item.metadata.eligible===false?'Fora do contexto':'Dados do laboratório';
      renderInput(item);renderReference(item);toggleAnswer(false);renderList();
      $('dataset-metadata').replaceChildren(...Object.entries(item.metadata).map(([key,value])=>element('p',`${key}: ${typeof value==='object'?JSON.stringify(value):value}`,'small muted')));
      $('dataset-raw').textContent=JSON.stringify(item.raw,null,2);
      updateCaseNavigation();
    } catch(failure) {if(failure.name!=='AbortError')error(failure.message);}
  }
  function toggleAnswer(show) {state.answer=show;$('dataset-reference').hidden=!show;$('dataset-show-answer').textContent=show?'Ocultar resposta esperada':'Mostrar resposta esperada';$('dataset-show-answer').setAttribute('aria-expanded',String(show));}
  function updateCaseNavigation() {
    const index=state.items.findIndex(item=>item.ordinal===state.current?.ordinal);
    $('dataset-prev-case').disabled=index<=0 && state.page<=1;
    $('dataset-next-case').disabled=index>=state.items.length-1 && state.page>=Math.ceil(state.total/20);
  }
  function nextCase(direction) {
    const index=state.items.findIndex(item=>item.ordinal===state.current?.ordinal),target=state.items[index+direction];
    if(target)openExample(target.ordinal);else if(direction>0 && state.page*20<state.total){state.page++;loadSamples();}else if(direction<0 && state.page>1){state.page--;loadSamples('last');}
  }
  $('dataset-version').addEventListener('change',selectVersion);$('dataset-split').addEventListener('change',selectSplit);
  $('dataset-search').addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.page=1;resetDetail();loadSamples();},350);});
  $('dataset-eligible').addEventListener('change',()=>{state.page=1;resetDetail();loadSamples();});
  $('dataset-prev-page').addEventListener('click',()=>{state.page--;resetDetail();loadSamples();});
  $('dataset-next-page').addEventListener('click',()=>{state.page++;resetDetail();loadSamples();});
  $('dataset-prev-case').addEventListener('click',()=>nextCase(-1));$('dataset-next-case').addEventListener('click',()=>nextCase(1));
  $('dataset-show-answer').addEventListener('click',()=>toggleAnswer(!state.answer));
  $('dataset-random').addEventListener('click',async()=>{try{const data=await get('/samples?'+datasetParams({random:'true'}));if(data.items.length){state.page=Math.floor(data.offset/20)+1;await loadSamples(data.items[0].ordinal);}}catch(failure){error(failure.message);}});
  $('dataset-download').addEventListener('click',()=>{if(!state.current)return;const blob=new Blob([JSON.stringify(state.current.raw,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=element('a');link.href=url;link.download='lab-'+state.current.lab+'-exemplo-'+state.current.ordinal+'.json';link.click();URL.revokeObjectURL(url);});
  (async()=>{try{const data=await get('');state.labs=data.labs;renderLabs();const preferred=new URLSearchParams(location.search).get('lab');selectLab(state.labs.find(lab=>lab.id===preferred&&lab.available)||state.labs.find(lab=>lab.available)||state.labs[0]);}catch(failure){error(failure.message);}finally{$('datasets-loading').hidden=true;}})();
})();
