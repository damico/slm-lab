(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  if (!$('decision-lab')) return;
  const storeKey = 'slm-lab04-tests-v1';
  const state = {
    history:[{role:'agent',text:'How can I help you today?'},{role:'customer',text:'I would like to return a shirt because it is the wrong size.'}],
    catalog:null, examples:[], expected:null, outcome:null, origin:'Caso livre / exemplo editável',
    result:null, busy:false, records:[], chart:null, loaded:false
  };
  try { const records = JSON.parse(localStorage.getItem(storeKey) || '[]'); if (Array.isArray(records)) state.records = records.slice(0,20); } catch (_) {}
  const percent = n => Number.isFinite(n) ? (n*100).toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1})+'%' : 'Indisponível';
  const seconds = n => Number.isFinite(n) ? n.toLocaleString('pt-BR',{maximumFractionDigits:3})+' s' : '';
  function element(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }
  function error(message) { $('request-error').textContent = message; $('request-error').hidden = !message; }
  function clearResult() {
    state.result = null;
    $('result-empty').hidden = false;
    $('result-filled').hidden = true;
    $('probabilities-panel').hidden = true;
    $('comparison-panel').hidden = true;
    $('explanation-panel').hidden = true;
    $('outcome-panel').hidden = true;
    $('journey-panel').hidden = true;
    $('result-state').textContent = 'Aguardando teste';
    $('result-state').className = 'pill subtle';
  }
  function edited() {
    state.expected = null;
    state.outcome = null;
    $('example-select').value = '';
    state.origin = 'Caso livre / histórico editado';
    $('example-origin').textContent = state.origin;
    clearResult();
  }
  function renderTimeline() {
    $('timeline').replaceChildren();
    state.history.forEach((turn,index) => {
      const row = element('article','timeline-event'); row.dataset.role = turn.role;
      row.append(element('span','event-marker'));
      const heading = element('div','event-heading');
      const role = element('select','form-select event-role');
      role.setAttribute('aria-label',`Papel do evento ${index+1}`);
      [['customer','Cliente'],['agent','Atendente'],['action','Ação executada']].forEach(([value,label]) => {
        const option = element('option','',label); option.value = value; role.append(option);
      });
      role.value = turn.role;
      role.addEventListener('change', () => {
        turn.role = role.value;
        if (turn.role !== 'action') delete turn.action;
        else turn.action = state.catalog?.actions?.[0] || '';
        edited(); renderTimeline();
      });
      const remove = element('button','remove-event','×'); remove.type = 'button';
      remove.setAttribute('aria-label',`Remover evento ${index+1}`);
      remove.addEventListener('click', () => {state.history.splice(index,1);edited();renderTimeline();});
      heading.append(role,element('span','',String(index+1).padStart(2,'0')),remove);
      row.append(heading);
      const text = element('textarea','form-control event-text');
      text.value = turn.text; text.rows = 2; text.maxLength = 16000;
      text.setAttribute('aria-label',`Mensagem ou resultado do evento ${index+1}`);
      text.placeholder = turn.role === 'action' ? 'Resultado de uma ação já executada' : 'Mensagem observada no atendimento';
      text.addEventListener('input', () => {turn.text = text.value; edited();});
      row.append(text);
      if (turn.role === 'action') {
        const wrapper = element('div','action-selector');
        const label = element('label','', 'Ação já executada');
        const select = element('select','form-select'); select.id = `action-${index}`; label.htmlFor = select.id;
        const values = state.catalog?.actions || (turn.action ? [turn.action] : []);
        values.forEach(action => {const option=element('option','',action);option.value=action;select.append(option);});
        select.value = turn.action || ''; select.disabled = !state.catalog;
        select.addEventListener('change', () => {turn.action = select.value;edited();});
        wrapper.append(label,select);row.append(wrapper);
      }
      $('timeline').append(row);
    });
    $('turn-count').textContent = `${state.history.length} eventos`;
    $('example-origin').textContent = state.origin;
  }
  function setBusy(value) {
    state.busy = value;
    $('decision-lab').classList.toggle('busy',value);
    document.querySelectorAll('#timeline input,#timeline textarea,#timeline select,#timeline button').forEach(el=>el.disabled=value);
    ['add-event','import-json'].forEach(id=>$(id).disabled=value);
    $('load-example').disabled = value || !state.examples.length;
    $('example-select').disabled = value || !state.examples.length;
    $('compare-openai').disabled = value || !window.labStatus?.comparison_enabled;
    $('decide').disabled = value || !window.labStatus?.available;
    $('decide').lastChild.textContent = value ? ' Executando teste...' : ' Decidir próximo passo';
  }
  async function getJson(path) {
    const response = await fetch(path,{signal:AbortSignal.timeout(10000)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || 'Não foi possível consultar a API.');
    return data;
  }
  async function loadData() {
    if (state.loaded) return;
    state.loaded = true;
    try {
      const [catalog,examples] = await Promise.all([getJson('/api/lab04/catalog'),getJson('/api/lab04/examples')]);
      state.catalog = catalog;
      state.examples = examples.examples || [];
      const blank=element('option','','Selecionar caso ABCD');blank.value='';
      $('example-select').replaceChildren(blank);
      state.examples.forEach(example=>{const option=element('option','',`${example.name} / ${example.id||'ABCD'}`);option.value=example.name;$('example-select').append(option);});
      renderTimeline();setBusy(state.busy);
    } catch (failure) { state.loaded=false;error(failure.message); }
  }
  function updateStatus(status) {
    $('model-state').textContent = status.available ? 'Modelo conectado' : 'API offline';
    $('connection-notice').hidden = status.available;
    $('connection-notice').textContent = 'A API Python ainda não está disponível. Inicie o serviço do Lab 04 e confira a configuração LAB04_API_URL.';
    $('comparison-hint').textContent = status.comparison_enabled
      ? `${status.comparison_model} / cada comparação envia o contexto e gera uma chamada paga.`
      : 'Configure OPENAI_API_KEY no servidor para habilitar a comparação opcional.';
    if (!status.comparison_enabled) $('compare-openai').checked = false;
    if (!state.busy) {
      setBusy(false);
      $('stage-message').textContent = status.available ? 'Pronto para um novo teste.' : 'Conecte a API para iniciar.';
    }
    if (status.available) loadData();
  }
  window.addEventListener('slm-status', event=>updateStatus(event.detail));
  if (window.labStatus) updateStatus(window.labStatus);
  function renderContext(context) {
    $('explanation-panel').hidden = false;
    $('prompt-view').value = context.prompt || '';
    $('policy-excerpts').replaceChildren();
    if (!(context.policies || []).length) $('policy-excerpts').append(element('p','muted small','Nenhum trecho de regra foi recuperado para este histórico.'));
    (context.policies || []).forEach(policy=>{
      const card=element('article','policy-card');
      card.append(element('strong','',policy.id),element('p','',policy.text));$('policy-excerpts').append(card);
    });
  }
  function renderDecision(decision, context) {
    $('result-empty').hidden = true;$('result-filled').hidden = false;
    const review = decision.requires_review;
    $('result-state').textContent = 'Teste concluído';$('result-state').className='pill purple';
    $('decision-choice').textContent = decision.choice || 'Revisão necessária';
    $('decision-confidence').textContent = percent(decision.confidence);
    $('confidence-fill').style.width = Number.isFinite(decision.confidence) ? `${decision.confidence*100}%` : '0%';
    $('decision-review').textContent = review ? 'Encaminhar para revisão' : 'Acima do limiar';
    $('decision-review').className='pill '+(review?'yellow':'green');
    $('decision-latency').textContent=seconds(decision.latency_s);
    $('decision-note').textContent = decision.reason === 'context_exceeded'
      ? 'O evento mais recente ultrapassou a janela. O caso deve ser revisto.'
      : `Limiar: ${percent(decision.threshold)}. ${decision.choice==='retrieve_utterance'?'O próximo passo é falar com o cliente ou coletar informação.':'Esta recomendação não executa a ação.'}`;
    $('input-tokens').textContent = context?.input_tokens ?? decision.input_tokens ?? 'Indisponível';
    $('dropped-turns').textContent = decision.dropped_turns ?? 0;
    $('policy-count').textContent = decision.policy_ids?.length ?? 0;
    $('expected-result').hidden = !state.expected;
    if (state.expected) $('expected-result').textContent = `Referência ABCD: ${state.expected}. ${decision.choice===state.expected?'A escolha coincide com o gabarito.':'A escolha difere do gabarito.'}`;
    const entries = Object.entries(decision.probabilities || {}).sort((a,b)=>b[1]-a[1]);
    $('probabilities-panel').hidden = !entries.length;
    if (state.chart) state.chart.destroy();
    state.chart = null;
    if (entries.length && window.Chart) {
      const top = entries.slice(0,8);
      state.chart=new Chart($('probability-chart'),{type:'bar',data:{labels:top.map(e=>e[0]),datasets:[{data:top.map(e=>e[1]*100),backgroundColor:top.map((_,i)=>i===0?'#b297ff':'#6b5c8d'),borderRadius:5,barThickness:13}]},options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>percent(ctx.parsed.x/100)}}},scales:{x:{min:0,max:100,ticks:{color:'#9395a4',font:{size:9},callback:n=>n+'%'},grid:{color:'#85859718'},border:{display:false}},y:{ticks:{color:'#9395a4',font:{size:9}},grid:{display:false},border:{display:false}}}}});
    }
    $('probability-table').replaceChildren();
    entries.forEach(([label,probability])=>{const row=element('div','probability-row');row.append(element('code','',label),element('span','',percent(probability)));$('probability-table').append(row);});
  }
  function renderComparison(comparison,decision) {
    $('comparison-panel').hidden = false;
    if (comparison.status === 'api_error') {
      $('comparison-model').textContent='Comparação não concluída';$('comparison-choice').textContent='Erro na API';
      $('comparison-detail').textContent=comparison.message || 'Confira a configuração do servidor.';
      $('comparison-match').textContent='Falha';return;
    }
    $('comparison-model').textContent = comparison.model;
    $('comparison-choice').textContent = comparison.choice;
    $('comparison-match').textContent = comparison.status !== 'ok' ? 'Rótulo inválido' : (comparison.choice===decision?.choice?'Mesma escolha':'Escolhas diferentes');
    $('comparison-detail').textContent = `${seconds(comparison.latency_s)} / ${comparison.prompt_tokens} tokens de entrada / ${comparison.completion_tokens} de saída / custo estimado: US$ ${Number(comparison.estimated_cost_usd).toFixed(6)}`;
  }
  const actionDescriptions = {
    'retrieve_utterance':'Conversar com o cliente ou coletar informações',
    'search-timing':'Consultar prazos e validade',
    'search-faq':'Consultar as perguntas frequentes',
    'pull-up-account':'Consultar a conta do cliente',
    'verify-identity':'Verificar a identidade do cliente',
    'validate-purchase':'Validar a compra',
    'offer-refund':'Oferecer reembolso',
    'update-order':'Atualizar o pedido',
    'update-account':'Atualizar a conta',
    'record-reason':'Registrar o motivo do atendimento',
    'notify-team':'Notificar a equipe responsável',
    'send-link':'Enviar um link',
    'search-policy':'Consultar a política aplicável',
    'shipping-status':'Consultar o estado do envio',
    'subscription-status':'Consultar o estado da assinatura'
  };
  function renderOutcome(result) {
    $('outcome-panel').hidden=false;
    $('journey-panel').hidden=false;
    const decision=result.slm;
    $('simple-choice').textContent=actionDescriptions[decision.choice] || (decision.choice?`Próxima ação: ${decision.choice}`:'Revisar o atendimento');
    $('simple-guidance').textContent=decision.reason==='context_exceeded'
      ? 'O histórico ultrapassou o limite de entrada. Encaminhe o caso para um atendente.'
      : (decision.requires_review?'Um atendente deve revisar esta recomendação antes de prosseguir.':`Etapa recomendada: ${decision.choice}. O sistema consumidor decide como responder ou executar a ação.`);
    $('simple-review').textContent=decision.requires_review?'Revisão indicada':'Recomendação acima do limiar';
    $('simple-review').className='pill '+(decision.requires_review?'yellow':'green');
    const outcome=state.outcome;
    $('delivered-messages').replaceChildren();
    const replies=outcome?.delivered_messages||[];
    $('outcome-empty').hidden=!!replies.length;
    $('outcome-empty').textContent=outcome?'O registro não contém uma mensagem posterior do atendente. Veja as ações e o desfecho no diagrama.':'Este histórico não possui continuação registrada. Selecione um exemplo ABCD para consultar a resposta efetivamente entregue.';
    replies.forEach(message=>{const quote=element('blockquote','delivered-message');quote.append(element('p','',message.text),element('small','',`Atendente / ${message.id}`));$('delivered-messages').append(quote);});
    $('recorded-action').hidden=!outcome?.last_action;
    if(outcome?.last_action){$('recorded-action-label').textContent=outcome.last_action.action;$('recorded-action-text').textContent=outcome.last_action.text;}
    $('outcome-source').textContent=outcome?`Fonte: ${outcome.source} / ${outcome.id}. Este registro mostra o que aconteceu no dataset e fica fora do prompt do modelo.`:'';
    window.LabJourney?.render(state.history,result,outcome);
  }
  function chooseExample(example) {
    if(!example)return;
    state.history=structuredClone(example.history);state.expected=example.expected;state.outcome=example.outcome||null;
    state.origin=`ABCD / ${example.name} / ${example.id||'demonstração'}`;
    $('example-select').value=example.name;
    clearResult();error('');renderTimeline();
  }
  function showFullResult(result) {
    state.result = result;
    renderContext(result.context);renderDecision(result.slm,result.context);
    if (result.comparison) renderComparison(result.comparison,result.slm);
    else $('comparison-panel').hidden = true;
    renderOutcome(result);
    $('result-json').value = JSON.stringify(result,null,2);
  }
  function saveRecord(payload,result) {
    const record={id:Date.now(),time:new Date().toISOString(),history:payload.history,expected:state.expected,outcome:state.outcome,origin:state.origin,result};
    state.records.unshift(record);state.records=state.records.slice(0,20);
    try {localStorage.setItem(storeKey,JSON.stringify(state.records));} catch (_) { $('stage-message').textContent+=' Histórico não salvo: armazenamento do navegador cheio.'; }
    renderRecords();
  }
  function renderRecords() {
    $('session-history').replaceChildren();
    if (!state.records.length) {$('session-history').append(element('p','muted small','Nenhum teste salvo ainda.'));return;}
    state.records.forEach(record=>{
      if (!record.result?.slm || !Array.isArray(record.history)) return;
      const button=element('button','session-record');button.type='button';
      button.append(element('small','',new Date(record.time).toLocaleString('pt-BR')),element('strong','',record.result.slm.choice||'Revisão'),element('small','',`${percent(record.result.slm.confidence)} / ${record.history.length} eventos`));
      button.addEventListener('click',()=>{if(state.busy)return;state.history=structuredClone(record.history);state.expected=record.expected;state.outcome=record.outcome||state.examples.find(example=>JSON.stringify(example.history)===JSON.stringify(record.history))?.outcome||null;state.origin=record.origin||'Teste restaurado';renderTimeline();showFullResult(record.result);$('stage-message').textContent='Resultado salvo restaurado. Execute novamente para uma nova inferência.';});
      $('session-history').append(button);
    });
  }
  function validate(history) {
    if (!Array.isArray(history) || !history.length || history.length>200) throw new Error('Use de 1 a 200 eventos no histórico.');
    history.forEach(turn=>{
      if (!turn || !['customer','agent','action'].includes(turn.role) || typeof turn.text!=='string' || !turn.text.trim()) throw new Error('Preencha a mensagem e o papel de cada evento.');
      if (turn.text.length>16000) throw new Error('Cada evento aceita até 16.000 caracteres.');
      if (Object.keys(turn).some(key=>!['role','text','action'].includes(key))) throw new Error('Cada evento aceita somente role, text e action.');
      if (turn.action && (turn.role!=='action' || !state.catalog?.actions?.includes(turn.action))) throw new Error('Use uma ação do catálogo em um evento de ação executada.');
    });
  }
  async function run() {
    if(state.busy)return;
    error('');
    try {validate(state.history);} catch(failure){error(failure.message);return;}
    const payload={history:structuredClone(state.history),compare:$('compare-openai').checked};
    clearResult();setBusy(true);$('stage-message').textContent='Enviando histórico...';
    const partial={};let finished=false;
    try {
      const response=await fetch('/api/lab04/predict/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:AbortSignal.timeout(185000)});
      if(!response.ok){const data=await response.json();throw new Error(data.message||'O teste foi recusado.');}
      if(!response.body)throw new Error('Este navegador não suporta streaming de respostas.');
      const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
      const eventBlock=block=>{
        let name='message';const dataLines=[];
        for(const line of block.split('\n')){if(line.startsWith('event:'))name=line.slice(6).trim();if(line.startsWith('data:'))dataLines.push(line.slice(5).trimStart());}
        if(!dataLines.length)return;
        const data=JSON.parse(dataLines.join('\n'));
        if(name==='stage')$('stage-message').textContent=data.message;
        if(name==='context'){partial.context=data;renderContext(data);}
        if(name==='slm'){partial.slm=data;renderDecision(data,partial.context);}
        if(name==='comparison'){partial.comparison=data;renderComparison(data,partial.slm);}
        if(name==='failure')throw new Error(data.message);
        if(name==='done'){finished=true;showFullResult(data);$('stage-message').textContent=`Teste concluído em ${seconds(data.elapsed_s)}.`;saveRecord(payload,data);}
      };
      while(true){const {value,done}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done}).replace(/\r\n/g,'\n');let index;while((index=buffer.indexOf('\n\n'))>=0){eventBlock(buffer.slice(0,index));buffer=buffer.slice(index+2);}if(done)break;}
      if(!finished)throw new Error('A conexão encerrou antes de concluir o teste.');
    } catch(failure){error(failure.message);$('stage-message').textContent='Teste não concluído.';}
    finally {setBusy(false);}
  }
  $('decide').addEventListener('click',run);
  $('add-event').addEventListener('click',()=>{if(state.history.length>=200){error('Limite de 200 eventos atingido.');return;}state.history.push({role:'customer',text:''});edited();renderTimeline();$('timeline').lastElementChild.querySelector('textarea').focus();});
  $('load-example').addEventListener('click',()=>{
    if(!state.examples.length)return;
    const example=state.examples[Math.floor(Math.random()*state.examples.length)];
    chooseExample(example);
  });
  $('example-select').addEventListener('change',()=>chooseExample(state.examples.find(example=>example.name===$('example-select').value)));
  $('import-json').addEventListener('click',()=>{
    try {const payload=JSON.parse($('history-json').value);if(!payload || Object.keys(payload).length!==1 || !payload.history)throw new Error('Envie um objeto contendo somente history.');validate(payload.history);state.history=structuredClone(payload.history);edited();renderTimeline();error('');}
    catch(failure){error(failure.message);}
  });
  $('download-result').addEventListener('click',()=>{
    if(!state.result)return;
    const data={history:state.history,result:state.result,reference_outcome:state.outcome};
    const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
    const link=element('a');link.href=url;link.download='slm-lab04-teste.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  $('clear-history').addEventListener('click',()=>{state.records=[];try{localStorage.removeItem(storeKey);}catch(_){}renderRecords();});
  renderTimeline();renderRecords();
})();
