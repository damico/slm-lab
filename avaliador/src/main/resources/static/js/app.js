(() => {
  const root = document.documentElement;
  document.querySelectorAll('[data-document-back]').forEach(link => link.addEventListener('click', event => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    if (document.referrer && new URL(document.referrer).origin === location.origin && history.length > 1) {
      event.preventDefault(); history.back();
    }
  }));
  document.getElementById('theme-toggle')?.addEventListener('click', () => {
    root.dataset.bsTheme = root.dataset.bsTheme === 'dark' ? 'light' : 'dark';
    try { localStorage.setItem('slm-theme', root.dataset.bsTheme); } catch (_) {}
    window.dispatchEvent(new CustomEvent('slm-theme-change'));
  });
  window.updateInferenceStatus=(labId,status)=>{
    const state=!status?'checking':!status.available?'offline':status.inference?.live_enabled===false?'benchmark':'ready';
    const labels={checking:'Verificando conexão',ready:'Inferência real disponível',benchmark:'Somente benchmark disponível',offline:'Inferência indisponível / reconectando'};
    document.querySelectorAll('[data-lab-inference]').forEach(badge=>{
      if(badge.dataset.labInference!==labId)return;
      badge.dataset.state=state;badge.querySelector('[data-inference-label]').textContent=labels[state];
    });
  };
  let refreshing=false, refreshAgain=false;
  async function refreshStatus() {
    if(refreshing){refreshAgain=true;return;}
    refreshing=true;
    try {
      const paths={'01':'/api/lab01/status','02':'/api/lab02/status','03':'/api/lab03/status','04':'/api/status','05':'/api/lab05/status'};
      const states=Object.fromEntries(await Promise.all(Object.entries(paths).map(async([id,path])=>{
        try {const response=await fetch(path,{cache:'no-store',signal:AbortSignal.timeout(7000)});if(!response.ok)throw Error('status');return [id,await response.json()];}
        catch(_){return [id,{available:false,comparison_enabled:false}];}
      })));
      window.labServices=states;window.labStatus=states['04'];
      Object.entries(states).forEach(([id,state])=>window.updateInferenceStatus(id,state));
      const names={'01':'Ferramentas','02':'SQL','03':'Recibos','04':'Decisões','05':'Finanças'};
      const connected=Object.keys(names).filter(id=>states[id].available).map(id=>names[id]);
      document.querySelectorAll('[data-api-status]').forEach(el=>el.textContent=connected.length?connected.join(' + ')+' / conectado':'APIs offline');
      document.querySelectorAll('[data-status-dot]').forEach(el=>el.classList.toggle('ready',connected.length>0));
      window.dispatchEvent(new CustomEvent('slm-services-status',{detail:states}));
      window.dispatchEvent(new CustomEvent('slm-status',{detail:window.labStatus}));
    } finally {refreshing=false;if(refreshAgain){refreshAgain=false;refreshStatus();}}
  }
  refreshStatus();setInterval(refreshStatus,30000);
  window.addEventListener('focus',refreshStatus);
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)refreshStatus();});
  const canvas = document.getElementById('benchmark-chart');
  if (canvas && window.Chart) {
    const colors = ['#b297ff','#b7f66a','#64606e','#ede0a4'];
    const chart = new Chart(canvas, {
      type:'bar', data:{labels:['SLM especializado','TF-IDF','Qwen base','OpenAI mini'],
        datasets:[{data:[85.45,75.87,65.13,55.37],backgroundColor:colors,borderRadius:7,barThickness:32}]},
      options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',animation:{duration:650},
        plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>`${ctx.parsed.x.toLocaleString('pt-BR')}%`}}},
        scales:{x:{min:0,max:100,ticks:{color:'#9395a4',font:{size:10},callback:v=>v+'%'},grid:{color:'#85859718'},border:{display:false}},
          y:{ticks:{color:'#9395a4',font:{size:11}},grid:{display:false},border:{display:false}}}}
    });
    window.addEventListener('slm-theme-change', () => chart.update());
  }
})();
