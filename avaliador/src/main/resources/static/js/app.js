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
  async function refreshStatus() {
    try {
      const response = await fetch('/api/status', {signal: AbortSignal.timeout(7000)});
      if (!response.ok) throw new Error('status');
      window.labStatus = await response.json();
    } catch (_) { window.labStatus = {available:false, comparison_enabled:false}; }
    let sqlAvailable=false;
    try { const response=await fetch('/api/lab02/status',{signal:AbortSignal.timeout(7000)}); sqlAvailable=response.ok && (await response.json()).available; } catch (_) {}
    let toolsAvailable=false;
    try { const response=await fetch('/api/lab01/status',{signal:AbortSignal.timeout(7000)}); toolsAvailable=response.ok && (await response.json()).available; } catch (_) {}
    let receiptsAvailable=false;
    try { const response=await fetch('/api/lab03/status',{signal:AbortSignal.timeout(7000)}); receiptsAvailable=response.ok && (await response.json()).available; } catch (_) {}
    document.querySelectorAll('[data-api-status]').forEach(el => {
      const connected=[toolsAvailable?'Ferramentas':null,receiptsAvailable?'Recibos':null,sqlAvailable?'SQL':null,window.labStatus.available?'Decisões':null].filter(Boolean);
      el.textContent=connected.length ? connected.join(' + ')+' / conectado' : 'APIs offline';
    });
    document.querySelectorAll('[data-status-dot]').forEach(el => el.classList.toggle('ready', window.labStatus.available||sqlAvailable||toolsAvailable||receiptsAvailable));
    window.dispatchEvent(new CustomEvent('slm-status', {detail:window.labStatus}));
  }
  refreshStatus();
  setInterval(refreshStatus, 30000);
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
