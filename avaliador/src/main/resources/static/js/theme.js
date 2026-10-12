(() => {
  let theme = 'dark';
  try { theme = localStorage.getItem('slm-theme') || theme; } catch (_) {}
  document.documentElement.dataset.bsTheme = theme === 'light' ? 'light' : 'dark';
})();
