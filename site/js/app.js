(() => {
  const glow = document.querySelector('.cursor-glow');
  const menu = document.querySelector('.mobile-menu');
  const nav = document.querySelector('.desktop-nav');

  window.addEventListener('pointermove', (event) => {
    if (glow) {
      glow.style.left = event.clientX + 'px';
      glow.style.top = event.clientY + 'px';
    }
  }, { passive: true });

  document.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener('click', () => {
      document.querySelector('.desktop-nav')?.classList.remove('open');
      menu?.setAttribute('aria-expanded', 'false');
    });
  });

  menu?.addEventListener('click', () => {
    const expanded = menu.getAttribute('aria-expanded') === 'true';
    menu.setAttribute('aria-expanded', String(!expanded));
    nav?.classList.toggle('open', !expanded);
  });

  // Keeps the page feeling like the product without simulating inference.
  document.querySelectorAll('.demo-card').forEach((card) => {
    card.addEventListener('mouseenter', () => card.classList.add('inspecting'));
    card.addEventListener('mouseleave', () => card.classList.remove('inspecting'));
  });
})();
