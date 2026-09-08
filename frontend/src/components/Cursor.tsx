import { useEffect, useRef } from 'react';

export default function Cursor() {
  const cursorRef = useRef<HTMLDivElement>(null);
  const ringRef = useRef<HTMLDivElement>(null);
  const glowRef = useRef<HTMLDivElement>(null);
  
  const mousePos = useRef({ x: -100, y: -100 });
  const renderPos = useRef({ x: -100, y: -100 });
  
  // Track hover states outside of React state
  const isHoveringNode = useRef(false);
  const isHoveringMedia = useRef(false);
  const hoverColor = useRef('var(--color-cyan-accent)');
  const rAF = useRef<number | null>(null);

  useEffect(() => {
    // Disable on touch devices
    if (window.matchMedia('(pointer: coarse)').matches) return;

    const onMouseMove = (e: MouseEvent) => {
      mousePos.current = { x: e.clientX, y: e.clientY };
      
      // Update global CSS variables for environment lighting instantly
      document.documentElement.style.setProperty('--mouse-x', `${e.clientX}px`);
      document.documentElement.style.setProperty('--mouse-y', `${e.clientY}px`);
    };

    const onMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const hoverEl = target.closest('[data-hover]');
      if (hoverEl) {
        const type = hoverEl.getAttribute('data-hover');
        isHoveringNode.current = type === 'node';
        isHoveringMedia.current = type === 'media';
        
        const customColor = hoverEl.getAttribute('data-color');
        hoverColor.current = customColor || 'var(--color-cyan-accent)';
      } else {
        isHoveringNode.current = false;
        isHoveringMedia.current = false;
        hoverColor.current = 'var(--color-cyan-accent)';
      }
    };

    window.addEventListener('mousemove', onMouseMove, { passive: true });
    window.addEventListener('mouseover', onMouseOver, { passive: true });

    const render = () => {
      // Linear interpolation for smooth trailing
      renderPos.current.x += (mousePos.current.x - renderPos.current.x) * 0.2;
      renderPos.current.y += (mousePos.current.y - renderPos.current.y) * 0.2;

      const { x, y } = renderPos.current;
      const hNode = isHoveringNode.current;
      const hMedia = isHoveringMedia.current;
      const c = hoverColor.current;

      if (cursorRef.current) {
        cursorRef.current.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${hNode ? 1.5 : 1})`;
        cursorRef.current.style.background = hNode || hMedia ? c : 'var(--color-text-primary)';
        cursorRef.current.style.boxShadow = hNode || hMedia ? `0 0 10px ${c}` : 'none';
      }

      if (ringRef.current) {
        ringRef.current.style.transform = `translate3d(${x}px, ${y}px, 0) scale(${hNode ? 1.5 : (hMedia ? 2 : 1)}) rotate(${hNode ? 45 : 0}deg)`;
        ringRef.current.style.border = hNode ? `1px dashed ${c}` : hMedia ? 'none' : '1px solid var(--color-border-medium)';
        ringRef.current.style.background = hNode ? c.replace(')', ', 0.05)').replace('var(', 'rgba(') : 'transparent';
      }

      if (glowRef.current) {
        glowRef.current.style.transform = `translate3d(${x}px, ${y}px, 0)`;
        glowRef.current.style.background = hNode || hMedia
          ? `radial-gradient(circle, ${c}22 0%, transparent 70%)` 
          : 'radial-gradient(circle, rgba(255,255,255,0.015) 0%, transparent 70%)';
      }

      rAF.current = requestAnimationFrame(render);
    };

    rAF.current = requestAnimationFrame(render);

    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseover', onMouseOver);
      if (rAF.current) cancelAnimationFrame(rAF.current);
    };
  }, []);

  return (
    <div className="hidden sm:block">
      {/* Base Dot */}
      <div 
        ref={cursorRef}
        className="pointer-events-none fixed z-[9999] rounded-full top-0 left-0"
        style={{ width: '4px', height: '4px', marginLeft: '-2px', marginTop: '-2px', transition: 'background 0.2s, transform 0.1s' }}
      />
      {/* Reticle */}
      <div
        ref={ringRef}
        className="pointer-events-none fixed z-[9998] rounded-full top-0 left-0 flex items-center justify-center"
        style={{ width: '32px', height: '32px', marginLeft: '-16px', marginTop: '-16px', transition: 'border 0.2s, transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)' }}
      />
      {/* Glow */}
      <div
        ref={glowRef}
        className="pointer-events-none fixed z-[9997] top-0 left-0 mix-blend-screen"
        style={{ width: '300px', height: '300px', marginLeft: '-150px', marginTop: '-150px', transition: 'background 0.5s' }}
      />
    </div>
  );
}
