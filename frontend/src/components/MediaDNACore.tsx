import { useMemo } from 'react';

interface MediaDNACoreProps {
  className?: string;
  isIgnited?: boolean;
}

export default function MediaDNACore({ className = "", isIgnited = false }: MediaDNACoreProps) {
  // Generate deterministic particles for the orbital rings
  const particles = useMemo(() => {
    return Array.from({ length: 40 }).map((_, i) => ({
      angle: (i * 9) % 360,
      radius: 60 + (i % 3) * 30, // rings at 60, 90, 120
      speed: 10 + (i % 5) * 5,
      size: i % 4 === 0 ? 3 : 1.5,
      opacity: 0.2 + (i % 5) * 0.15,
      color: i % 2 === 0 ? 'var(--color-cyan-accent)' : 'var(--color-indigo-accent)'
    }));
  }, []);

  return (
    <div className={`relative ${className}`}>
      {/* ─── Ambient Core Lighting ─── */}
      <div 
        className="absolute inset-0 rounded-full pointer-events-none transition-opacity duration-1000"
        style={{ 
          background: 'radial-gradient(circle, var(--color-cyan-glow) 0%, transparent 60%)',
          opacity: isIgnited ? 0.6 : 0.2,
          transform: isIgnited ? 'scale(1.2)' : 'scale(1)',
          filter: isIgnited ? 'blur(20px)' : 'blur(10px)'
        }}
      />

      <svg viewBox="0 0 500 500" className="w-full h-full relative z-10" data-hover="node">
        
        <defs>
          <radialGradient id="coreEnergy" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="var(--color-cyan-accent)" stopOpacity={isIgnited ? "0.8" : "0.3"} />
            <stop offset="40%" stopColor="var(--color-cyan-accent)" stopOpacity={isIgnited ? "0.3" : "0.1"} />
            <stop offset="100%" stopColor="var(--color-cyan-accent)" stopOpacity="0" />
          </radialGradient>
          <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <g transform="translate(250, 250)">
          
          {/* ─── Level 4: Telemetry Ring ─── */}
          <circle cx="0" cy="0" r="220" fill="none" stroke="var(--color-border-subtle)" strokeWidth="1" />
          <circle cx="0" cy="0" r="220" fill="none" stroke="var(--color-cyan-accent)" strokeWidth="1"
            strokeDasharray="1 10" opacity="0.3"
            style={{ animation: 'spin-slow 60s linear infinite' }} />
          
          <path id="textPath" d="M -230 0 A 230 230 0 1 1 230 0 A 230 230 0 1 1 -230 0" fill="none" />
          <text fill="var(--color-text-muted)" fontSize="7" fontFamily="'JetBrains Mono', monospace" letterSpacing="0.2em" opacity="0.5">
            <textPath href="#textPath" startOffset="0%">
              0x00A1 • MEDIA DNA FORENSIC ANALYSIS ENGINE • SYS_ACTIVE • TENSOR_FUSION_V1
            </textPath>
            <textPath href="#textPath" startOffset="50%">
              0x00A2 • VIDEO_CAVMAEFT • VISUAL_ANOMALY • DETERMINISTIC_WEIGHTING
            </textPath>
          </text>

          {/* ─── Level 3: Measurement Hash Ring ─── */}
          <circle cx="0" cy="0" r="180" fill="none" stroke="var(--color-border-subtle)" strokeWidth="0.5" />
          <g style={{ animation: 'spin-slow 40s linear infinite reverse' }}>
            {Array.from({ length: 72 }).map((_, i) => (
              <line key={i} x1="0" y1="-175" x2="0" y2={i % 6 === 0 ? "-185" : "-180"} 
                stroke={i % 6 === 0 ? "var(--color-cyan-accent)" : "var(--color-border-medium)"} 
                strokeWidth={i % 6 === 0 ? "2" : "1"} 
                transform={`rotate(${i * 5})`} opacity={i % 6 === 0 ? 0.6 : 0.3} />
            ))}
          </g>

          {/* ─── Level 2: Signal Data Tracks ─── */}
          <circle cx="0" cy="0" r="140" fill="none" stroke="var(--color-border-subtle)" strokeWidth="1" strokeDasharray="4 8" />
          <circle cx="0" cy="0" r="140" fill="none" stroke="var(--color-indigo-accent)" strokeWidth="2"
            strokeDasharray="40 120" opacity="0.6" filter="url(#glow)"
            style={{ animation: isIgnited ? 'spin-slow 2s linear infinite' : 'spin-slow 15s linear infinite' }} />
          
          <circle cx="0" cy="0" r="110" fill="none" stroke="var(--color-border-subtle)" strokeWidth="1" strokeDasharray="2 6" />
          <circle cx="0" cy="0" r="110" fill="none" stroke="var(--color-emerald)" strokeWidth="1.5"
            strokeDasharray="20 80" opacity="0.8" filter="url(#glow)"
            style={{ animation: isIgnited ? 'spin-slow 1.5s linear infinite reverse' : 'spin-slow 10s linear infinite reverse' }} />

          {/* ─── Level 1: Particles ─── */}
          {particles.map((p, i) => {
            const rad = (p.angle * Math.PI) / 180;
            const x = p.radius * Math.cos(rad);
            const y = p.radius * Math.sin(rad);
            return (
              <circle key={i} cx={x} cy={y} r={p.size} fill={p.color} opacity={p.opacity}
                style={{ 
                  animation: `pulse-glow ${p.speed}s infinite alternate`,
                  transformOrigin: '0 0',
                  transform: `rotate(${p.angle}deg)` 
                }} 
              />
            );
          })}

          {/* ─── The Core Geometry ─── */}
          <circle cx="0" cy="0" r="60" fill="url(#coreEnergy)" />
          
          <g style={{ animation: 'dna-pulse 4s infinite' }}>
            <circle cx="0" cy="0" r="50" fill="none" stroke="var(--color-cyan-accent)" strokeWidth="1" opacity="0.4" />
            <path d="M -40 0 C -20 -40, 20 40, 40 0 C 20 -40, -20 40, -40 0" fill="none" stroke="var(--color-cyan-accent)" strokeWidth="1.5" opacity="0.8" />
            <path d="M 0 -40 C 40 -20, -40 20, 0 40 C -40 20, 40 -20, 0 -40" fill="none" stroke="var(--color-cyan-accent)" strokeWidth="1.5" opacity="0.8" />
          </g>
          
          {/* Central Ignition Point */}
          <circle cx="0" cy="0" r="8" fill="var(--color-cyan-accent)" filter="url(#glow)">
            <animate attributeName="r" values={isIgnited ? "8;16;8" : "6;8;6"} dur={isIgnited ? "0.5s" : "2s"} repeatCount="indefinite" />
          </circle>

          {/* ─── Signal Input Nodes ─── */}
          {[
            { angle: -144, label: 'VIDEO', color: 'var(--color-cyan-accent)' },
            { angle: -72, label: 'AUDIO', color: 'var(--color-indigo-accent)' },
            { angle: 0, label: 'VISUAL', color: 'var(--color-emerald)' },
            { angle: 72, label: 'TEMPORAL', color: 'var(--color-crimson)' },
            { angle: 144, label: 'METADATA', color: 'var(--color-amber)' },
          ].map((signal, i) => {
            const rad = (signal.angle * Math.PI) / 180;
            const xOuter = 220 * Math.cos(rad);
            const yOuter = 220 * Math.sin(rad);
            const xInner = 60 * Math.cos(rad);
            const yInner = 60 * Math.sin(rad);
            
            return (
              <g key={i}>
                {/* Connecting Line */}
                <line x1={xOuter} y1={yOuter} x2={xInner} y2={yInner}
                  stroke={signal.color} strokeWidth="1" opacity="0.2"
                  strokeDasharray="4 4" />
                
                {/* Flowing Signal (only when ignited) */}
                {isIgnited && (
                  <circle cx="0" cy="0" r="3" fill={signal.color} filter="url(#glow)">
                    <animateMotion path={`M ${xOuter} ${yOuter} L ${xInner} ${yInner}`} dur={`${0.8 + i*0.2}s`} repeatCount="indefinite" />
                  </circle>
                )}

                {/* Outer Node */}
                <circle cx={xOuter} cy={yOuter} r="6" fill="var(--color-surface-2)" stroke={signal.color} strokeWidth="2" />
                <circle cx={xOuter} cy={yOuter} r="2" fill={signal.color} />
                
                {/* Label */}
                <g transform={`translate(${xOuter + (Math.cos(rad) * 20)}, ${yOuter + (Math.sin(rad) * 20)})`}>
                  <text textAnchor={Math.cos(rad) > 0.1 ? "start" : Math.cos(rad) < -0.1 ? "end" : "middle"} 
                        y={Math.sin(rad) > 0.1 ? 10 : Math.sin(rad) < -0.1 ? -5 : 4} 
                        fill={signal.color} fontSize="8" fontFamily="'JetBrains Mono', monospace" letterSpacing="0.1em" fontWeight="600">
                    {signal.label}
                  </text>
                </g>
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}
