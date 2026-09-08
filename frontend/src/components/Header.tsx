import { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import axios from 'axios';
import { Activity, Cpu, HardDrive, Database, Server } from 'lucide-react';

interface HealthData {
  status: string;
  model_loaded: boolean;
  device: string;
  checkpoint: string;
  gpu_name: string;
}

export default function Header() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [showStatus, setShowStatus] = useState(false);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await axios.get('/api/health');
        setHealth(res.data);
      } catch (err) {
        console.error("Health check failed", err);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `relative px-3 py-1.5 text-[0.65rem] tech-mono tracking-[0.2em] transition-all duration-300 ${
      isActive
        ? 'text-cyan-accent'
        : 'text-[#888888] hover:text-[#cccccc]'
    }`;

  return (
    <header className="fixed top-6 left-0 right-0 z-50 flex justify-center pointer-events-none px-4">
      <div className="glass-surface rounded-full flex items-center px-6 py-2 gap-8 pointer-events-auto shadow-[0_4px_30px_rgba(0,0,0,0.5)] bg-black/60 border border-white/10">
        
        {/* Logo */}
        <NavLink to="/" className="flex items-center gap-2 group" data-hover="node">
          <span className="text-sm font-bold tracking-[0.25em] text-[#eeeeee] flex items-center">
            MEDIA<span className="text-cyan-accent font-light ml-1">DNA</span>
          </span>
        </NavLink>

        <div className="w-[1px] h-4 bg-white/10"></div>

        {/* Navigation */}
        <nav className="flex items-center gap-1 hidden sm:flex">
          <NavLink to="/analyze" className={linkClass} data-hover="node">
            {({ isActive }) => (
              <>
                ANALYZE
                {isActive && <div className="absolute -bottom-1 left-1/2 w-1 h-1 bg-[var(--color-cyan-accent)] rounded-full transform -translate-x-1/2 shadow-[0_0_8px_var(--color-cyan-glow)]"></div>}
              </>
            )}
          </NavLink>
          <NavLink to="/architecture" className={linkClass} data-hover="node">
            {({ isActive }) => (
              <>
                ARCHITECTURE
                {isActive && <div className="absolute -bottom-1 left-1/2 w-1 h-1 bg-[var(--color-cyan-accent)] rounded-full transform -translate-x-1/2 shadow-[0_0_8px_var(--color-cyan-glow)]"></div>}
              </>
            )}
          </NavLink>
          <NavLink to="/history" className={linkClass} data-hover="node">
            {({ isActive }) => (
              <>
                HISTORY
                {isActive && <div className="absolute -bottom-1 left-1/2 w-1 h-1 bg-[var(--color-cyan-accent)] rounded-full transform -translate-x-1/2 shadow-[0_0_8px_var(--color-cyan-glow)]"></div>}
              </>
            )}
          </NavLink>
          <NavLink to="/research" className={linkClass} data-hover="node">
            {({ isActive }) => (
              <>
                RESEARCH
                {isActive && <div className="absolute -bottom-1 left-1/2 w-1 h-1 bg-[var(--color-cyan-accent)] rounded-full transform -translate-x-1/2 shadow-[0_0_8px_var(--color-cyan-glow)]"></div>}
              </>
            )}
          </NavLink>
        </nav>

        <div className="w-[1px] h-4 bg-white/10 hidden sm:block"></div>

        {/* System Status */}
        <div className="relative">
          <button 
            onClick={() => setShowStatus(!showStatus)}
            data-hover="node"
            className="flex items-center gap-3 px-2 py-1 rounded transition-colors group hover:bg-white/5"
          >
            <div className={`w-1.5 h-1.5 rounded-full ${health ? 'signal-dot-online animate-[pulse_2s_infinite]' : 'bg-[var(--color-crimson)] shadow-[0_0_8px_var(--color-crimson)]'}`}></div>
            <span className="tech-mono text-[0.6rem] text-[#888888] group-hover:text-white transition-colors flex items-center gap-2">
              {health ? 'SYSTEM ONLINE' : 'SYSTEM OFFLINE'}
              {health && health.device === 'cuda' && (
                 <span className="hidden sm:inline">| GPU ACTIVE</span>
              )}
            </span>
          </button>

          {showStatus && health && (
            <div className="absolute right-0 mt-4 w-72 glass-surface rounded p-4 text-xs font-mono animate-reveal border-t border-[var(--color-cyan-accent)] bg-black/90">
              <div className="flex items-center gap-2 mb-4 pb-2 border-b border-white/10">
                <Activity className="w-3.5 h-3.5 text-[var(--color-cyan-accent)]" />
                <span className="text-[var(--color-cyan-accent)] tech-mono font-bold">TELEMETRY</span>
              </div>
              <div className="space-y-3 tech-mono text-[0.65rem]">
                <div className="flex justify-between items-center">
                  <span className="text-[#888888] flex items-center gap-2"><Server className="w-3 h-3"/> BACKEND</span>
                  <span className="text-[#2a9d8f]">ONLINE</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#888888] flex items-center gap-2"><Cpu className="w-3 h-3"/> COMPUTE</span>
                  <span className="text-white uppercase">{health.device === 'cuda' ? 'CUDA ACTIVE' : 'CPU ACTIVE'}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#888888] flex items-center gap-2"><HardDrive className="w-3 h-3"/> GPU</span>
                  <span className="text-white truncate max-w-[120px]" title={health.gpu_name}>{health.gpu_name || 'NONE'}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#888888] flex items-center gap-2"><Database className="w-3 h-3"/> MODEL</span>
                  <span className="text-white truncate max-w-[120px]" title={health.checkpoint}>OpenAVFF</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
