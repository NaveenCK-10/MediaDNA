import { motion, AnimatePresence } from 'framer-motion';
import { Check, Loader, Crosshair, Fingerprint, Activity, Clock, Cpu } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { JobEvent } from '../types';

interface ProcessingViewProps {
  filename: string;
  jobProgress: JobEvent | null;
}

const FORENSIC_STAGES = [
  { id: 'QUEUED', title: 'QUEUED', activeNodes: [] },
  { id: 'VALIDATING', title: 'INPUT VALIDATION', activeNodes: [] },
  { id: 'INSPECTING_MEDIA', title: 'MEDIA INSPECTION', activeNodes: ['METADATA'] },
  { id: 'EXTRACTING_VIDEO', title: 'VIDEO EXTRACTION', activeNodes: ['VISUAL'] },
  { id: 'VISUAL_ANALYSIS', title: 'VISUAL DIAGNOSTICS', activeNodes: ['VISUAL', 'TEMPORAL'] },
  { id: 'EXTRACTING_AUDIO', title: 'AUDIO EXTRACTION', activeNodes: ['AUDIO'] },
  { id: 'AUDIO_PREPROCESSING', title: 'AUDIO PREPROCESSING', activeNodes: ['AUDIO'] },
  { id: 'AUDIO_ANALYSIS', title: 'AUDIO DIAGNOSTICS', activeNodes: ['AUDIO', 'TEMPORAL'] },
  { id: 'TEMPORAL_ANALYSIS', title: 'TEMPORAL FORENSICS', activeNodes: ['TEMPORAL'] },
  { id: 'LATE_FUSION', title: 'V22.4F MULTIMODAL ANALYSIS', activeNodes: ['AUDIO', 'VISUAL', 'TEMPORAL'] },
  { id: 'CALIBRATION', title: 'DECISION SCORE CALIBRATION', activeNodes: ['AUDIO', 'VISUAL'] },
  { id: 'DECISION', title: 'POLICY APPLICATION', activeNodes: ['METADATA'] },
  { id: 'FORENSIC_EVIDENCE', title: 'EVIDENCE AGGREGATION', activeNodes: ['METADATA'] },
  { id: 'REPORT_GENERATION', title: 'REPORT GENERATION', activeNodes: [] }
];

export default function ProcessingView({ filename, jobProgress }: ProcessingViewProps) {
  const currentStageId = jobProgress?.stage || 'QUEUED';
  const [startTime] = useState(Date.now());
  const [elapsed, setElapsed] = useState('00:00.0');
  
  useEffect(() => {
    const timer = setInterval(() => {
      const ms = Date.now() - startTime;
      const secs = Math.floor(ms / 1000);
      const dec = Math.floor((ms % 1000) / 100);
      const m = Math.floor(secs / 60).toString().padStart(2, '0');
      const s = (secs % 60).toString().padStart(2, '0');
      setElapsed(`${m}:${s}.${dec}`);
    }, 100);
    return () => clearInterval(timer);
  }, [startTime]);
  
  let currentStageIdx = FORENSIC_STAGES.findIndex(s => s.id === currentStageId);
  if (currentStageIdx === -1) currentStageIdx = 0;
  const activeStage = FORENSIC_STAGES[currentStageIdx];
  const isFinalizing = jobProgress?.status === 'complete';
  const progressPercent = jobProgress?.progress || 0;

  return (
    <div className="flex flex-col xl:flex-row gap-8 items-center justify-center py-8 w-full max-w-[1400px] mx-auto min-h-[70vh] opacity-0 animate-reveal">
      
      {/* ─── LEFT: PIPELINE STAGES ─── */}
      <div className="w-full xl:w-[45%] flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-white/10 pb-4 mb-2">
          <h3 className="font-mono text-xs font-bold tracking-[0.2em] text-white flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            FORENSIC PIPELINE
          </h3>
          <div className="font-mono text-[0.65rem] text-cyan-400 border border-cyan-500/30 bg-cyan-900/20 px-2 py-1 rounded">
             V22.4F FUSION ENGINE
          </div>
        </div>
        
        <div className="flex flex-col gap-1.5 overflow-hidden rounded-xl border border-white/5 bg-black/40 p-2 relative">
          <div className="absolute top-0 bottom-0 left-8 w-[1px] bg-white/5"></div>
          {FORENSIC_STAGES.map((stage, idx) => {
            const isActive = stage.id === currentStageId && !isFinalizing;
            const isPast = (jobProgress?.completed_stages || []).includes(stage.id) || currentStageIdx > idx || isFinalizing;
            
            let colorClass = 'text-gray-500';
            let bgClass = 'transparent';
            if (isActive) {
              colorClass = 'text-cyan-400';
              bgClass = 'bg-cyan-500/10 border-cyan-500/30';
            } else if (isPast) {
              colorClass = 'text-emerald-400/80';
            }

            return (
              <div key={stage.id} className={`flex items-center gap-4 p-2 rounded-lg transition-all duration-500 relative z-10 border border-transparent ${bgClass}`}>
                <div className="w-5 text-center font-mono text-[0.55rem] tracking-widest text-white/30">
                  {(idx + 1).toString().padStart(2, '0')}
                </div>
                
                <div className="w-4 h-4 flex items-center justify-center shrink-0">
                  {isPast ? (
                    <Check className="w-3.5 h-3.5 text-emerald-500" />
                  ) : isActive ? (
                    <Loader className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
                  ) : (
                    <div className="w-1.5 h-1.5 rounded-full bg-white/10"></div>
                  )}
                </div>
                
                <div className={`flex-1 font-mono text-[0.65rem] tracking-widest uppercase transition-colors duration-500 ${colorClass}`}>
                  {stage.title}
                </div>
                
                {isActive && (
                  <div className="font-mono text-[0.55rem] text-cyan-400 bg-cyan-900/30 px-2 py-0.5 rounded ml-auto border border-cyan-500/20">
                    {progressPercent}%
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* ─── RIGHT: SCANNING VIEWPORT ─── */}
      <div className="w-full xl:w-[55%] flex flex-col gap-6">
        
        {/* Telemetry Strip */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="glass-card p-4 rounded-xl border border-white/5">
             <div className="font-mono text-[0.55rem] text-white/40 tracking-widest mb-2 uppercase flex items-center gap-2"><Fingerprint className="w-3 h-3"/> CURRENT OPERATION</div>
             <div className="font-mono text-[0.65rem] font-bold text-cyan-400 truncate uppercase" title={activeStage.title}>
               {isFinalizing ? 'FINALIZING' : activeStage.title}
             </div>
          </div>
          <div className="glass-card p-4 rounded-xl border border-white/5">
             <div className="font-mono text-[0.55rem] text-white/40 tracking-widest mb-2 uppercase flex items-center gap-2"><Clock className="w-3 h-3"/> ELAPSED TIME</div>
             <div className="font-mono text-[0.65rem] font-bold text-white">{elapsed}</div>
          </div>
          <div className="glass-card p-4 rounded-xl border border-white/5">
             <div className="font-mono text-[0.55rem] text-white/40 tracking-widest mb-2 uppercase flex items-center gap-2"><Crosshair className="w-3 h-3"/> TARGET</div>
             <div className="font-mono text-[0.65rem] font-bold text-white truncate" title={filename}>{filename}</div>
          </div>
          <div className="glass-card p-4 rounded-xl border border-white/5">
             <div className="font-mono text-[0.55rem] text-white/40 tracking-widest mb-2 uppercase flex items-center gap-2"><Cpu className="w-3 h-3"/> DEVICE</div>
             <div className="font-mono text-[0.65rem] font-bold text-indigo-400">V22.4F / CUDA</div>
          </div>
        </div>

        {/* Scanning viewport */}
        <div className="relative w-full aspect-video glass-card overflow-hidden flex items-center justify-center border border-white/10 rounded-2xl shadow-2xl">
          
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(0,212,255,0.05)_0%,transparent_70%)] pointer-events-none"></div>
          
          <div className={`reticle-corner reticle-tl transition-colors duration-700 ${isFinalizing ? 'border-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)]' : 'border-cyan-500'}`}></div>
          <div className={`reticle-corner reticle-tr transition-colors duration-700 ${isFinalizing ? 'border-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)]' : 'border-cyan-500'}`}></div>
          <div className={`reticle-corner reticle-bl transition-colors duration-700 ${isFinalizing ? 'border-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)]' : 'border-cyan-500'}`}></div>
          <div className={`reticle-corner reticle-br transition-colors duration-700 ${isFinalizing ? 'border-emerald-500 shadow-[0_0_15px_rgba(16,185,129,0.5)]' : 'border-cyan-500'}`}></div>

          {/* Scanning beam */}
          {!isFinalizing && (
            <div className="absolute top-0 left-0 w-full h-1/4 animate-scanline pointer-events-none opacity-50"
              style={{ background: 'linear-gradient(to bottom, transparent, rgba(0,212,255,0.2), transparent)' }}></div>
          )}

          {/* Central content */}
          <div className="relative z-10 flex flex-col items-center">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeStage.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 1.05 }}
                transition={{ duration: 0.4 }}
                className="flex flex-col items-center"
              >
                <div className="font-mono text-[0.55rem] text-cyan-500/70 tracking-[0.3em] uppercase mb-4">LOCAL ANALYSIS SOURCE</div>
                <div className="font-mono text-xl md:text-2xl font-bold tracking-widest text-white/90 mb-4 px-8 text-center drop-shadow-[0_0_10px_rgba(255,255,255,0.2)]">
                  {isFinalizing ? 'ANALYSIS COMPLETE' : activeStage.title}
                </div>
                <div className="font-mono text-[0.65rem] text-cyan-400 bg-cyan-900/20 px-4 py-1.5 rounded-full border border-cyan-500/20">
                  {isFinalizing ? 'AWAITING REPORT' : `PROGRESS: ${progressPercent}%`}
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Telemetry Corner Overlays */}
          <div className="absolute left-6 top-6 font-mono text-[0.5rem] text-white/30 tracking-widest">
            FRAME: {Math.floor(Math.random() * 1000 + 1000)}
          </div>
          <div className="absolute right-6 top-6 font-mono text-[0.5rem] text-white/30 tracking-widest">
            1920×1080 / 30 FPS
          </div>
          <div className="absolute left-6 bottom-6 font-mono text-[0.5rem] text-white/30 tracking-widest flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full ${isFinalizing ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500 animate-pulse'}`}></span>
            {isFinalizing ? 'READY' : 'PROCESSING'}
          </div>

          {/* Center Connection Lines during Fusion */}
          {(activeStage.id === 'LATE_FUSION' || activeStage.id === 'CALIBRATION') && (
            <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-40" preserveAspectRatio="none">
              <line x1="20%" y1="20%" x2="50%" y2="50%" stroke="#00d4ff" strokeWidth="1" strokeDasharray="4" className="animate-[dash_1.5s_linear_infinite]" />
              <line x1="80%" y1="20%" x2="50%" y2="50%" stroke="#00d4ff" strokeWidth="1" strokeDasharray="4" className="animate-[dash_1.5s_linear_infinite]" />
              <line x1="20%" y1="80%" x2="50%" y2="50%" stroke="#00d4ff" strokeWidth="1" strokeDasharray="4" className="animate-[dash_1.5s_linear_infinite]" />
              <line x1="80%" y1="80%" x2="50%" y2="50%" stroke="#00d4ff" strokeWidth="1" strokeDasharray="4" className="animate-[dash_1.5s_linear_infinite]" />
              <circle cx="50%" cy="50%" r="40" fill="none" stroke="#00d4ff" strokeWidth="1" strokeDasharray="4" className="animate-[dash_2s_linear_infinite_reverse]" />
            </svg>
          )}
        </div>
      </div>
    </div>
  );
}
