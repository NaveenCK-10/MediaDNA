import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface ProcessingViewProps {
  filename: string;
}

const FORENSIC_STAGES = [
  { id: '01', title: 'MEDIA INGESTION', desc: 'Inspecting uploaded media container', activeNodes: ['METADATA'] },
  { id: '02', title: 'VIDEO STREAM', desc: 'Extracting visual frames', activeNodes: ['VISUAL'] },
  { id: '03', title: 'AUDIO STREAM', desc: 'Extracting audio features', activeNodes: ['AUDIO'] },
  { id: '04', title: 'VISUAL ENCODER', desc: 'Processing facial/visual representations', activeNodes: ['VISUAL', 'TEMPORAL'] },
  { id: '05', title: 'AUDIO ENCODER', desc: 'Processing acoustic representations', activeNodes: ['AUDIO', 'TEMPORAL'] },
  { id: '06', title: 'CROSS-MODAL FUSION', desc: 'Running audio ↔ visual feature interaction', activeNodes: ['AUDIO', 'VISUAL', 'TEMPORAL', 'METADATA'] },
  { id: '07', title: 'OPENAVFF INFERENCE', desc: 'Running the trained classifier', activeNodes: ['AUDIO', 'VISUAL', 'TEMPORAL', 'METADATA'] },
  { id: '08', title: 'FORENSIC DECISION', desc: 'Preparing the final evidence result', activeNodes: [] },
];

export default function ProcessingView({ filename }: ProcessingViewProps) {
  const [currentStage, setCurrentStage] = useState(0);

  useEffect(() => {
    // Advance sequence every 800ms
    if (currentStage >= FORENSIC_STAGES.length - 1) return;
    const timer = setTimeout(() => {
      setCurrentStage(prev => prev + 1);
    }, 800);
    return () => clearTimeout(timer);
  }, [currentStage]);

  const activeStage = FORENSIC_STAGES[currentStage];
  const isFinalizing = currentStage === FORENSIC_STAGES.length - 1;

  return (
    <div className="flex flex-col md:flex-row gap-8 items-center justify-center py-10 w-full max-w-6xl mx-auto min-h-[60vh] opacity-0 animate-reveal">
      
      {/* ─── LEFT: PIPELINE STAGES ─── */}
      <div className="w-full md:w-1/3 flex flex-col gap-2">
        <h3 className="font-mono text-[0.65rem] font-bold tracking-[0.3em] text-gray-500 mb-4">FORENSIC PIPELINE</h3>
        <div className="flex flex-col gap-2">
          {FORENSIC_STAGES.map((stage, idx) => {
            const isActive = idx === currentStage;
            const isPast = idx < currentStage;
            let colorClass = 'text-gray-600 border-gray-800';
            if (isActive) colorClass = 'text-[#00e5ff] border-[#00e5ff] bg-[#00e5ff]/10';
            if (isPast) colorClass = 'text-emerald-500 border-emerald-500/30 bg-emerald-500/5';

            return (
              <div key={stage.id} className={`flex items-center gap-4 p-2 border-l-2 transition-all duration-300 ${colorClass}`}>
                <div className="font-mono text-[0.6rem] font-bold tracking-widest">{stage.id}</div>
                <div className="flex-1">
                  <div className="font-mono text-[0.6rem] font-bold tracking-widest uppercase">{stage.title}</div>
                  {isActive && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="font-mono text-[0.55rem] text-gray-400 mt-1"
                    >
                      {stage.desc}...
                    </motion.div>
                  )}
                </div>
                {isActive && <div className="w-1.5 h-1.5 rounded-full bg-[#00e5ff] animate-pulse shrink-0"></div>}
              </div>
            );
          })}
        </div>
      </div>

      {/* ─── RIGHT: SCANNING VIEWPORT ─── */}
      <div className="w-full md:w-2/3 flex flex-col items-center">
        
        <div className="mb-6 h-12 flex flex-col items-center justify-center">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeStage.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="font-mono text-[0.7rem] font-bold tracking-[0.2em] text-[#00e5ff]"
            >
              {isFinalizing ? (
                <span className="text-emerald-500 animate-pulse">FINALIZING INFERENCE...</span>
              ) : (
                <span>EXECUTING: {activeStage.title}</span>
              )}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Scanning viewport */}
        <div className="relative w-full aspect-video surface-2 overflow-hidden flex items-center justify-center border border-white/5 rounded-xl shadow-2xl">
          
          <div className={`reticle-corner reticle-tl transition-colors duration-500 ${isFinalizing ? 'border-emerald-500' : 'border-[#00e5ff]'}`}></div>
          <div className={`reticle-corner reticle-tr transition-colors duration-500 ${isFinalizing ? 'border-emerald-500' : 'border-[#00e5ff]'}`}></div>
          <div className={`reticle-corner reticle-bl transition-colors duration-500 ${isFinalizing ? 'border-emerald-500' : 'border-[#00e5ff]'}`}></div>
          <div className={`reticle-corner reticle-br transition-colors duration-500 ${isFinalizing ? 'border-emerald-500' : 'border-[#00e5ff]'}`}></div>

          {/* Scanning beam */}
          {!isFinalizing && (
            <div className="absolute top-0 left-0 w-full h-12 animate-scanline pointer-events-none"
              style={{ background: 'linear-gradient(to bottom, transparent, rgba(0,212,255,0.15), transparent)' }}></div>
          )}

          {/* Central content */}
          <div className="relative z-10 text-center">
            <div className="tech-label mb-2 text-gray-500">TARGET ACQUIRED</div>
            <div className="text-3xl font-bold mb-4 tracking-widest text-white/20">MEDIA</div>
            <div className="tech-mono text-[0.6rem] truncate max-w-[200px] mx-auto px-4 text-[#00e5ff] bg-[#00e5ff]/10 py-1 rounded">
              {filename}
            </div>
          </div>

          {/* Nodes */}
          <div className="absolute left-6 top-6 flex items-center gap-2">
            <div className={`signal-dot transition-all duration-300 ${activeStage.activeNodes.includes('AUDIO') ? 'bg-[#b388ff] scale-150 animate-pulse shadow-[0_0_10px_#b388ff]' : 'bg-gray-700'}`}></div>
            <span className="tech-label text-[0.55rem]">AUDIO</span>
          </div>
          <div className="absolute right-6 top-6 flex items-center gap-2">
            <span className="tech-label text-[0.55rem]">VISUAL</span>
            <div className={`signal-dot transition-all duration-300 ${activeStage.activeNodes.includes('VISUAL') ? 'bg-[#00e5ff] scale-150 animate-pulse shadow-[0_0_10px_#00e5ff]' : 'bg-gray-700'}`}></div>
          </div>
          <div className="absolute left-6 bottom-6 flex items-center gap-2">
            <div className={`signal-dot transition-all duration-300 ${activeStage.activeNodes.includes('TEMPORAL') ? 'bg-[#ef4444] scale-150 animate-pulse shadow-[0_0_10px_#ef4444]' : 'bg-gray-700'}`}></div>
            <span className="tech-label text-[0.55rem]">TEMPORAL</span>
          </div>
          <div className="absolute right-6 bottom-6 flex items-center gap-2">
            <span className="tech-label text-[0.55rem]">METADATA</span>
            <div className={`signal-dot transition-all duration-300 ${activeStage.activeNodes.includes('METADATA') ? 'bg-[#ffb300] scale-150 animate-pulse shadow-[0_0_10px_#ffb300]' : 'bg-gray-700'}`}></div>
          </div>

          {/* Center Connection Lines during Fusion */}
          {activeStage.id === '06' && (
            <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-30" preserveAspectRatio="none">
              <line x1="10%" y1="10%" x2="50%" y2="50%" stroke="#00e5ff" strokeWidth="1" strokeDasharray="4" className="animate-[dash_1s_linear_infinite]" />
              <line x1="90%" y1="10%" x2="50%" y2="50%" stroke="#00e5ff" strokeWidth="1" strokeDasharray="4" className="animate-[dash_1s_linear_infinite]" />
              <line x1="10%" y1="90%" x2="50%" y2="50%" stroke="#00e5ff" strokeWidth="1" strokeDasharray="4" className="animate-[dash_1s_linear_infinite]" />
              <line x1="90%" y1="90%" x2="50%" y2="50%" stroke="#00e5ff" strokeWidth="1" strokeDasharray="4" className="animate-[dash_1s_linear_infinite]" />
            </svg>
          )}
        </div>
      </div>
    </div>
  );
}
