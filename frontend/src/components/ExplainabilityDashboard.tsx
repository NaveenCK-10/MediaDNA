import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Maximize, Target, Radio, Hexagon, FileText } from 'lucide-react';
import type { AnalysisResponse } from '../types';

interface ExplainabilityDashboardProps {
  result: AnalysisResponse;
}

export default function ExplainabilityDashboard({ result }: ExplainabilityDashboardProps) {
  const explainability = result.explainability;
  if (!explainability) return null;

  const [activeTab, setActiveTab] = useState<'SUMMARY' | 'VISUAL' | 'AUDIO' | 'TEMPORAL' | 'A/V'>('SUMMARY');

  // Handle both deep narrative (from PDF gen) and fallback narrative
  const n: any = explainability.narrative || (result as any).narrative || {};
  const hasDeepNarrative = !!n.executive;
  const narrativeMode = n.mode || 'UNKNOWN';
  const narrativeText = n.text || 'No narrative available.';

  const correlation = explainability.av_alignment?.correlation_score;

  const renderTab = (id: typeof activeTab, label: string) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`relative px-4 py-2 font-mono text-[0.65rem] tracking-widest transition-colors ${
        activeTab === id ? 'text-white font-bold' : 'text-white/40 hover:text-white/80'
      }`}
    >
      {label}
      {activeTab === id && (
        <motion.div
          layoutId="activeTab"
          className="absolute bottom-0 left-0 right-0 h-[2px] bg-cyan-400 shadow-[0_0_10px_rgba(0,212,255,0.5)]"
        />
      )}
    </button>
  );

  return (
    <div className="mt-12 space-y-8">
      <div className="flex flex-col items-center justify-center h-12 relative font-mono text-[0.65rem] font-bold tracking-[0.3em]">
        <div className="text-[#a5b4fc] uppercase flex items-center gap-2">
          <Hexagon className="w-4 h-4 text-indigo-400" />
          Deep Explainability Forensics
        </div>
      </div>

      <div className="flex justify-center border-b border-white/10 mb-6 flex-wrap gap-x-2 gap-y-4">
        {renderTab('SUMMARY', 'SUMMARY')}
        {renderTab('VISUAL', 'VISUAL EVIDENCE')}
        {renderTab('AUDIO', 'AUDIO EVIDENCE')}
        {renderTab('TEMPORAL', 'TEMPORAL TIMELINE')}
        {renderTab('A/V', 'A/V CONSISTENCY')}
      </div>

      <AnimatePresence mode="wait">
        {activeTab === 'SUMMARY' && (
          <motion.div
            key="summary"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-6"
          >
            {/* ─── Forensic Narrative ─── */}
            <div className="glass-card p-8 border border-white/10 rounded-xl relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 to-transparent pointer-events-none"></div>
              <h3 className="font-mono text-sm font-bold tracking-[0.2em] text-white mb-6 flex items-center gap-3">
                <FileText className="w-5 h-5 text-indigo-400" />
                FORENSIC INTERPRETATION
              </h3>
              
              {hasDeepNarrative ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 text-sm font-mono text-gray-300 leading-relaxed">
                  <div>
                    <h4 className="text-[0.65rem] text-indigo-400 tracking-widest mb-2 border-b border-white/10 pb-1">EXECUTIVE</h4>
                    <p className="mb-4">{n.executive}</p>
                    <h4 className="text-[0.65rem] text-indigo-400 tracking-widest mb-2 border-b border-white/10 pb-1">EVIDENCE</h4>
                    <p>{n.evidence}</p>
                  </div>
                  <div>
                    <h4 className="text-[0.65rem] text-indigo-400 tracking-widest mb-2 border-b border-white/10 pb-1">MODALITY</h4>
                    <p className="mb-4">{n.modality}</p>
                    <h4 className="text-[0.65rem] text-indigo-400 tracking-widest mb-2 border-b border-white/10 pb-1">CONTEXT & LIMITATIONS</h4>
                    <p className="text-gray-400">{n.context} {n.limitations}</p>
                  </div>
                </div>
              ) : (
                <div className="text-sm font-mono text-gray-300 leading-relaxed pl-4 border-l-2 border-indigo-500/30">
                  <span className="text-[0.65rem] text-indigo-400 tracking-widest uppercase block mb-2 border-b border-white/10 pb-1 inline-block">EXECUTIVE ASSESSMENT ({narrativeMode})</span>
                  <br/>
                  {narrativeText}
                </div>
              )}
            </div>

            {/* ─── Modality Comparison ─── */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                { label: 'VISUAL', val: explainability.modality_sensitivity?.visual_only, status: 'ISOLATED' },
                { label: 'AUDIO', val: explainability.modality_sensitivity?.audio_only, status: 'ISOLATED' },
                { label: 'MULTIMODAL', val: result.classification?.decision_score, status: 'FUSED', focus: true }
              ].map((mod, i) => (
                <div key={i} className={`glass-card p-6 border rounded-xl flex flex-col justify-between ${mod.focus ? 'border-cyan-500/30 bg-cyan-900/5 ring-1 ring-cyan-500/10' : 'border-white/5'}`}>
                  <div className="flex justify-between items-center mb-4">
                    <span className="font-mono text-[0.65rem] text-white/50 tracking-widest">{mod.label}</span>
                    <span className="font-mono text-[0.55rem] px-2 py-0.5 rounded bg-white/5 text-white/40">{mod.status}</span>
                  </div>
                  <div className="font-mono text-3xl font-bold mb-4" style={{ color: mod.focus ? '#00d4ff' : '#ffffff' }}>
                    {mod.val !== undefined && mod.val !== null ? mod.val.toFixed(4) : 'N/A'}
                  </div>
                  <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
                    {mod.val !== undefined && mod.val !== null && (
                      <div className="h-full" style={{ width: `${Math.min(100, Math.max(0, mod.val * 100))}%`, backgroundColor: mod.focus ? '#00d4ff' : 'rgba(255,255,255,0.3)' }}></div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {activeTab === 'VISUAL' && (
          <motion.div
            key="visual"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-6"
          >
            {explainability.region_sensitivity && explainability.region_sensitivity.length > 0 ? (
              <div className="glass-card p-6 border border-white/10 rounded-xl">
                <h3 className="font-mono text-sm font-bold tracking-[0.2em] text-white mb-6 flex items-center gap-3">
                  <Target className="w-5 h-5 text-cyan-400" />
                  MODEL SENSITIVITY
                </h3>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  {/* Left: Video Frame visualization placeholder */}
                  <div className="relative aspect-video bg-black/50 border border-white/10 rounded overflow-hidden flex items-center justify-center">
                    <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:20px_20px]"></div>
                    
                    {/* Face Structure Visualization */}
                    <div className="absolute inset-0 flex items-center justify-center opacity-40 mix-blend-screen pointer-events-none">
                       <svg viewBox="0 0 100 100" className="w-3/4 h-3/4 stroke-cyan-500/50 fill-none" strokeWidth="0.5">
                          {/* Minimal geometric face topology */}
                          <path d="M 30 20 Q 50 10 70 20 Q 85 40 80 65 Q 50 95 20 65 Q 15 40 30 20 Z" />
                          <path d="M 40 45 Q 50 55 60 45" />
                          <circle cx="35" cy="35" r="4" className={explainability.region_sensitivity.some(r => r.region.toLowerCase().includes('eye') || r.region.toLowerCase().includes('left')) ? "fill-cyan-400/50 shadow-[0_0_15px_cyan]" : ""} />
                          <circle cx="65" cy="35" r="4" className={explainability.region_sensitivity.some(r => r.region.toLowerCase().includes('eye') || r.region.toLowerCase().includes('right')) ? "fill-cyan-400/50 shadow-[0_0_15px_cyan]" : ""} />
                          <path d="M 40 70 Q 50 80 60 70" strokeWidth="1" className={explainability.region_sensitivity.some(r => r.region.toLowerCase().includes('mouth')) ? "stroke-cyan-400 shadow-[0_0_15px_cyan]" : ""} />
                          {/* Map landmarks if explicitly mentioned */}
                       </svg>
                    </div>

                    <div className="font-mono text-[0.65rem] text-white/30 tracking-widest flex flex-col items-center gap-2 z-10">
                       <Maximize className="w-6 h-6" />
                       FRAME SENSITIVITY VIEWPORT
                    </div>
                  </div>

                  {/* Right: Regions */}
                  <div className="space-y-3">
                    <div className="font-mono text-[0.65rem] text-white/50 tracking-widest mb-4">MODEL-RESPONSIVE REGIONS</div>
                    {explainability.region_sensitivity.slice(0, 5).map((region, idx) => (
                      <div key={idx} className="flex items-center justify-between p-3 bg-white/5 hover:bg-white/10 transition-colors rounded border border-white/5 group">
                        <div className="flex items-center gap-4">
                          <div className="font-mono text-[0.6rem] text-cyan-500/50 group-hover:text-cyan-400 transition-colors">0{region.rank}</div>
                          <div className="font-mono text-sm text-gray-200">{region.region}</div>
                        </div>
                        <div className="flex items-center gap-6">
                          <div className="text-right w-20">
                            <div className="font-mono text-[0.55rem] text-gray-500">IMPACT &Delta;</div>
                            <div className={`font-mono text-sm font-bold ${region.delta > 0.05 ? 'text-red-400' : region.delta < -0.05 ? 'text-emerald-400' : 'text-gray-400'}`}>
                              {region.delta > 0 ? '+' : ''}{region.delta.toFixed(3)}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="glass-card p-12 border border-white/10 rounded-xl text-center">
                 <div className="font-mono text-[0.65rem] text-white/40 tracking-widest">VISUAL SENSITIVITY DATA UNAVAILABLE</div>
              </div>
            )}
          </motion.div>
        )}

        {activeTab === 'AUDIO' && (
          <motion.div
            key="audio"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-6"
          >
             <div className="glass-card p-12 border border-white/10 rounded-xl text-center">
                 <div className="font-mono text-[0.65rem] text-white/40 tracking-widest">AUDIO SENSITIVITY VISUALIZATION IN DEVELOPMENT</div>
             </div>
          </motion.div>
        )}

        {activeTab === 'TEMPORAL' && (
          <motion.div
            key="temporal"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-6"
          >
             <div className="glass-card p-6 border border-white/10 rounded-xl">
                <h3 className="font-mono text-sm font-bold tracking-[0.2em] text-white mb-10 flex items-center gap-3">
                  <Activity className="w-5 h-5 text-indigo-400" />
                  TEMPORAL TIMELINE
                </h3>
                
                {result.temporal?.evidence_intervals && result.temporal.evidence_intervals.length > 0 ? (
                  <div className="relative py-8 px-4 group">
                     {/* Base timeline line */}
                     <div className="absolute left-0 right-0 top-1/2 h-[1px] bg-white/10 -translate-y-1/2"></div>
                     <div className="absolute left-0 right-0 top-1/2 h-[1px] bg-indigo-500/30 -translate-y-1/2 w-0 group-hover:w-full transition-all duration-1000 ease-out"></div>
                     
                     {/* Timestamp markers */}
                     <div className="absolute left-0 top-1/2 -mt-6 font-mono text-[0.55rem] text-white/30 tracking-widest">00:00</div>
                     <div className="absolute right-0 top-1/2 -mt-6 font-mono text-[0.55rem] text-white/30 tracking-widest">END</div>

                     {/* Events */}
                     <div className="relative z-10 flex justify-between items-center w-full px-12">
                       {result.temporal.evidence_intervals.map((interval, idx) => (
                          <div key={idx} className="relative group/marker flex flex-col items-center cursor-pointer">
                             {/* Line connector */}
                             <div className="w-[1px] h-4 bg-white/20 absolute bottom-full mb-1"></div>
                             
                             {/* The point */}
                             <div className="w-2 h-2 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.8)] ring-2 ring-black group-hover/marker:scale-150 transition-transform"></div>
                             
                             {/* Hover tooltip */}
                             <div className="absolute top-full mt-3 opacity-0 group-hover/marker:opacity-100 transition-opacity bg-black/80 backdrop-blur border border-indigo-500/30 p-2 rounded w-32 pointer-events-none flex flex-col gap-1 z-20">
                                <span className="font-mono text-[0.5rem] text-indigo-400 tracking-widest border-b border-white/10 pb-1">TIMESTAMP EVENT</span>
                                <span className="font-mono text-[0.6rem] text-white break-words text-center">{interval}</span>
                             </div>
                          </div>
                       ))}
                     </div>
                  </div>
                ) : (
                  <div className="py-12 text-center font-mono text-[0.65rem] text-white/40 tracking-widest">
                    TEMPORAL EVENT DATA UNAVAILABLE
                  </div>
                )}
             </div>
          </motion.div>
        )}

        {activeTab === 'A/V' && (
          <motion.div
            key="av"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-6"
          >
             <div className="glass-card p-6 border border-white/10 rounded-xl">
                <h3 className="font-mono text-sm font-bold tracking-[0.2em] text-white mb-6 flex items-center gap-3">
                  <Radio className="w-5 h-5 text-amber-400" />
                  A/V CONSISTENCY
                </h3>
                
                {correlation !== undefined && correlation !== null ? (
                  <div className="flex flex-col items-center py-8">
                     <div className="font-mono text-[0.65rem] text-white/40 tracking-widest mb-4">CORRELATION SCORE</div>
                     <div className="font-mono text-5xl font-bold text-amber-400 mb-8">{correlation.toFixed(4)}</div>
                     
                     <div className="w-full max-w-lg h-24 relative flex flex-col justify-center">
                       {/* Abstract waveform representation */}
                       <div className="absolute w-full h-[1px] bg-white/20 top-1/2"></div>
                       <div className="flex justify-between items-center w-full px-4 h-full relative z-10">
                          {Array.from({length: 20}).map((_, i) => (
                             <div key={i} className="w-1 bg-amber-500/50 rounded-full" style={{ height: `${Math.max(20, Math.random() * 100)}%` }}></div>
                          ))}
                       </div>
                     </div>
                     <div className="mt-8 font-mono text-sm text-gray-400 border border-white/10 px-6 py-3 rounded bg-white/5">
                        {explainability.av_alignment?.interpretation}
                     </div>
                  </div>
                ) : (
                  <div className="py-12 text-center font-mono text-[0.65rem] text-white/40 tracking-widest">
                    A/V CORRELATION DATA UNAVAILABLE
                  </div>
                )}
             </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
