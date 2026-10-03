import { AlertTriangle, ShieldCheck, FileDown, FileText, ChevronDown, Copy, Check } from 'lucide-react';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import MediaDNACore from './MediaDNACore';
import ExplainabilityDashboard from './ExplainabilityDashboard';
import type { AnalysisResponse, HistoryItem } from '../types';

interface ResultCardProps {
  result: AnalysisResponse;
  isArchived?: boolean;
}

export default function ResultCard({ result, isArchived }: ResultCardProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [reportState, setReportState] = useState<'idle' | 'generating' | 'ready' | 'failed'>('idle');
  const [reportSteps, setReportSteps] = useState('');
  const [copied, setCopied] = useState(false);
  const [expandEvidence, setExpandEvidence] = useState(false);
  
  // The final backend decision is stored in authenticity.uncertainty as AUTHENTIC / SYNTHETIC / UNCERTAIN
  const decision = result.authenticity?.uncertainty?.toUpperCase() || 'UNCERTAIN';
  
  let assessmentColor = '#f59e0b'; // Borderline / amber
  if (decision === 'SYNTHETIC') assessmentColor = '#ef4444';
  if (decision === 'AUTHENTIC') assessmentColor = '#10b981';
  
  const assessmentText = decision;
  const rawScore = result.trust?.raw_model_score || result.trust?.calibrated_score || 0;
    
  const handleGeneratePDF = async (download: boolean = false) => {
    const caseId = (result as HistoryItem).id || (result as any).run_id || (result as any).case_id || String(Date.now());
    if (reportState === 'ready' && !download) {
      window.open(`http://localhost:8000/api/report/${caseId}`, '_blank');
      return;
    }
    
    setIsGenerating(true);
    setReportState('generating');
    setReportSteps('Preparing evidence...');
    
    try {
      setTimeout(() => setReportSteps('Rendering report...'), 600);
      setTimeout(() => setReportSteps('Finalizing PDF...'), 1200);
      
      const response = await fetch(`http://localhost:8000/api/report/${caseId}`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to generate PDF');
      
      await response.json();
      setReportState('ready');
      
      if (download) {
        window.open(`http://localhost:8000/api/report/${caseId}?download=true`, '_blank');
      } else {
        window.open(`http://localhost:8000/api/report/${caseId}`, '_blank');
      }
      
    } catch (err) {
      console.error(err);
      setReportState('failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(result.case_id || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-8">
      {/* ─── CLIMAX SEQUENCE HEADER ─── */}
      <div className="flex flex-col items-center justify-center mb-8 h-12 relative font-mono text-[0.65rem] font-bold tracking-[0.3em]">
        <div className="text-emerald-500 uppercase">Forensic Assessment</div>
      </div>

      {/* ─── Ultra-Premium Hero ─── */}
      <div className="glass-card p-12 text-center relative overflow-hidden transition-all duration-1000 border rounded-2xl"
        style={{ borderColor: `${assessmentColor}30` }}>
        
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <motion.div 
            animate={{ rotate: 360 }}
            transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
            className="w-[800px] h-[800px] opacity-10 rounded-full"
            style={{ background: `conic-gradient(from 0deg, transparent, ${assessmentColor}40, transparent)` }}
          ></motion.div>
          <div className="absolute inset-0 w-full h-full opacity-20"
            style={{ background: `radial-gradient(circle at center, ${assessmentColor}, transparent 50%)` }}></div>
        </div>

        <div className="relative z-10 flex flex-col items-center">
          <div className="mb-8 relative transition-transform duration-1000" style={{ transform: 'scale(1)', opacity: 1 }}>
             <MediaDNACore className="w-56 h-56 mx-auto" isIgnited={true} />
             
             <motion.div 
               initial={{ opacity: 0, scale: 0.8 }}
               animate={{ opacity: 1, scale: 1 }}
               transition={{ duration: 0.8, ease: "easeOut" }}
               className="absolute inset-0 flex flex-col items-center justify-center bg-black/70 backdrop-blur-md rounded-full border border-white/10 shadow-[inset_0_0_20px_rgba(255,255,255,0.05)]"
             >
               <div className="font-mono text-[0.55rem] font-bold tracking-[0.2em] mb-2 text-white/70">PRIMARY FINDING</div>
               <span className="text-3xl font-bold font-mono tracking-widest text-center px-4" style={{ color: assessmentColor, textShadow: `0 0 30px ${assessmentColor}` }}>
                 {assessmentText}
               </span>
               <div className="font-mono text-[0.65rem] mt-3 text-white/90 bg-white/5 px-3 py-1 rounded border border-white/10" title="Raw sigmoid model output used by the V22.4F operating policy. Not a calibrated probability.">
                 RAW SIGMOID DECISION SCORE: {rawScore.toFixed(4)}
               </div>
             </motion.div>
          </div>

          <div className="w-full max-w-3xl mx-auto transition-opacity duration-1000">
            {isArchived && (
              <div className="font-mono text-[0.65rem] font-bold tracking-[0.3em] mb-4 text-[#ffb300]">ARCHIVED RECORD</div>
            )}
            
            {/* ─── Score Band ─── */}
            <div className="mt-6 mb-12 px-8 relative">
              <div className="flex justify-between font-mono text-[0.55rem] text-white/40 tracking-wider mb-2">
                <span>0.00</span>
                <span>0.20</span>
                <span>0.45</span>
                <span>1.00</span>
              </div>
              <div className="relative h-[1px] bg-white/20 w-full mb-3 flex">
                 <div className="h-full bg-emerald-500/50" style={{ width: '20%' }}></div>
                 <div className="h-full bg-amber-500/50" style={{ width: '25%' }}></div>
                 <div className="h-full bg-red-500/50" style={{ width: '55%' }}></div>
                 
                 <div className="absolute top-0 bottom-0 left-[20%] w-[1px] h-3 -mt-1 bg-white/40"></div>
                 <div className="absolute top-0 bottom-0 left-[45%] w-[1px] h-3 -mt-1 bg-white/40"></div>
                 
                 <motion.div 
                   initial={{ left: 0, opacity: 0 }}
                   animate={{ left: `${Math.min(100, Math.max(0, rawScore * 100))}%`, opacity: 1 }}
                   transition={{ duration: 1.5, ease: "easeOut", delay: 0.5 }}
                   className="absolute top-0 -mt-[14px] flex flex-col items-center transform -translate-x-1/2"
                 >
                   <span className="text-[0.6rem] font-mono text-white mb-0.5">{rawScore.toFixed(4)}</span>
                   <div className="w-0 h-0 border-l-[4px] border-r-[4px] border-t-[6px] border-l-transparent border-r-transparent border-t-white"></div>
                 </motion.div>
              </div>
              <div className="flex justify-between font-mono text-[0.5rem] tracking-widest uppercase">
                <span className="text-emerald-500/70 w-[20%] text-center">AUTHENTIC</span>
                <span className="text-amber-500/70 w-[25%] text-center">UNCERTAIN</span>
                <span className="text-red-500/70 w-[55%] text-center">SYNTHETIC</span>
              </div>
            </div>

            <div className="flex flex-col items-center justify-center gap-2">
              <div className="font-mono text-[0.6rem] text-white/50 tracking-widest mt-1">NOTE: MODEL OUTPUT IS A STATISTICAL ESTIMATE, NOT A GUARANTEE.</div>
            </div>
          </div>
        </div>
      </div>

      {/* ─── Result Metric Strip ─── */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {[
          { label: 'VISUAL', value: result.visual?.anomaly_score, source: 'DIAGNOSTIC' },
          { label: 'AUDIO', value: result.audio?.anomaly_score, source: 'DIAGNOSTIC' },
          { label: 'MULTIMODAL', value: result.multimodal?.fusion_output, source: 'PRIMARY V22.4F', focus: true },
          { label: 'TEMPORAL', value: result.temporal?.anomaly_score, source: 'L2 SHIFT' }
        ].map((metric, i) => (
          <div key={i} className={`glass-card p-4 rounded-xl flex flex-col justify-between ${metric.focus ? 'ring-1 ring-cyan-500/30 bg-cyan-900/10' : ''}`}>
             <div className="font-mono text-[0.6rem] text-white/60 tracking-widest mb-3 uppercase">{metric.label}</div>
             <div className="font-mono text-xl font-bold text-white mb-2">
               {metric.value !== undefined && metric.value !== null ? metric.value.toFixed(4) : 'N/A'}
             </div>
             <div className="w-full h-[1px] bg-white/10 mb-2 overflow-hidden">
               {metric.value !== undefined && metric.value !== null && (
                 <div className="h-full bg-cyan-500/50" style={{ width: `${Math.min(100, Math.max(0, metric.value * 100))}%` }}></div>
               )}
             </div>
             <div className="font-mono text-[0.55rem] text-white/40 tracking-widest uppercase">{metric.source}</div>
          </div>
        ))}
      </div>

      {/* ─── Why This Result (Expandable) ─── */}
      <div className="glass-card rounded-xl overflow-hidden">
        <button 
          onClick={() => setExpandEvidence(!expandEvidence)}
          className="w-full p-5 flex items-center justify-between hover:bg-white/5 transition-colors"
        >
          <div className="flex flex-col items-start">
            <span className="font-mono text-xs font-bold tracking-widest text-white">WHY THIS RESULT</span>
            <span className="font-mono text-[0.65rem] text-white/50 mt-1">Evidence summary</span>
          </div>
          <ChevronDown className={`w-5 h-5 text-white/50 transition-transform duration-300 ${expandEvidence ? 'rotate-180' : ''}`} />
        </button>
        
        <AnimatePresence>
          {expandEvidence && (
            <motion.div 
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="border-t border-white/5"
            >
              <div className="p-5 grid grid-cols-2 md:grid-cols-3 gap-6">
                <div>
                  <div className="font-mono text-[0.6rem] text-white/40 mb-1">PRIMARY MODEL</div>
                  <div className="font-mono text-sm text-white">V22.4F / VideoCAVMAEFT</div>
                </div>
                <div>
                  <div className="font-mono text-[0.6rem] text-white/40 mb-1">RAW SCORE</div>
                  <div className="font-mono text-sm text-white">{rawScore.toFixed(4)}</div>
                </div>
                <div>
                  <div className="font-mono text-[0.6rem] text-white/40 mb-1">OPERATING STATE</div>
                  <div className="font-mono text-sm text-white" style={{ color: assessmentColor }}>{decision}</div>
                </div>
                <div>
                  <div className="font-mono text-[0.6rem] text-white/40 mb-1">VISUAL DIAGNOSTIC</div>
                  <div className="font-mono text-sm text-white">{result.visual?.anomaly_score?.toFixed(4) || 'UNAVAILABLE'}</div>
                </div>
                <div>
                  <div className="font-mono text-[0.6rem] text-white/40 mb-1">AUDIO DIAGNOSTIC</div>
                  <div className="font-mono text-sm text-white">{result.audio?.anomaly_score?.toFixed(4) || 'UNAVAILABLE'}</div>
                </div>
                <div>
                  <div className="font-mono text-[0.6rem] text-white/40 mb-1">TEMPORAL SIGNAL</div>
                  <div className="font-mono text-sm text-white">{result.temporal?.anomaly_score?.toFixed(4) || 'UNAVAILABLE'}</div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="transition-all duration-1000 opacity-100 translate-y-0">
        {/* Phase 10: SYSTEM LIMITATIONS */}
        <div className="bg-red-500/5 border border-red-500/20 p-5 rounded-xl flex items-start gap-4 mb-6 shadow-lg shadow-red-900/10">
          <AlertTriangle className="w-5 h-5 text-red-500/80 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-mono text-[0.7rem] font-bold tracking-widest text-red-400 mb-2">SYSTEM LIMITATIONS</h4>
            <p className="font-mono text-[0.65rem] text-red-200/80 leading-relaxed mb-3">
              <strong>CRITICAL WARNING:</strong> Current benchmark performance is based on the validated research evaluation protocol and should not be represented as universal real-world accuracy.
            </p>
            <ul className="list-disc list-inside font-mono text-[0.65rem] text-red-200/70 space-y-1">
              {result.limitations?.map((lim, i) => <li key={i}>{lim}</li>)}
            </ul>
          </div>
        </div>

        {result.explainability && <ExplainabilityDashboard result={result} />}

        {/* ─── Case Integrity ─── */}
        <div className="glass-card rounded-xl overflow-hidden p-6 mt-8">
          <div className="flex items-center justify-between mb-6">
            <h3 className="font-mono text-sm font-bold tracking-widest text-white flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-cyan-400" />
              CASE INTEGRITY
            </h3>
            <div className="flex items-center gap-2 font-mono text-[0.65rem]">
               <span className="text-white/40">STATUS</span>
               <span className="text-emerald-400 font-bold px-2 py-1 bg-emerald-900/20 rounded border border-emerald-500/20">COMPLETED</span>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-y-4 gap-x-8">
            <div className="flex flex-col">
              <span className="font-mono text-[0.6rem] text-white/40 mb-1">CASE ID</span>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs text-white truncate">{result.case_id}</span>
                <button onClick={handleCopy} className="text-white/30 hover:text-white transition-colors" title="Copy Case ID">
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>
            <div className="flex flex-col">
              <span className="font-mono text-[0.6rem] text-white/40 mb-1">ASSET ID</span>
              <span className="font-mono text-xs text-white truncate">{result.asset_id}</span>
            </div>
            <div className="flex flex-col">
              <span className="font-mono text-[0.6rem] text-white/40 mb-1">RUN ID</span>
              <span className="font-mono text-xs text-white truncate">{result.run_id}</span>
            </div>
            <div className="flex flex-col">
              <span className="font-mono text-[0.6rem] text-white/40 mb-1">SHA-256</span>
              <span className="font-mono text-[0.65rem] text-white/70 truncate">{result.asset_hash}</span>
            </div>
            <div className="flex flex-col">
              <span className="font-mono text-[0.6rem] text-white/40 mb-1">MODEL</span>
              <span className="font-mono text-xs text-white">{result.model_name} {result.model_version}</span>
            </div>
          </div>
        </div>
        
        {/* ─── PDF Action Area ─── */}
        <div className="glass-card rounded-xl overflow-hidden p-6 mt-6 flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-4">
             <div className="w-10 h-10 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center">
               <FileText className="w-5 h-5 text-cyan-400" />
             </div>
             <div>
               <div className="font-mono text-[0.65rem] font-bold tracking-[0.2em] text-white mb-1">
                 {reportState === 'generating' ? 'REPORT GENERATION' : reportState === 'failed' ? 'REPORT GENERATION FAILED' : 'FORENSIC REPORT'}
               </div>
               <div className="font-mono text-[0.65rem] text-white/50">
                 {reportState === 'generating' ? reportSteps : reportState === 'failed' ? 'Please try again.' : 'Generate a printable PDF dossier of this analysis.'}
               </div>
             </div>
          </div>
          
          <div className="flex flex-col items-end gap-3 w-full sm:w-auto">
            <div className="flex items-center gap-2 font-mono text-[0.55rem] tracking-widest px-2 py-1 rounded bg-black/40 border border-white/5">
              <span className="text-white/40">REPORT</span>
              {reportState === 'ready' ? (
                 <span className="text-emerald-400 flex items-center gap-1.5"><span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse"></span> READY</span>
              ) : (
                 <span className="text-white/30 flex items-center gap-1.5"><span className="w-1.5 h-1.5 border border-white/30 rounded-full"></span> NOT GENERATED</span>
              )}
            </div>
            <div className="flex gap-3 w-full sm:w-auto">
              <button 
                onClick={() => handleGeneratePDF(false)}
                disabled={isGenerating}
                className="btn-primary py-2.5 px-5 flex items-center justify-center gap-2 flex-1 sm:flex-none"
                data-hover="node"
              >
                {reportState === 'ready' ? '[ PREVIEW ]' : isGenerating ? 'GENERATING...' : 'GENERATE REPORT'}
              </button>
              
              {reportState === 'ready' && (
                <button 
                  onClick={() => handleGeneratePDF(true)}
                  className="btn-secondary py-2.5 px-5 flex items-center justify-center gap-2 flex-1 sm:flex-none bg-blue-500/5 text-blue-400 border border-blue-500/20 hover:bg-blue-500/10 hover:border-blue-500/40"
                >
                  <FileDown className="w-4 h-4" />
                  [ DOWNLOAD ]
                </button>
              )}
            </div>
          </div>
        </div>
        
      </div>
    </div>
  );
}
