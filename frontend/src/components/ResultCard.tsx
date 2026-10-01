import { AlertTriangle, ShieldCheck, FileDown, FileText } from 'lucide-react';
import { useState } from 'react';
import { motion } from 'framer-motion';
import MediaDNACore from './MediaDNACore';
import DetailedAnalysis from './DetailedAnalysis';
import ExplainabilityDashboard from './ExplainabilityDashboard';
import type { AnalysisResponse, HistoryItem } from '../types';

interface ResultCardProps {
  result: AnalysisResponse;
  isArchived?: boolean;
}

export default function ResultCard({ result, isArchived }: ResultCardProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  
  // The final backend decision is stored in authenticity.uncertainty as AUTHENTIC / SYNTHETIC / UNCERTAIN
  const decision = result.authenticity?.uncertainty?.toUpperCase() || 'UNCERTAIN';
  
  let assessmentColor = '#f59e0b'; // Borderline / amber
  if (decision === 'SYNTHETIC') assessmentColor = '#ef4444';
  if (decision === 'AUTHENTIC') assessmentColor = '#10b981';
  
  const assessmentText = decision;
    
  const handleGeneratePDF = async () => {
    const caseId = (result as HistoryItem).id || String(Date.now());
    setIsGenerating(true);
    try {
      const response = await fetch(`http://localhost:8000/api/report/${caseId}`, {
        method: 'POST',
      });
      if (!response.ok) throw new Error('Failed to generate PDF');
      
      await response.json();
      window.open(`http://localhost:8000/api/report/${caseId}`, '_blank');
      
    } catch (err) {
      console.error(err);
      alert('Error generating PDF report. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };


  return (
    <div className="space-y-6">
      {/* ─── CLIMAX SEQUENCE HEADER ─── */}
      <div className="flex flex-col items-center justify-center mb-8 h-12 relative font-mono text-[0.65rem] font-bold tracking-[0.3em]">
        <div className="text-emerald-500">FORENSIC ASSESSMENT FINALIZED</div>
      </div>

      {/* ─── Primary Assessment ─── */}
      <div className="glass-card p-10 text-center relative overflow-hidden transition-all duration-1000 border rounded-2xl"
        style={{ borderColor: `${assessmentColor}80` }}>
        
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-full h-full opacity-10"
            style={{ background: `radial-gradient(circle at center, ${assessmentColor}, transparent 60%)` }}></div>
        </div>

        <div className="relative z-10 flex flex-col items-center">
          <div className="mb-8 relative transition-transform duration-1000" style={{ transform: 'scale(1)', opacity: 1 }}>
             <MediaDNACore className="w-48 h-48 mx-auto" isIgnited={true} />
             
             <motion.div 
               initial={{ opacity: 0, scale: 0.8 }}
               animate={{ opacity: 1, scale: 1 }}
               className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 backdrop-blur-md rounded-full border border-white/10"
             >
               <div className="font-mono text-[0.5rem] font-bold tracking-[0.2em] mb-1 text-white">PRIMARY FINDING</div>
               <span className="text-xl font-bold font-mono tracking-widest text-center px-4" style={{ color: assessmentColor, textShadow: `0 0 20px ${assessmentColor}80` }}>
                 {assessmentText}
               </span>
               <div className="font-mono text-[0.6rem] mt-2 text-white bg-white/10 px-2 py-0.5 rounded">
                 CONFIDENCE: {result.trust?.model_confidence || 'N/A'}
               </div>
             </motion.div>
          </div>

          <div className="w-full transition-opacity duration-1000" style={{ opacity: 1 }}>
            {isArchived && (
              <div className="font-mono text-[0.65rem] font-bold tracking-[0.3em] mb-4 text-[#ffb300]">ARCHIVED RECORD</div>
            )}
            <div className="flex flex-col items-center justify-center gap-2">
              <div className="flex items-center gap-4">
                {decision === 'SYNTHETIC' ? (
                  <AlertTriangle className="w-6 h-6" style={{ color: assessmentColor }} />
                ) : (
                  <ShieldCheck className="w-6 h-6" style={{ color: assessmentColor }} />
                )}
                <span className="text-2xl font-bold font-mono tracking-tight" style={{ color: assessmentColor }}>
                  {assessmentText}
                </span>
              </div>
              <div className="font-mono text-xs text-white/70 tracking-widest mt-2">HUMAN DETERMINATION: {result.human_determination.toUpperCase()}</div>
              <div className="font-mono text-[0.6rem] text-white/50 tracking-widest mt-1">NOTE: MODEL OUTPUT IS A STATISTICAL ESTIMATE, NOT A GUARANTEE.</div>
            </div>
          </div>
        </div>
      </div>

      <div className="transition-all duration-1000 opacity-100 translate-y-0">
        
        {/* Phase 10: SYSTEM LIMITATIONS */}
        <div className="bg-red-500/10 border border-red-500/20 p-5 rounded-xl flex items-start gap-4 mb-6">
          <AlertTriangle className="w-6 h-6 text-red-500 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-mono text-[0.7rem] font-bold tracking-widest text-red-400 mb-1">SYSTEM LIMITATIONS</h4>
            <p className="font-mono text-xs text-red-200/80 leading-relaxed mb-2">
              <strong>CRITICAL WARNING:</strong> Current benchmark performance is based on the validated research evaluation protocol and should not be represented as universal real-world accuracy.
            </p>
            <ul className="list-disc list-inside font-mono text-xs text-red-200/80">
              {result.limitations?.map((lim, i) => <li key={i}>{lim}</li>)}
            </ul>
          </div>
        </div>

        {result.explainability && <ExplainabilityDashboard result={result} />}

        <DetailedAnalysis result={result} />
        
        <div className="mt-8 flex flex-wrap justify-end gap-4">
          <button 
            onClick={handleGeneratePDF}
            disabled={isGenerating}
            className="btn-primary py-3 px-6 flex items-center gap-2"
            data-hover="node"
          >
            <FileText className="w-5 h-5" />
            {isGenerating ? 'GENERATING PDF...' : 'PREVIEW REPORT'}
          </button>
          
          <button 
            onClick={() => {
              const caseId = (result as HistoryItem).id || String(Date.now());
              window.open(`http://localhost:8000/api/report/${caseId}?download=true`, '_blank');
            }}
            className="btn-secondary py-3 px-6 flex items-center gap-2 bg-blue-500/10 text-blue-400 border border-blue-500/30 hover:bg-blue-500/20 rounded font-mono font-bold tracking-widest transition-colors"
          >
            <FileDown className="w-5 h-5" />
            DOWNLOAD PDF
          </button>
        </div>
        
      </div>
    </div>
  );
}
