import os

content = """import { useState } from 'react';
import { ChevronDown, Activity, Eye, Mic, Layers, GitCommit, FileText, Cloud, CheckCircle, AlertTriangle } from 'lucide-react';
import type { AnalysisResponse } from '../types';

interface DetailedAnalysisProps {
  result: AnalysisResponse;
}

function Section({ title, icon: Icon, children, description }: { title: string, icon: any, children: React.ReactNode, description?: string }) {
  const [open, setOpen] = useState(true);
  
  return (
    <div className="rounded-xl border border-white/10 bg-black/40 overflow-hidden mb-4">
      <button 
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-4 bg-white/5 hover:bg-white/10 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Icon className="w-4 h-4 text-gray-400" />
          <span className="font-mono text-xs font-bold tracking-widest text-white">{title}</span>
        </div>
        <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      
      {open && (
        <div className="p-4 border-t border-white/5">
          {description && (
            <div className="mb-4 text-xs font-mono text-gray-400 border-l-2 border-white/20 pl-3 py-1">
              {description}
            </div>
          )}
          {children}
        </div>
      )}
    </div>
  );
}

function DataRow({ label, value, highlight = false }: { label: string, value: React.ReactNode, highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-white/5 last:border-0">
      <span className="text-[0.65rem] font-mono tracking-widest text-gray-500 font-bold uppercase">{label}</span>
      <span className={`text-xs font-mono ${highlight ? 'text-white font-bold' : 'text-gray-300'}`}>{value}</span>
    </div>
  );
}

export default function DetailedAnalysis({ result }: DetailedAnalysisProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="mt-8">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-center gap-2 py-3 border border-white/10 rounded-xl text-xs font-mono tracking-widest text-gray-400 hover:text-white hover:border-white/30 transition-all"
      >
        <FileText className="w-4 h-4" />
        {isOpen ? "HIDE DETAILED ANALYSIS" : "VIEW DETAILED ANALYSIS"}
      </button>

      {isOpen && (
        <div className="mt-6 space-y-2 animate-fade-in">
          
          <Section 
            title="CASE INTEGRITY" 
            icon={CheckCircle}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
              <div>
                <DataRow label="CASE ID" value={result.case_id} highlight />
                <DataRow label="ASSET ID" value={result.asset_id} />
                <DataRow label="RUN ID" value={result.run_id} />
              </div>
              <div>
                <DataRow label="SHA-256" value={result.asset_hash} highlight />
                <DataRow label="MODEL VERSION" value={`${result.model_name} ${result.model_version}`} />
                <DataRow label="PROCESSING STATUS" value={result.processing_status} />
              </div>
            </div>
          </Section>

          <Section 
            title="INPUT QUALITY" 
            icon={Eye}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
              <div>
                <DataRow label="RESOLUTION" value={result.quality?.resolution || 'N/A'} />
                <DataRow label="DURATION" value={`${result.quality?.duration?.toFixed(2) || 0}s`} />
                <DataRow label="FPS" value={result.quality?.frame_rate?.toFixed(2) || 'N/A'} />
              </div>
              <div>
                <DataRow label="AUDIO PRESENCE" value={result.quality?.audio_presence ? 'YES' : 'NO'} />
                <DataRow label="FINDINGS" value={result.quality?.findings?.length > 0 ? result.quality.findings.join(", ") : "NONE"} highlight />
              </div>
            </div>
          </Section>

          <Section 
            title="DETECTION SCORE & UNCERTAINTY" 
            icon={Activity}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
              <div>
                <DataRow label="RAW MODEL SCORE" value={result.trust?.raw_model_score?.toFixed(4) || "N/A"} highlight />
                <DataRow label="CALIBRATED PROBABILITY" value={result.trust?.calibrated_probability !== null ? `${(result.trust.calibrated_probability! * 100).toFixed(2)}%` : "N/A"} highlight />
                <DataRow label="ABSTENTION STATE" value={result.trust?.abstention_state || "N/A"} />
              </div>
              <div>
                <DataRow label="CALIBRATION STATUS" value={result.trust?.calibration_status || "N/A"} />
                <DataRow label="MODEL CONFIDENCE" value={result.trust?.model_confidence || "N/A"} />
                <DataRow label="OOD STATUS" value={result.trust?.ood_status || "N/A"} />
              </div>
            </div>
          </Section>

          <Section 
            title="ATTRIBUTION EVIDENCE" 
            icon={Layers}
            description="Model-sensitive regions and artifacts"
          >
            {result.evidence_items?.map((ev, idx) => (
              <div key={idx} className="mb-4 border-b border-white/10 pb-4 last:border-0 last:pb-0">
                 <h4 className="font-mono text-sm font-bold tracking-widest text-white mb-2">{ev.type.replace('_', ' ').toUpperCase()} ({ev.modality.toUpperCase()})</h4>
                 <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
                   <div>
                     <DataRow label="VALUE" value={`${ev.value.toFixed(4)} ${ev.unit}`} />
                     <DataRow label="METHOD" value={ev.method} />
                   </div>
                   <div>
                     <DataRow label="RELIABILITY" value={ev.reliability} />
                     <DataRow label="CALIBRATION" value={ev.calibration_status} />
                   </div>
                 </div>
                 <div className="mt-2 text-xs font-mono text-gray-400">
                    <strong>Interpretation:</strong> {ev.interpretation}<br/>
                    <strong>Limitations:</strong> {ev.limitations}
                 </div>
              </div>
            ))}
          </Section>

          <Section 
            title="PROVENANCE" 
            icon={GitCommit}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
              <div>
                <DataRow label="BROAD FAMILY" value={result.provenance_v20?.broad_manipulation_family || "N/A"} highlight />
                <DataRow label="HEURISTIC" value={result.provenance_v20?.heuristic_provenance || "N/A"} />
                <DataRow label="EXACT GENERATOR" value={result.provenance_v20?.exact_generator_attribution || "N/A"} />
              </div>
              <div>
                <DataRow label="METADATA" value={result.provenance_v20?.metadata_provenance || "N/A"} />
                <DataRow label="CRYPTOGRAPHIC" value={result.provenance_v20?.cryptographic_provenance || "N/A"} />
                <DataRow label="SOURCE DEVICE" value={result.provenance_v20?.source_device_clues || "N/A"} />
              </div>
            </div>
          </Section>

        </div>
      )}
    </div>
  );
}
"""

with open('c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/frontend/src/components/DetailedAnalysis.tsx', 'w') as f:
    f.write(content)
print("Updated DetailedAnalysis.tsx")

content2 = """import { AlertTriangle, ShieldCheck, FileDown, FileText } from 'lucide-react';
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
  
  // Phase 10: PRIMARY FINDING
  const isFake = result.model_finding.toUpperCase().includes('MANIPULATED');
  const isAuthentic = result.model_finding.toUpperCase().includes('AUTHENTIC');
  const isBorderline = !isFake && !isAuthentic;
  
  let assessmentColor = '#f59e0b'; // Borderline / amber
  if (isFake) assessmentColor = '#ef4444';
  if (isAuthentic) assessmentColor = '#10b981';
  
  const assessmentText = result.model_finding.toUpperCase();
    
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
                {isFake ? (
                  <AlertTriangle className="w-6 h-6" style={{ color: assessmentColor }} />
                ) : (
                  <ShieldCheck className="w-6 h-6" style={{ color: assessmentColor }} />
                )}
                <span className="text-2xl font-bold font-mono tracking-tight" style={{ color: assessmentColor }}>
                  {assessmentText}
                </span>
              </div>
              <div className="font-mono text-xs text-white/70 tracking-widest">HUMAN DETERMINATION: {result.human_determination.toUpperCase()}</div>
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
"""
with open('c:/Users/navee/Desktop/Projects/MediaDna/OpenAVFF/frontend/src/components/ResultCard.tsx', 'w', encoding='utf-8') as f:
    f.write(content2)
print("Updated ResultCard.tsx")
