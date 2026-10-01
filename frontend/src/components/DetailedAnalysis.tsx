import { useState } from 'react';
import { ChevronDown, Activity, Eye, Layers, GitCommit, FileText, CheckCircle } from 'lucide-react';
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
                <DataRow label="DECISION SCORE" value={result.trust?.calibrated_score !== null && result.trust?.calibrated_score !== undefined ? `${(result.trust.calibrated_score * 100).toFixed(2)}%` : "N/A"} highlight />
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
