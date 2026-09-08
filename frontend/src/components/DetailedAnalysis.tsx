import { useState } from 'react';
import { ChevronDown, Activity, Eye, Mic, Info, Layers, GitCommit, FileText } from 'lucide-react';
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

function EvidenceTimeline() {
  const steps = [
    "MEDIA INGESTED",
    "VIDEO SIGNAL",
    "AUDIO SIGNAL",
    "MODEL SIGNAL",
    "VISUAL ANALYSIS",
    "MEDIADNA FUSION",
    "ASSESSMENT"
  ];
  
  return (
    <div className="flex flex-col items-center py-4">
      {steps.map((step, i) => (
        <div key={step} className="flex flex-col items-center">
          <div className="flex items-center gap-2 bg-black/60 border border-white/10 px-4 py-2 rounded-full shadow-lg">
            <GitCommit className="w-3 h-3 text-[#00e5ff]" />
            <span className="font-mono text-[0.65rem] font-bold tracking-[0.2em] text-white">{step}</span>
          </div>
          {i < steps.length - 1 && (
            <div className="w-px h-6 bg-gradient-to-b from-white/20 to-transparent my-1"></div>
          )}
        </div>
      ))}
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
            title="EVIDENCE TIMELINE" 
            icon={Layers}
            description="Conceptual evidence flow. Distinguish this from actual processing telemetry."
          >
            <EvidenceTimeline />
          </Section>
          
          <Section 
            title="MODEL SIGNAL" 
            icon={Activity}
            description="Learned multimodal representation produced by OpenAVFF."
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
              <div>
                <DataRow label="MODEL PREDICTION" value={result.prediction.toUpperCase()} highlight />
                <DataRow label="MODEL PROBABILITY" value={`${(result.openavff_fake_prob * 100).toFixed(4)}%`} highlight />
                <DataRow label="THRESHOLD" value="0.5" />
              </div>
              <div>
                <DataRow label="RAW LOGITS" value={`[${result.raw_logits.map(l => l.toFixed(4)).join(', ')}]`} />
                <DataRow label="INFERENCE TIME" value={`${result.inference_time.toFixed(3)}s`} />
                <DataRow label="MODEL PIPELINE" value={`${result.model} (${result.device})`} />
              </div>
            </div>
          </Section>

          {result.visual_signals && (
            <Section 
              title="VISUAL SIGNAL" 
              icon={Eye}
              description="Derived from measured visual consistency characteristics."
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
                <div>
                  <DataRow label="BLUR VARIANCE (MEAN)" value={result.visual_signals.blur_variance_mean.toFixed(4)} />
                  <DataRow label="BLUR VARIABILITY (CV)" value={result.visual_signals.blur_coefficient_of_variation.toFixed(4)} />
                </div>
                <div>
                  <DataRow label="INTER-FRAME MAE" value={result.visual_signals.frame_mae_mean.toFixed(4)} />
                  <DataRow label="TEMPORAL VOLATILITY" value={result.visual_signals.frame_mae_std.toFixed(4)} />
                  <DataRow label="FRAMES ANALYZED" value={result.visual_signals.frames_analyzed} />
                </div>
              </div>
            </Section>
          )}

          <Section 
            title="AUDIO SIGNAL" 
            icon={Mic}
            description="OpenAVFF incorporates acoustic representations through its learned audio pathway. Explicit acoustic heuristics are currently not exposed."
          >
            <div className="flex items-center justify-center py-4 opacity-50">
              <span className="font-mono text-xs tracking-widest">[ ACOUSTIC REPRESENTATION INTERNALIZED ]</span>
            </div>
          </Section>

          {result.metadata && (
            <Section 
              title="MEDIA METADATA" 
              icon={Info}
              description="Extracted structural and encoding information."
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
                <div>
                  <DataRow label="SOURCE" value={result.metadata.source || "Unknown"} />
                  <DataRow label="FILE SIZE" value={result.metadata.file_size || "Unknown"} />
                  <DataRow label="DURATION" value={result.metadata.duration || "Unknown"} />
                  <DataRow label="RESOLUTION" value={result.metadata.resolution || "Unknown"} />
                </div>
                <div>
                  <DataRow label="FRAMERATE" value={result.metadata.fps || "Unknown"} />
                  <DataRow label="VIDEO CODEC" value={result.metadata.video_codec || "Unknown"} />
                  <DataRow label="AUDIO CODEC" value={result.metadata.audio_codec || "Unknown"} />
                  <DataRow label="BITRATE" value={result.metadata.bitrate || "Unknown"} />
                </div>
              </div>
            </Section>
          )}

        </div>
      )}
    </div>
  );
}
