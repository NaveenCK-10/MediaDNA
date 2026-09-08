import { AlertTriangle, ShieldCheck } from 'lucide-react';
import { motion } from 'framer-motion';
import MediaDNACore from './MediaDNACore';
import DetailedAnalysis from './DetailedAnalysis';
import type { AnalysisResponse } from '../types';

interface ResultCardProps {
  result: AnalysisResponse;
  isArchived?: boolean;
}

function SignalBar({ label, value, color }: { label: string, value: number, color: string }) {
  const pct = Math.min(Math.max(value * 100, 0), 100);
  return (
    <div className="mb-4 last:mb-0">
      <div className="flex justify-between items-center mb-1">
        <span className="font-mono text-[0.65rem] font-bold tracking-[0.2em] text-gray-400">{label}</span>
        <span className="font-mono text-xs font-bold text-white">{pct.toFixed(1)}%</span>
      </div>
      <div className="w-full h-2 rounded-full relative overflow-hidden bg-white/10">
        <div 
          className="absolute top-0 left-0 h-full rounded-full transition-all duration-1000 ease-out" 
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

export default function ResultCard({ result, isArchived }: ResultCardProps) {
  const isFake = result.prediction === 'fake';
  const assessmentColor = isFake ? '#ef4444' : '#10b981';
  const assessmentText = isFake ? 'SYNTHETIC / MANIPULATED' : 'AUTHENTIC / PRISTINE';
  
  const fusionScore = result.visual_signals
    ? (result.openavff_fake_prob * 0.8 + result.visual_signals.visual_anomaly_score * 0.2)
    : result.openavff_fake_prob;

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
               <div className="font-mono text-[0.6rem] font-bold tracking-[0.2em] mb-1" style={{ color: assessmentColor }}>FINAL ASSESSMENT</div>
               <span className="text-3xl font-bold font-mono tracking-widest" style={{ color: assessmentColor, textShadow: `0 0 20px ${assessmentColor}80` }}>
                 {isFake ? 'FAKE' : 'REAL'}
               </span>
               <div className="font-mono text-[0.6rem] mt-2 text-white bg-white/10 px-2 py-0.5 rounded">{(fusionScore * 100).toFixed(1)}% SIGNAL</div>
             </motion.div>
          </div>

          <div className="w-full transition-opacity duration-1000" style={{ opacity: 1 }}>
            {isArchived && (
              <div className="font-mono text-[0.65rem] font-bold tracking-[0.3em] mb-4 text-[#ffb300]">ARCHIVED RECORD</div>
            )}
            <div className="flex items-center justify-center gap-4">
              {isFake ? (
                <AlertTriangle className="w-6 h-6" style={{ color: assessmentColor }} />
              ) : (
                <ShieldCheck className="w-6 h-6" style={{ color: assessmentColor }} />
              )}
              <span className="text-2xl font-bold font-mono tracking-tight" style={{ color: assessmentColor }}>
                {assessmentText}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ─── Signal Comparison & Limitations (only shown when complete) ─── */}
      <div className="transition-all duration-1000 opacity-100 translate-y-0">
        
        <div className="glass-card p-6 border border-white/10 rounded-xl mb-6">
          <h3 className="font-mono text-xs font-bold tracking-widest text-white mb-6">SIGNAL COMPARISON</h3>
          <SignalBar label="MODEL SIGNAL" value={result.openavff_fake_prob} color="#ef4444" />
          <SignalBar label="VISUAL SIGNAL" value={result.visual_signals?.visual_anomaly_score || 0} color="#00e5ff" />
          <SignalBar label="MEDIADNA FUSION" value={fusionScore} color="#b388ff" />
        </div>

        <div className="bg-red-500/10 border border-red-500/20 p-5 rounded-xl flex items-start gap-4">
          <AlertTriangle className="w-6 h-6 text-red-500 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-mono text-[0.7rem] font-bold tracking-widest text-red-400 mb-1">INTERPRETATION LIMITATIONS</h4>
            <p className="font-mono text-xs text-red-200/80 leading-relaxed">
              <strong>FakeVideo + RealAudio Blindspot:</strong> The OpenAVFF model is highly vulnerable to authentic audio injected over manipulated video. If the audio stream is perfectly pristine, it can overwhelm visual anomalies and produce a false "REAL" assessment. Always verify cross-modality signals.
            </p>
          </div>
        </div>

        <DetailedAnalysis result={result} />
        
        <div className="mt-8 flex justify-end">
          <button 
            onClick={() => {
              const report = `MEDIA DNA FORENSIC REPORT
==================================================
Case ID: ${isArchived ? 'ARCHIVED' : Date.now()}
Target: ${result.video_filename}
Timestamp: ${new Date().toISOString()}

FINAL ASSESSMENT: ${isFake ? 'SYNTHETIC / MANIPULATED' : 'AUTHENTIC / PRISTINE'}
Fusion Signal Score: ${(fusionScore * 100).toFixed(1)}%

--- SIGNAL COMPARISON ---
Model Signal (OpenAVFF): ${(result.openavff_fake_prob * 100).toFixed(1)}%
Visual Signal (Spatial Anomaly): ${((result.visual_signals?.visual_anomaly_score || 0) * 100).toFixed(1)}%
MediaDNA Fusion: ${(fusionScore * 100).toFixed(1)}%

--- METADATA ---
Duration: ${result.metadata?.duration || 'Unknown'}s
Dimensions: ${result.metadata?.resolution || 'Unknown'}
FPS: ${result.metadata?.fps || 'Unknown'}
Source: ${result.metadata?.source || 'Unknown'}
Size: ${result.metadata?.file_size || 'Unknown'} bytes

--- LIMITATIONS ---
FakeVideo + RealAudio Blindspot: The OpenAVFF model is highly vulnerable to authentic audio injected over manipulated video. If the audio stream is perfectly pristine, it can overwhelm visual anomalies and produce a false "REAL" assessment. Always verify cross-modality signals.
==================================================
GENERATED BY MEDIA DNA FORENSIC INSTRUMENT`;
              
              const blob = new Blob([report], { type: 'text/plain' });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `MediaDNA_Report_${result.video_filename}.txt`;
              document.body.appendChild(a);
              a.click();
              document.body.removeChild(a);
              URL.revokeObjectURL(url);
            }}
            className="btn-primary py-3 px-6 flex items-center gap-2"
            data-hover="node"
          >
            EXPORT FORENSIC REPORT
          </button>
        </div>
        
      </div>
    </div>
  );
}
