import type { AnalysisResponse } from '../types';

interface ExplainabilityDashboardProps {
  result: AnalysisResponse;
}

export default function ExplainabilityDashboard({ result }: ExplainabilityDashboardProps) {
  const explainability = result.explainability;
  if (!explainability) return null;

  const narrativeMode = explainability.narrative?.mode || 'UNKNOWN';
  const narrativeText = explainability.narrative?.text || 'No narrative available.';
  const narrativeColor = narrativeMode === 'FAKE' ? 'text-red-400' :
                         narrativeMode === 'REAL' ? 'text-emerald-400' : 'text-amber-400';
  
  const narrativeBg = narrativeMode === 'FAKE' ? 'bg-red-500/10 border-red-500/30' :
                      narrativeMode === 'REAL' ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-amber-500/10 border-amber-500/30';

  return (
    <div className="space-y-6 mb-8 mt-8">
      <div className="flex flex-col items-center justify-center h-12 relative font-mono text-[0.65rem] font-bold tracking-[0.3em]">
        <div className="text-[#a5b4fc]">DEEP EXPLAINABILITY FORENSICS</div>
      </div>
      
      {/* Narrative Section */}
      <div className={`p-6 border rounded-xl ${narrativeBg}`}>
        <h3 className="font-mono text-xs font-bold tracking-widest text-white mb-4">FORENSIC NARRATIVE ({narrativeMode})</h3>
        <p className={`font-mono text-sm leading-relaxed ${narrativeColor}`}>
          {narrativeText}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        {/* Modality Sensitivity */}
        <div className="glass-card p-6 border border-white/10 rounded-xl flex flex-col justify-between">
          <h3 className="font-mono text-xs font-bold tracking-widest text-white mb-6">MODALITY SENSITIVITY</h3>
          <div className="flex justify-between items-center h-full pb-4">
            <div className="text-center">
              <div className="font-mono text-[0.65rem] text-gray-500 tracking-wider mb-2">VISUAL ONLY</div>
              <div className="font-mono text-2xl font-bold text-[#00e5ff]">{explainability.modality_sensitivity?.visual_only?.toFixed(2) || 'N/A'}</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-[0.65rem] text-gray-500 tracking-wider mb-2">AUDIO ONLY</div>
              <div className="font-mono text-2xl font-bold text-[#b388ff]">{explainability.modality_sensitivity?.audio_only?.toFixed(2) || 'N/A'}</div>
            </div>
            <div className="text-center">
              <div className="font-mono text-[0.65rem] text-gray-500 tracking-wider mb-2">MULTIMODAL</div>
              <div className="font-mono text-2xl font-bold text-white">{result.classification?.decision_score?.toFixed(2) || 'N/A'}</div>
            </div>
          </div>
        </div>

        {/* Audio Visual Alignment */}
        <div className="glass-card p-6 border border-white/10 rounded-xl">
          <h3 className="font-mono text-xs font-bold tracking-widest text-white mb-4">A/V CONSISTENCY</h3>
          <div className="mb-4">
            <div className="flex justify-between text-xs font-mono mb-1">
              <span className="text-gray-400">Correlation Score</span>
              <span className="text-emerald-400">{explainability.av_alignment?.correlation_score?.toFixed(2) || 'N/A'}</span>
            </div>
            <div className="w-full bg-gray-800 h-2 rounded-full overflow-hidden">
              <div 
                className="bg-emerald-500 h-full rounded-full" 
                style={{ width: `${(explainability.av_alignment?.correlation_score || 0) * 100}%` }}
              ></div>
            </div>
          </div>
          <p className="font-mono text-[0.75rem] text-gray-400 mt-4 leading-relaxed">
            {explainability.av_alignment?.interpretation || "A/V consistency interpretation not available."}
          </p>
        </div>

      </div>

      {/* Model-Sensitive Regions */}
      {explainability.region_sensitivity && explainability.region_sensitivity.length > 0 && (
        <div className="glass-card p-6 border border-white/10 rounded-xl">
          <h3 className="font-mono text-xs font-bold tracking-widest text-white mb-6">MODEL-SENSITIVE REGIONS (TOP)</h3>
          <div className="space-y-3">
            {explainability.region_sensitivity.slice(0, 3).map((region, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-white/5 rounded border border-white/5">
                <div className="flex items-center gap-4">
                  <div className="font-mono text-[0.6rem] text-gray-500">#{region.rank}</div>
                  <div className="font-mono text-sm text-gray-200">{region.region}</div>
                </div>
                <div className="flex items-center gap-6">
                  <div className="text-right hidden sm:block">
                    <div className="font-mono text-[0.6rem] text-gray-500">ORIGINAL</div>
                    <div className="font-mono text-xs text-white">{region.original_score.toFixed(3)}</div>
                  </div>
                  <div className="text-right hidden sm:block">
                    <div className="font-mono text-[0.6rem] text-gray-500">PERTURBED</div>
                    <div className="font-mono text-xs text-gray-400">{region.perturbed_score.toFixed(3)}</div>
                  </div>
                  <div className="text-right w-20">
                    <div className="font-mono text-[0.6rem] text-gray-500">IMPACT Δ</div>
                    <div className={`font-mono text-sm font-bold ${region.delta > 0.05 ? 'text-red-400' : region.delta < -0.05 ? 'text-emerald-400' : 'text-gray-400'}`}>
                      {region.delta > 0 ? '+' : ''}{region.delta.toFixed(3)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
