import { Activity, ShieldAlert, Cpu } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Research() {
  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      
      <div className="mb-16">
        <div className="font-mono text-[0.65rem] font-bold tracking-[0.3em] text-[#00e5ff] mb-4">MEDIA DNA / EMPIRICAL DATA</div>
        <h1 className="editorial-headline tracking-tighter text-white mb-6" style={{ fontSize: 'clamp(2.5rem, 5vw, 4rem)', lineHeight: 1 }}>
          Experimental <span className="text-[#00e5ff]">Results</span>
        </h1>
        <p className="text-gray-400 max-w-2xl text-sm leading-relaxed font-mono">
          MediaDNA is built upon the OpenAVFF multimodal architecture. The results below reflect the official benchmark accuracy across the FakeAVCeleb dataset, alongside our experimental deterministic fusion approach.
        </p>
      </div>

      <div className="grid lg:grid-cols-2 gap-8 mb-16">
        {/* Core Baseline */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="glass-card p-8 rounded-xl border border-white/5 relative overflow-hidden group"
        >
          <div className="absolute top-0 right-0 p-6 opacity-10 group-hover:opacity-20 transition-opacity">
            <Cpu className="w-24 h-24 text-[#00e5ff]" />
          </div>
          <div className="relative z-10">
            <div className="font-mono text-[0.65rem] tracking-[0.2em] text-gray-500 mb-6">OPENAVFF / VIDEOCAVMAEFT</div>
            <h2 className="text-5xl font-bold text-white mb-2 font-mono">64.69<span className="text-2xl text-gray-500">%</span></h2>
            <div className="text-sm font-bold text-[#00e5ff] tracking-widest uppercase mb-6">Baseline Accuracy</div>
            
            <p className="text-sm text-gray-400 leading-relaxed max-w-[85%]">
              The underlying multimodal transformer achieves state-of-the-art representation fusion, effectively correlating acoustic phenomena with visual lip-sync and facial artifacts.
            </p>
          </div>
        </motion.div>

        {/* MediaDNA Fusion */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="glass-card p-8 rounded-xl border border-white/5 relative overflow-hidden group"
        >
          <div className="absolute top-0 right-0 p-6 opacity-10 group-hover:opacity-20 transition-opacity">
            <Activity className="w-24 h-24 text-[#b388ff]" />
          </div>
          <div className="relative z-10">
            <div className="font-mono text-[0.65rem] tracking-[0.2em] text-gray-500 mb-6">MEDIADNA / HEURISTIC ENSEMBLE</div>
            <h2 className="text-5xl font-bold text-white mb-2 font-mono">60.73<span className="text-2xl text-gray-500">%</span></h2>
            <div className="text-sm font-bold text-[#b388ff] tracking-widest uppercase mb-6">Experimental Fusion Accuracy</div>
            
            <p className="text-sm text-gray-400 leading-relaxed max-w-[85%]">
              Our experimental deterministic fusion layer incorporates Laplacian variance and metadata analysis. While increasing robustness against superficial tampering, the rigid heuristics slightly degrade overall performance on high-quality deepfakes.
            </p>
          </div>
        </motion.div>
      </div>

      {/* Critical Blindspot */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className="border border-red-500/30 bg-red-500/5 rounded-xl p-8 relative overflow-hidden"
      >
        <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_25%,rgba(239,68,68,0.03)_50%,transparent_75%,transparent_100%)] bg-[length:20px_20px] animate-[scanline_10s_linear_infinite]"></div>
        
        <div className="relative z-10 grid md:grid-cols-3 gap-8 items-center">
          <div className="md:col-span-2 space-y-4">
            <div className="flex items-center gap-3">
              <ShieldAlert className="w-5 h-5 text-red-500" />
              <div className="font-mono text-[0.65rem] font-bold tracking-[0.2em] text-red-500">KNOWN ARCHITECTURAL BLINDSPOT</div>
            </div>
            <h3 className="text-2xl font-bold text-white font-mono uppercase tracking-tight">FakeVideo / RealAudio (FV-RA)</h3>
            <p className="text-sm text-gray-300 leading-relaxed max-w-xl">
              The architecture heavily relies on audio-to-visual cross-attention. When presented with completely pristine, authentic audio paired with a synthetic video track (such as sophisticated face-swaps or Wav2Lip), the model drastically over-indexes on the authentic audio signal.
              <br/><br/>
              <span className="font-bold text-red-400">MediaDNA's deterministic fusion is experimental and did NOT solve this primary blindspot.</span>
            </p>
          </div>
          
          <div className="flex flex-col items-center justify-center p-6 bg-black/40 rounded-lg border border-red-500/20">
            <div className="font-mono text-[0.65rem] text-gray-500 mb-2 tracking-widest">FV-RA SUBSET ACCURACY</div>
            <div className="text-4xl font-bold text-red-500 font-mono mb-2">27.20%</div>
            <div className="text-xs text-gray-400 text-center font-mono uppercase">Vulnerable to high-fidelity visual manipulation over pristine audio</div>
          </div>
        </div>
      </motion.div>

    </div>
  );
}
