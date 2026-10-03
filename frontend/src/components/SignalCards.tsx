import { Eye, Mic, Clock } from 'lucide-react';
import { motion } from 'framer-motion';

export default function SignalCards() {
  return (
    <div className="max-w-7xl mx-auto pt-24 pb-12 px-6 lg:px-12">
      {/* Editorial Text */}
      <div className="mb-24 flex flex-col items-center justify-center text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 1 }}
          className="editorial-headline text-white tracking-tighter uppercase"
          style={{ fontSize: 'clamp(2rem, 4vw, 3rem)', lineHeight: 1.1 }}
        >
          <span className="opacity-50">One file.</span><br />
          <span className="opacity-75">Multiple signals.</span><br />
          <span className="text-[#00e5ff]">One forensic view.</span>
        </motion.div>
      </div>

      {/* Cards */}
      <div className="grid md:grid-cols-3 gap-6 relative">
        <div className="absolute top-1/2 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-[#00e5ff]/20 to-transparent -translate-y-1/2 -z-10" />
        
        {/* ======================================= */}
        {/* VISUAL CHANNEL */}
        {/* ======================================= */}
        <motion.div 
          whileHover="hover"
          initial="initial"
          animate="animate"
          className="glass-card flex flex-col rounded-xl overflow-hidden group border border-white/10 hover:border-[#00e5ff]/50 transition-all duration-700 bg-black/40 backdrop-blur-xl relative min-h-[400px]"
        >
          <div className="absolute top-0 right-0 w-32 h-32 bg-[#00e5ff]/5 blur-[50px] group-hover:bg-[#00e5ff]/20 transition-colors duration-700" />
          
          {/* INTERNAL VISUALIZATION */}
          <div className="absolute inset-0 z-0 overflow-hidden opacity-30 group-hover:opacity-100 transition-opacity duration-1000">
            {/* Grid */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(0,229,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(0,229,255,0.05)_1px,transparent_1px)] bg-[size:20px_20px]" />
            
            {/* Scanner */}
            <motion.div 
              variants={{
                animate: { y: ['0%', '100%', '0%'] },
                hover: { y: ['0%', '100%', '0%'], transition: { duration: 2, ease: "linear", repeat: Infinity } }
              }}
              transition={{ duration: 4, ease: "linear", repeat: Infinity }}
              className="absolute top-0 left-0 right-0 h-1 bg-[#00e5ff]/50 shadow-[0_0_15px_#00e5ff]"
            />
            
            {/* Frame Structure & Signal Points */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 border border-[#00e5ff]/20 flex items-center justify-center">
              <motion.div
                variants={{
                  animate: { scale: [1, 1.05, 1], opacity: [0.5, 0.8, 0.5] },
                  hover: { scale: [1, 1.1, 1], opacity: [0.8, 1, 0.8], transition: { duration: 1, repeat: Infinity } }
                }}
                transition={{ duration: 3, repeat: Infinity }}
                className="w-16 h-16 border border-[#00e5ff]/40 rotate-45"
              />
              {/* Signal points */}
              <motion.div 
                variants={{ animate: { opacity: [0, 1, 0] } }} 
                transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
                className="absolute top-2 left-2 w-1.5 h-1.5 bg-[#00e5ff]" 
              />
              <motion.div 
                variants={{ animate: { opacity: [0, 1, 0] } }} 
                transition={{ duration: 2, repeat: Infinity, delay: 1.5 }}
                className="absolute bottom-2 right-2 w-1.5 h-1.5 bg-[#00e5ff]" 
              />
            </div>
          </div>

          <div className="p-8 pb-0 flex-1 relative z-10">
            <div className="font-mono text-[0.6rem] text-[#00e5ff] tracking-[0.3em] font-bold mb-8">VISUAL CHANNEL</div>
            <Eye className="w-8 h-8 text-white/50 group-hover:text-white transition-colors duration-500 mb-6 drop-shadow-[0_0_8px_rgba(0,229,255,0)] group-hover:drop-shadow-[0_0_8px_rgba(0,229,255,0.5)]" strokeWidth={1} />
            <h3 className="font-mono text-xl font-bold tracking-widest text-white mb-4 uppercase">Spatial Evidence</h3>
            <p className="text-xs text-gray-400 font-mono leading-relaxed pb-8">
              Micro-level inspection of spatial artifacts, compression inconsistencies, and blending boundaries in visual geometry.
            </p>
          </div>
        </motion.div>

        {/* ======================================= */}
        {/* AUDIO CHANNEL */}
        {/* ======================================= */}
        <motion.div 
          whileHover="hover"
          initial="initial"
          animate="animate"
          className="glass-card flex flex-col rounded-xl overflow-hidden group border border-white/10 hover:border-[#b388ff]/50 transition-all duration-700 bg-black/40 backdrop-blur-xl relative min-h-[400px]"
        >
          <div className="absolute top-0 right-0 w-32 h-32 bg-[#b388ff]/5 blur-[50px] group-hover:bg-[#b388ff]/20 transition-colors duration-700" />
          
          {/* INTERNAL VISUALIZATION */}
          <div className="absolute inset-0 z-0 overflow-hidden flex items-end justify-center pb-12 opacity-30 group-hover:opacity-100 transition-opacity duration-1000">
            <div className="w-full px-8 flex items-end justify-between h-32 gap-[2px]">
              {[...Array(40)].map((_, i) => (
                <motion.div 
                  key={i}
                  variants={{
                    animate: { 
                      height: [`${Math.random() * 20 + 5}%`, `${Math.random() * 50 + 10}%`, `${Math.random() * 20 + 5}%`] 
                    },
                    hover: { 
                      height: [`${Math.random() * 40 + 10}%`, `${Math.random() * 90 + 20}%`, `${Math.random() * 40 + 10}%`],
                      backgroundColor: ['rgba(179,136,255,0.3)', 'rgba(179,136,255,0.8)', 'rgba(179,136,255,0.3)'],
                      transition: { duration: Math.random() * 0.5 + 0.3, repeat: Infinity, ease: "easeInOut" } 
                    }
                  }}
                  transition={{ duration: Math.random() * 1.5 + 1, repeat: Infinity, ease: "easeInOut" }}
                  className="w-full bg-[#b388ff]/30 rounded-t-sm"
                />
              ))}
            </div>
            
            {/* Traveling Signal */}
            <motion.div 
              variants={{
                animate: { x: ['-100%', '400%'] },
                hover: { x: ['-100%', '400%'], transition: { duration: 1.5, repeat: Infinity, ease: "linear" } }
              }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              className="absolute bottom-12 left-0 w-1/4 h-[2px] bg-[#b388ff] shadow-[0_0_10px_#b388ff]"
            />
          </div>

          <div className="p-8 pb-0 flex-1 relative z-10">
            <div className="font-mono text-[0.6rem] text-[#b388ff] tracking-[0.3em] font-bold mb-8">AUDIO CHANNEL</div>
            <Mic className="w-8 h-8 text-white/50 group-hover:text-white transition-colors duration-500 mb-6 drop-shadow-[0_0_8px_rgba(179,136,255,0)] group-hover:drop-shadow-[0_0_8px_rgba(179,136,255,0.5)]" strokeWidth={1} />
            <h3 className="font-mono text-xl font-bold tracking-widest text-white mb-4 uppercase">Spectral Trace</h3>
            <p className="text-xs text-gray-400 font-mono leading-relaxed pb-8">
              Frequency-domain analysis revealing vocoder artifacts, synthetic generation signatures, and acoustic inconsistencies.
            </p>
          </div>
        </motion.div>

        {/* ======================================= */}
        {/* TEMPORAL CONTINUITY */}
        {/* ======================================= */}
        <motion.div 
          whileHover="hover"
          initial="initial"
          animate="animate"
          className="glass-card flex flex-col rounded-xl overflow-hidden group border border-white/10 hover:border-white/50 transition-all duration-700 bg-black/40 backdrop-blur-xl relative min-h-[400px]"
        >
          <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 blur-[50px] group-hover:bg-white/10 transition-colors duration-700" />
          
          {/* INTERNAL VISUALIZATION */}
          <div className="absolute inset-0 z-0 overflow-hidden flex flex-col justify-end pb-12 opacity-30 group-hover:opacity-100 transition-opacity duration-1000">
            {/* Timeline track */}
            <div className="relative w-full h-8 border-t border-b border-white/10 px-4">
              <motion.div 
                variants={{
                  animate: { x: ['0%', '-50%'] },
                  hover: { x: ['0%', '-50%'], transition: { duration: 4, ease: "linear", repeat: Infinity } }
                }}
                transition={{ duration: 10, ease: "linear", repeat: Infinity }}
                className="absolute inset-y-0 left-0 w-[200%] flex items-center gap-4"
              >
                {[...Array(20)].map((_, i) => (
                  <div key={i} className="flex flex-col gap-1 items-center">
                    <div className="w-8 h-1 bg-white/10" />
                    {/* Tick mark */}
                    <div className="w-px h-2 bg-white/30" />
                  </div>
                ))}
              </motion.div>
              
              {/* Scan Marker */}
              <div className="absolute top-0 bottom-0 left-1/2 w-px bg-white/50 shadow-[0_0_10px_rgba(255,255,255,0.5)] z-10" />
            </div>
            
            {/* Sequential Frame Activation */}
            <div className="absolute bottom-24 left-1/2 -translate-x-1/2 flex gap-2">
              {[0, 1, 2].map((i) => (
                <motion.div 
                  key={i}
                  variants={{
                    animate: { opacity: [0.2, 0.2, 0.2] },
                    hover: { 
                      opacity: [0.2, 1, 0.2], 
                      scale: [1, 1.1, 1],
                      transition: { duration: 1.5, repeat: Infinity, delay: i * 0.5 }
                    }
                  }}
                  className="w-10 h-6 border border-white/20 bg-white/5"
                />
              ))}
            </div>
          </div>

          <div className="p-8 pb-0 flex-1 relative z-10">
            <div className="font-mono text-[0.6rem] text-white/70 tracking-[0.3em] font-bold mb-8">TEMPORAL CONTINUITY</div>
            <Clock className="w-8 h-8 text-white/50 group-hover:text-white transition-colors duration-500 mb-6 drop-shadow-[0_0_8px_rgba(255,255,255,0)] group-hover:drop-shadow-[0_0_8px_rgba(255,255,255,0.5)]" strokeWidth={1} />
            <h3 className="font-mono text-xl font-bold tracking-widest text-white mb-4 uppercase">Sequence Sync</h3>
            <p className="text-xs text-gray-400 font-mono leading-relaxed pb-8">
              Cross-modal temporal alignment detecting desynchronization between acoustic phonemes and visual visemes.
            </p>
          </div>
        </motion.div>

      </div>
    </div>
  );
}
