import { useState, useEffect } from 'react';
import { motion, useMotionValue, useSpring, useTransform, AnimatePresence } from 'framer-motion';

const ENTRANCE_STATES = [
  'VARIANT_01', 'VARIANT_02', 'VARIANT_03', 'VARIANT_04', 'VARIANT_05', 'VARIANT_06'
];

export default function MediaDNAEngine() {
  const [entranceState, setEntranceState] = useState<string>('VARIANT_06');
  const [mounted, setMounted] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);

  // Sync Event Loop
  useEffect(() => {
    const syncInterval = setInterval(() => {
      setIsSyncing(true);
      setTimeout(() => setIsSyncing(false), 800);
    }, 12000); // Every 12 seconds
    return () => clearInterval(syncInterval);
  }, []);

  // Parallax setup
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  const springConfig = { damping: 50, stiffness: 100 };
  const smoothX = useSpring(mouseX, springConfig);
  const smoothY = useSpring(mouseY, springConfig);

  // Layer parallax transforms
  const bgX = useTransform(smoothX, [-500, 500], [-2, 2]);
  const bgY = useTransform(smoothY, [-500, 500], [-2, 2]);
  
  const outerX = useTransform(smoothX, [-500, 500], [-4, 4]);
  const outerY = useTransform(smoothY, [-500, 500], [-4, 4]);

  const middleX = useTransform(smoothX, [-500, 500], [-6, 6]);
  const middleY = useTransform(smoothY, [-500, 500], [-6, 6]);

  const coreX = useTransform(smoothX, [-500, 500], [-10, 10]);
  const coreY = useTransform(smoothY, [-500, 500], [-10, 10]);

  useEffect(() => {
    const randomState = ENTRANCE_STATES[Math.floor(Math.random() * ENTRANCE_STATES.length)];
    setEntranceState(randomState);
    setMounted(true);
  }, []);

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    mouseX.set(e.clientX - centerX);
    mouseY.set(e.clientY - centerY);
  };

  const handleMouseLeave = () => {
    mouseX.set(0);
    mouseY.set(0);
  };

  if (!mounted) return null;

  // Delays based on variant
  const getDelay = (layer: string) => {
    if (entranceState === 'VARIANT_06') return layer === 'core' ? 0.9 : layer === 'orbits' ? 0.5 : layer === 'signals' ? 1.1 : 0.7;
    if (entranceState === 'VARIANT_01') return layer === 'orbits' ? 0 : layer === 'core' ? 0.8 : layer === 'signals' ? 1.5 : 1.0;
    if (entranceState === 'VARIANT_02') return layer === 'signals' ? 0 : layer === 'core' ? 0.8 : layer === 'orbits' ? 1.5 : 1.0;
    if (entranceState === 'VARIANT_03') return layer === 'signals' ? 0 : layer === 'orbits' ? 0.8 : layer === 'core' ? 1.5 : 1.0;
    if (entranceState === 'VARIANT_04') return layer === 'orbits' ? 0 : layer === 'orbits' ? 0.8 : layer === 'core' ? 1.5 : 1.0;
    if (entranceState === 'VARIANT_05') return layer === 'core' ? 0 : layer === 'signals' ? 0.8 : layer === 'orbits' ? 1.5 : 1.0;
    return 0;
  };

  return (
    <div 
      className="relative w-full h-[450px] lg:h-[650px] flex items-center justify-center group"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {/* ========================================================= */}
      {/* 17. BACKGROUND ATMOSPHERE & 19. SCAN GRID */}
      {/* ========================================================= */}
      <motion.div 
        style={{ x: bgX, y: bgY }}
        className="absolute inset-0 pointer-events-none"
      >
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(0,229,255,0.06)_0%,transparent_40%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_40%_60%,rgba(139,92,255,0.04)_0%,transparent_30%)]" />
        
        {/* Subtle Grid */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(0,229,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(0,229,255,0.02)_1px,transparent_1px)] bg-[size:40px_40px] opacity-50" />
        
        {/* Periodic Grid Scans */}
        <motion.div 
          animate={{ y: ['-10%', '110%'] }}
          transition={{ duration: 6, ease: "linear", repeat: Infinity, repeatDelay: 4 }}
          className="absolute left-0 right-0 h-32 bg-gradient-to-b from-transparent via-[#00e5ff]/5 to-transparent"
        />
        <motion.div 
          animate={{ x: ['-10%', '110%'] }}
          transition={{ duration: 8, ease: "linear", repeat: Infinity, repeatDelay: 2 }}
          className="absolute top-0 bottom-0 w-32 bg-gradient-to-r from-transparent via-[#2f7cff]/5 to-transparent"
        />
      </motion.div>

      {/* Breathing Container */}
      <motion.div
        animate={{ scale: [1, 1.01, 1] }}
        transition={{ duration: 6, ease: "easeInOut", repeat: Infinity }}
        className="relative w-full h-full flex items-center justify-center pointer-events-none"
      >
        
        {/* ========================================================= */}
        {/* 5. OUTER ORBIT (LAYER 01 & 02) + 6. ORBITING PARTICLES */}
        {/* ========================================================= */}
        <motion.div 
          style={{ x: outerX, y: outerY }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.5, delay: getDelay('orbits'), ease: "easeOut" }}
          className="absolute w-[380px] h-[380px] lg:w-[540px] lg:h-[540px] flex items-center justify-center"
        >
          {/* Layer 01: Large cyan outer ring (Clockwise) */}
          <motion.div 
            animate={{ rotate: 360 }}
            transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
            className="absolute inset-0 rounded-full border-[1px] border-[#00e5ff]/10 border-t-[#00e5ff]/40 shadow-[0_0_15px_rgba(0,229,255,0.05)]"
          >
            {/* Travelling Bright Segment (Signal Orbit) */}
            <motion.div 
              animate={{ rotate: 360 }}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
              className="absolute inset-0 rounded-full border-t-[2px] border-transparent border-t-[#00e5ff] blur-[2px] opacity-70"
            />
          </motion.div>

          {/* Layer 02: Dashed counter-clockwise orbit */}
          <motion.div 
            animate={{ rotate: -360 }}
            transition={{ duration: 45, repeat: Infinity, ease: "linear" }}
            className="absolute inset-4 rounded-full border-[1px] border-dashed border-[#2f7cff]/20"
          >
            {/* Orbiting Neon Particles */}
            <div className="absolute top-0 left-1/2 w-1.5 h-1.5 bg-[#00e5ff] rounded-full shadow-[0_0_8px_#00e5ff] -translate-x-1/2 -translate-y-1/2" />
            <div className="absolute bottom-1/4 right-0 w-1 h-1 bg-[#8b5cff] rounded-full shadow-[0_0_6px_#8b5cff]" />
            <div className="absolute top-1/4 left-0 w-1 h-1 bg-[#b7ff35] rounded-full shadow-[0_0_5px_#b7ff35] opacity-50" />
          </motion.div>

          <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-[#00e5ff]/60 font-mono text-[0.45rem] tracking-[0.4em]">SIGNAL_ORBIT</div>
        </motion.div>


        {/* ========================================================= */}
        {/* 8. AUDIO WAVE + 9. TEMPORAL PATH + 10. VISUAL SIGNAL */}
        {/* ========================================================= */}
        <motion.div 
          style={{ x: middleX, y: middleY }}
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.5, delay: getDelay('signals'), ease: "easeOut" }}
          className="absolute w-[280px] h-[280px] lg:w-[380px] lg:h-[380px] flex items-center justify-center"
        >
          {/* Angular Geometric Orbit (Layer 03) */}
          <motion.div 
            animate={{ rotate: 360 }}
            transition={{ duration: 80, repeat: Infinity, ease: "linear" }}
            className="absolute inset-0 border-[1px] border-[#00e5ff]/10 rotate-12"
          />

          {/* 8. AUDIO WAVE (Violet) */}
          <div className="absolute w-[80%] h-[80%] border-[1px] border-[#8b5cff]/10 rounded-full flex items-center justify-center -translate-y-8 translate-x-8">
            <svg className="absolute w-full h-full animate-[spin_40s_linear_infinite_reverse]" viewBox="0 0 100 100">
              <motion.path
                animate={{
                  d: [
                    "M10,50 Q25,30 40,50 T70,50 T90,50",
                    "M10,50 Q25,70 40,50 T70,30 T90,50",
                    "M10,50 Q25,30 40,50 T70,50 T90,50"
                  ]
                }}
                transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                fill="none" stroke="#8b5cff" strokeWidth="0.5" className="opacity-60 drop-shadow-[0_0_4px_#8b5cff]"
              />
            </svg>
            <div className="absolute -right-12 font-mono text-[0.45rem] text-[#8b5cff]/60 tracking-[0.3em]">AUDIO_WAVE</div>
          </div>

          {/* 9. TEMPORAL PATH (Blue) */}
          <motion.div 
            animate={{ rotate: -15 }}
            className="absolute w-[90%] h-[90%] border-[1px] border-transparent border-b-[#2f7cff]/30 rounded-full flex items-end justify-center pb-2 translate-y-6"
          >
            {/* Ticks & Scanner */}
            <div className="relative w-1/2 h-4 overflow-hidden flex items-end justify-between px-2">
              {[...Array(12)].map((_, i) => (
                <div key={i} className="w-[1px] h-2 bg-[#2f7cff]/30" />
              ))}
              <motion.div 
                animate={{ x: ['-200%', '400%'] }}
                transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                className="absolute top-0 bottom-0 w-4 bg-gradient-to-r from-transparent via-[#00e5ff]/80 to-transparent blur-[1px]"
              />
            </div>
            <div className="absolute -bottom-6 font-mono text-[0.45rem] text-[#2f7cff]/60 tracking-[0.3em]">TEMPORAL</div>
          </motion.div>

          {/* 10. VISUAL SIGNAL (Cyan Frames) */}
          <div className="absolute top-10 left-10 w-24 h-24">
            <motion.div 
              animate={{ 
                scale: [1, 0.8, 1.1, 1], 
                rotate: [0, 45, 90, 0],
                opacity: [0.2, 0.8, 0, 0.2] 
              }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              className="absolute inset-0 border-[1px] border-[#00e5ff]/40 shadow-[0_0_8px_rgba(0,229,255,0.3)]"
            />
            {/* Corner Brackets */}
            <div className="absolute -top-1 -left-1 w-2 h-2 border-t-[1px] border-l-[1px] border-[#00e5ff]" />
            <div className="absolute -bottom-1 -right-1 w-2 h-2 border-b-[1px] border-r-[1px] border-[#00e5ff]" />
            <div className="absolute -top-6 -left-6 font-mono text-[0.45rem] text-[#00e5ff]/60 tracking-[0.3em]">VISUAL</div>
          </div>

          {/* 11. ORBITAL INTERSECTIONS (Flash Points) */}
          <motion.div 
            animate={{ opacity: [0, 1, 0] }}
            transition={{ duration: 2, repeat: Infinity, delay: 1, times: [0, 0.1, 1] }}
            className="absolute top-12 right-24 w-1.5 h-1.5 bg-[#ff3dce] rounded-full shadow-[0_0_10px_#ff3dce]"
          />
          <motion.div 
            animate={{ opacity: [0, 1, 0] }}
            transition={{ duration: 3, repeat: Infinity, delay: 2, times: [0, 0.1, 1] }}
            className="absolute bottom-20 left-16 w-1.5 h-1.5 bg-[#00e5ff] rounded-full shadow-[0_0_10px_#00e5ff]"
          />
        </motion.div>


        {/* ========================================================= */}
        {/* 2. CORE GLOW + 4. INTERNAL ANIMATION + 3. ENERGY FLOW */}
        {/* ========================================================= */}
        <motion.div 
          style={{ x: coreX, y: coreY }}
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 2, delay: getDelay('core'), type: "spring", stiffness: 40 }}
          className="relative z-30 w-40 h-40 lg:w-56 lg:h-56 flex items-center justify-center group-hover:scale-[1.02] transition-transform duration-500"
        >
          {/* Core Breathing Glow */}
          <motion.div 
            animate={{ 
              boxShadow: [
                '0 0 30px rgba(0,229,255,0.1)', 
                '0 0 50px rgba(0,229,255,0.25)', 
                '0 0 30px rgba(0,229,255,0.1)'
              ] 
            }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            className="absolute inset-0 bg-black/60 backdrop-blur-md"
            style={{ clipPath: 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)' }}
          />

          {/* Hexagon Outlines (Multi-layer Luminous) */}
          <div className="absolute inset-0 border-[1px] border-[#00e5ff]/20 shadow-[inset_0_0_20px_rgba(0,229,255,0.1)]" style={{ clipPath: 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)' }} />
          <div className="absolute inset-1 border-[1px] border-[#2f7cff]/30" style={{ clipPath: 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)' }} />

          {/* 3. CORE ENERGY FLOW (Travelling particle on border) */}
          <svg viewBox="0 0 224 224" className="absolute inset-0 w-full h-full" style={{ filter: 'drop-shadow(0 0 6px #00e5ff)' }}>
            <motion.polygon 
              points="112,0 224,56 224,168 112,224 0,168 0,56"
              fill="none" 
              stroke="#00e5ff" 
              strokeWidth="2"
              strokeDasharray="40 800"
              animate={{ strokeDashoffset: [840, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
              className="w-full h-full"
              vectorEffect="non-scaling-stroke"
            />
          </svg>

          {/* 4. INTERNAL ANIMATION (Micro geometry) */}
          <div className="absolute inset-0 overflow-hidden" style={{ clipPath: 'polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)' }}>
            <motion.div 
              animate={{ rotate: 90 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="absolute inset-0 flex items-center justify-center opacity-30"
            >
              <div className="w-[150%] h-[1px] bg-[#00e5ff]/30" />
              <div className="h-[150%] w-[1px] bg-[#00e5ff]/30" />
            </motion.div>
            {/* Tiny Nodes */}
            <motion.div animate={{ opacity: [0.2, 0.8, 0.2] }} transition={{ duration: 2, repeat: Infinity }} className="absolute top-1/4 left-1/4 w-1 h-1 bg-[#2f7cff] rounded-full" />
            <motion.div animate={{ opacity: [0.2, 0.8, 0.2] }} transition={{ duration: 3, repeat: Infinity }} className="absolute bottom-1/4 right-1/4 w-1 h-1 bg-[#8b5cff] rounded-full" />
          </div>

          {/* Stable Text */}
          <div className="absolute text-center flex flex-col items-center z-40 pointer-events-none drop-shadow-[0_0_8px_rgba(0,0,0,1)]">
            <div className="font-mono text-[0.65rem] lg:text-[0.75rem] font-bold text-white tracking-[0.3em]">MEDIADNA</div>
            <div className="font-mono text-[0.45rem] lg:text-[0.5rem] text-[#00e5ff] mt-1 tracking-[0.5em]">CORE</div>
          </div>

        </motion.div>

        {/* ========================================================= */}
        {/* 12. MAJOR SYNCHRONIZATION EVENT & 13. NEON SHOCKWAVE */}
        {/* ========================================================= */}
        <AnimatePresence>
          {isSyncing && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex items-center justify-center pointer-events-none z-50"
            >
              {/* Cyan Shockwave */}
              <motion.div 
                initial={{ scale: 0.5, opacity: 0.8 }}
                animate={{ scale: 2.5, opacity: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="absolute w-40 h-40 rounded-full border-[2px] border-[#00e5ff] shadow-[0_0_20px_#00e5ff]"
              />
              {/* Blue Shockwave */}
              <motion.div 
                initial={{ scale: 0.5, opacity: 0.5 }}
                animate={{ scale: 2.8, opacity: 0 }}
                transition={{ duration: 1.0, ease: "easeOut", delay: 0.1 }}
                className="absolute w-40 h-40 rounded-full border-[1px] border-[#2f7cff] shadow-[0_0_10px_#2f7cff]"
              />
              
              {/* Sync Label */}
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.2 }}
                className="absolute -top-16 bg-[#00e5ff]/10 border border-[#00e5ff]/50 px-3 py-1 font-mono text-[0.5rem] font-bold text-[#00e5ff] tracking-[0.4em] shadow-[0_0_15px_rgba(0,229,255,0.4)] backdrop-blur-md"
              >
                MULTIMODAL SYNC
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

      </motion.div>
    </div>
  );
}
