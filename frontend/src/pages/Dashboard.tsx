import { useNavigate } from 'react-router-dom';
import { Play, Fingerprint, ScanEye } from 'lucide-react';
import MediaDNAEngine from '../components/MediaDNAEngine';
import { motion } from 'framer-motion';
import { useState } from 'react';
import SignalCards from '../components/SignalCards';

export default function Dashboard() {
  const navigate = useNavigate();
  const [activePreview, setActivePreview] = useState<string | null>(null);

  const demos = [
    {
      id: 'real_real',
      type: 'real_real',
      title: 'REAL / REAL',
      audio: 'AUTHENTIC',
      video: 'AUTHENTIC',
      desc: 'Pristine, untampered media. Both visual and acoustic modalities are entirely genuine and naturally synchronized.',
      videoUrl: '/test_real.mp4',
      color: '#10b981'
    },
    {
      id: 'real_fake',
      type: 'real_fake',
      title: 'REAL / FAKE',
      audio: 'AUTHENTIC',
      video: 'SYNTHETIC',
      desc: 'Authentic audio track paired with a synthesized visual track (e.g., Deepfake face-swap or Wav2Lip).',
      videoUrl: '/test_fake.mp4',
      color: '#ef4444'
    },
    {
      id: 'fake_real',
      type: 'fake_real',
      title: 'FAKE / REAL',
      audio: 'SYNTHETIC',
      video: 'AUTHENTIC',
      desc: 'Synthetically generated or cloned audio track dubbed over pristine, unedited video footage.',
      videoUrl: '/test_fake.mp4',
      color: '#ef4444'
    },
    {
      id: 'fake_fake',
      type: 'fake_fake',
      title: 'FAKE / FAKE',
      audio: 'SYNTHETIC',
      video: 'SYNTHETIC',
      desc: 'Fully synthetic generation where both visual geometry and acoustic characteristics are artificially constructed.',
      videoUrl: '/test_fake.mp4',
      color: '#ef4444'
    }
  ];

  const handleDemoClick = (type: string) => {
    navigate('/analyze', { state: { demoType: type } });
  };

  return (
    <div className="min-h-screen pt-24 pb-12 px-6 lg:px-12 relative overflow-hidden bg-[#030508]">
      
      {/* ─── Atmospheric Background ─── */}
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_top_right,rgba(0,229,255,0.03)_0%,transparent_50%)]" />
      <div className="absolute inset-0 pointer-events-none bg-[linear-gradient(rgba(255,255,255,0.015)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.015)_1px,transparent_1px)] bg-[size:48px_48px] opacity-30" />
      <div className="absolute inset-0 pointer-events-none mix-blend-overlay opacity-10 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0IiBoZWlnaHQ9IjQiPgo8cmVjdCB3aWR0aD0iNCIgaGVpZ2h0PSI0IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMDUiLz4KPC9zdmc+')] " />
      
      {/* ─── Hero Section ─── */}
      <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-12 lg:gap-24 items-center mb-32">
        <motion.div 
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="space-y-8 z-10 relative"
        >
          <div>
            <div className="font-mono text-[0.65rem] font-bold tracking-[0.4em] text-[#00e5ff] mb-6 flex items-center gap-2">
              <Fingerprint className="w-4 h-4" />
              DIGITAL MEDIA FORENSICS
            </div>
            <h1 className="editorial-headline text-white tracking-tighter" style={{ fontSize: 'clamp(4rem, 8vw, 6.5rem)', lineHeight: 0.85 }}>
              MEDIA<br />
              <span className="text-[#00e5ff] border-b-[6px] border-[#00e5ff]/80 inline-block pb-2 pr-4 shadow-[0_4px_20px_rgba(0,229,255,0.15)]">
                DNA
              </span>
            </h1>
          </div>
          <p className="text-sm lg:text-[0.95rem] text-gray-400 font-mono max-w-lg leading-[1.8] tracking-wide">
            Multimodal media authenticity analysis. Interrogate visual, audio, and temporal signals. Build evidence. Preserve provenance.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-5 pt-6">
            <button 
              onClick={() => navigate('/analyze')}
              data-hover="node"
              className="btn-primary py-4 px-8 text-[0.7rem] w-full sm:w-auto flex justify-center items-center gap-3 font-bold tracking-[0.2em] relative overflow-hidden group border border-[#00e5ff]/40 hover:border-[#00e5ff] transition-all bg-[#00e5ff]/5 hover:bg-[#00e5ff]/10"
            >
              <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-[#00e5ff]/30 to-transparent group-hover:animate-[scanline_1.5s_ease-in-out_infinite]" />
              <ScanEye className="w-4 h-4 z-10" /> 
              <span className="z-10">ANALYZE MEDIA</span>
            </button>
            <button 
              onClick={() => {
                const el = document.getElementById('demo-lab');
                el?.scrollIntoView({ behavior: 'smooth' });
              }}
              data-hover="node"
              className="btn-secondary py-4 px-8 text-[0.7rem] w-full sm:w-auto flex justify-center items-center gap-3 font-bold tracking-[0.2em] group hover:text-white transition-colors"
            >
              <span className="group-hover:translate-x-1 transition-transform flex items-center gap-3">
                <Play className="w-4 h-4" /> EXPLORE SYSTEM
              </span>
            </button>
          </div>
        </motion.div>

        {/* Abstract Core Visualization */}
        <MediaDNAEngine />
      </div>
      
      <SignalCards />

      {/* ─── Forensic Demo Lab (2x2 Matrix) ─── */}
      <div id="demo-lab" className="max-w-6xl mx-auto pt-12">
        <div className="flex items-center gap-4 mb-12">
          <div className="w-8 h-[1px] bg-[#00e5ff]"></div>
          <h2 className="font-mono text-sm font-bold text-[#00e5ff] tracking-[0.3em] uppercase">FORENSIC DEMO LAB</h2>
          <div className="flex-1 h-[1px] bg-white/10"></div>
        </div>

        <div className="grid md:grid-cols-2 gap-6 relative">
          
          {/* Axis Labels - Desktop Only */}
          <div className="hidden md:flex absolute -left-12 top-0 bottom-0 flex-col justify-around text-[#00e5ff] font-mono text-[0.6rem] font-bold tracking-[0.3em] uppercase" style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)' }}>
            <span>VIDEO SYNTHETIC</span>
            <span>VIDEO AUTHENTIC</span>
          </div>
          <div className="hidden md:flex absolute -top-8 left-0 right-0 justify-around text-[#00e5ff] font-mono text-[0.6rem] font-bold tracking-[0.3em] uppercase">
            <span>AUDIO AUTHENTIC</span>
            <span>AUDIO SYNTHETIC</span>
          </div>

          {demos.map((demo, idx) => (
            <motion.div 
              key={demo.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: idx * 0.1, duration: 0.5 }}
              className="glass-card flex flex-col rounded-xl overflow-hidden group border border-white/10 hover:border-[#00e5ff]/50 transition-all duration-500 relative"
              onMouseEnter={() => setActivePreview(demo.id)}
              onMouseLeave={() => setActivePreview(null)}
            >
              {/* Preview Window */}
              <div className="relative h-48 bg-[#06080d] border-b border-white/10 overflow-hidden flex items-center justify-center">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.05)_0%,transparent_100%)]"></div>
                
                {/* Simulated playback */}
                <video 
                  src={demo.videoUrl} 
                  autoPlay 
                  loop 
                  muted 
                  playsInline
                  className={`absolute w-full h-full object-cover transition-opacity duration-700 ${activePreview === demo.id ? 'opacity-50 grayscale-0' : 'opacity-20 grayscale'}`}
                />
                
                <div className="absolute inset-0 bg-black/40"></div>

                <div className="relative z-10 flex flex-col items-center gap-2">
                  <div className="font-mono text-xl font-bold tracking-widest text-white">{demo.title}</div>
                  <div className="flex gap-4 font-mono text-[0.6rem] tracking-[0.2em] font-bold">
                    <span style={{ color: demo.audio === 'AUTHENTIC' ? '#10b981' : '#ef4444' }}>A: {demo.audio}</span>
                    <span style={{ color: demo.video === 'AUTHENTIC' ? '#10b981' : '#ef4444' }}>V: {demo.video}</span>
                  </div>
                </div>

                {/* Reticles */}
                <div className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-white/20 group-hover:border-[#00e5ff] transition-colors"></div>
                <div className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-white/20 group-hover:border-[#00e5ff] transition-colors"></div>
                <div className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-white/20 group-hover:border-[#00e5ff] transition-colors"></div>
                <div className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-white/20 group-hover:border-[#00e5ff] transition-colors"></div>
              </div>

              {/* Explanation & Action */}
              <div className="p-6 flex flex-col flex-1 bg-black/40 backdrop-blur-xl">
                <p className="text-sm text-gray-400 font-mono leading-relaxed mb-6 flex-1">
                  {demo.desc}
                </p>
                <button 
                  onClick={() => handleDemoClick(demo.type)}
                  data-hover="node"
                  className="w-full py-3 px-4 border border-white/10 hover:border-[#00e5ff]/50 bg-white/5 hover:bg-[#00e5ff]/10 text-white font-mono text-[0.65rem] font-bold tracking-[0.2em] transition-all flex items-center justify-center gap-2 uppercase"
                >
                  <ScanEye className="w-3.5 h-3.5 text-[#00e5ff]" />
                  BEGIN FORENSIC ANALYSIS
                </button>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

    </div>
  );
}
