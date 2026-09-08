import { useState, useEffect, useRef, useCallback, useLayoutEffect } from 'react';
import { Layers, Hexagon, Database, Activity, Play, Info, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

type Mode = 'SYSTEM' | 'DATA_FLOW' | 'ANATOMY' | 'X_RAY';

type NodeID = 
  | 'MEDIA' | 'VIDEO_STREAM' | 'AUDIO_STREAM' | 'FRAME_SAMPLING' | 'KALDI_FBANK'
  | 'OPENAVFF' | 'VISUAL_ENCODER' | 'AUDIO_ENCODER' | 'A2V' | 'V2A' 
  | 'FEATURE_CONCAT' | 'MEAN_POOLING' | 'MLP_VISION' | 'MLP_AUDIO' | 'MLP_HEAD'
  | 'VISUAL_ANOMALY' | 'METADATA' | 'FUSION' | 'RESULT';

interface InspectorData {
  title: string;
  what: string;
  input: string;
  process: string;
  output: string;
  why: string;
  color: string;
}

const INSPECTOR_DB: Record<string, InspectorData> = {
  MEDIA: {
    title: "MEDIA INGESTION",
    what: "The pipeline entry point where raw video files are received and validated.",
    input: "Video Container (.mp4, .mkv, etc.)",
    process: "Validates container format, metadata integrity, and ensures stream multiplexing is intact.",
    output: "Raw Byte Streams",
    why: "Ensures the engine operates only on structurally sound data, preventing decoding crashes downstream.",
    color: "#ffffff"
  },
  VIDEO_STREAM: {
    title: "VIDEO PATHWAY",
    what: "Extracts the raw visual bitstream.",
    input: "Media Byte Stream",
    process: "Demuxes the video track utilizing decord to parse H.264/H.265 frames.",
    output: "Raw Frames",
    why: "Provides the visual basis for both semantic (OpenAVFF) and deterministic (Anomaly) analysis.",
    color: "#00e5ff"
  },
  AUDIO_STREAM: {
    title: "AUDIO PATHWAY",
    what: "Extracts the raw acoustic track.",
    input: "Media Byte Stream",
    process: "Utilizes FFmpeg subprocesses to isolate and standardize the audio track.",
    output: "16kHz Mono WAV",
    why: "Isolates the acoustic signal necessary for voice cloning and synchronization analysis.",
    color: "#b388ff"
  },
  FRAME_SAMPLING: {
    title: "FRAME SAMPLING",
    what: "Uniformly samples frames across time.",
    input: "Raw Frames",
    process: "Selects 16 equidistant frames across the video duration to form a temporal representation.",
    output: "Tensor [3, 16, 224, 224]",
    why: "Balances temporal context with computational efficiency for the Vision Transformer.",
    color: "#00e5ff"
  },
  KALDI_FBANK: {
    title: "KALDI FBANK",
    what: "Computes acoustic spectral features.",
    input: "16kHz Mono WAV",
    process: "Applies Mel-filterbanks over short-time Fourier transforms using Torchaudio.",
    output: "FBank Tensor [1024, 128]",
    why: "Provides a structured frequency-domain representation optimal for audio neural networks.",
    color: "#b388ff"
  },
  OPENAVFF: {
    title: "OPENAVFF MULTIMODAL CORE",
    what: "The primary AI inference engine (VideoCAVMAEFT).",
    input: "Visual Tensors & Audio FBanks",
    process: "Combines learned visual and acoustic representations through cross-modal attention.",
    output: "Deep Multimodal Features",
    why: "Detects subtle de-synchronizations and deepfake artifacts imperceptible to single-modality models.",
    color: "#e0f7fa"
  },
  VISUAL_ENCODER: {
    title: "VISUAL ENCODER",
    what: "Extracts spatial and temporal features.",
    input: "Frames Tensor",
    process: "Processes 16x16 patches through Vision Transformer (ViT) self-attention blocks.",
    output: "Visual Embeddings",
    why: "Identifies facial warping, unnatural blending, and spatial anomalies.",
    color: "#00e5ff"
  },
  AUDIO_ENCODER: {
    title: "AUDIO ENCODER",
    what: "Extracts spectral frequency structures.",
    input: "FBank Tensor",
    process: "Applies Audio Spectrogram Transformer (AST) self-attention.",
    output: "Audio Embeddings",
    why: "Identifies vocoder artifacts, frequency banding, and synthetic generation traits.",
    color: "#b388ff"
  },
  A2V: {
    title: "AUDIO-TO-VISUAL ATTENTION",
    what: "Audio-to-visual cross-modal interaction.",
    input: "Audio Embeddings, Visual Embeddings",
    process: "Audio signals query visual features to find matching lip/facial movements.",
    output: "Audio-Attended Visual Features",
    why: "Crucial for detecting lip-sync failures in dubbed or generated videos.",
    color: "#e0f7fa"
  },
  V2A: {
    title: "VISUAL-TO-AUDIO ATTENTION",
    what: "Visual-to-audio cross-modal interaction.",
    input: "Visual Embeddings, Audio Embeddings",
    process: "Visual signals query audio features to correlate scene events with sound.",
    output: "Visual-Attended Audio Features",
    why: "Ensures acoustic characteristics match the generated visual environment.",
    color: "#e0f7fa"
  },
  FEATURE_CONCAT: {
    title: "FEATURE CONCATENATION",
    what: "Merges the attended modalities.",
    input: "Attended Audio & Visual Tensors",
    process: "Concatenates the representations along the feature dimension.",
    output: "Joint Multimodal Tensor",
    why: "Prepares a unified representation for final classification.",
    color: "#e0f7fa"
  },
  MEAN_POOLING: {
    title: "MEAN POOLING",
    what: "Reduces dimensionality.",
    input: "Joint Multimodal Tensor",
    process: "Averages features across the temporal/sequence dimension.",
    output: "Pooled Vector",
    why: "Summarizes the sequence into a compact vector representing the entire media clip.",
    color: "#e0f7fa"
  },
  MLP_VISION: {
    title: "MLP VISION PROJECTION",
    what: "Projects visual dimension.",
    input: "Pooled Visual Component",
    process: "Linear projection into unified classification space.",
    output: "1024-d Vector",
    why: "Aligns representation scales.",
    color: "#00e5ff"
  },
  MLP_AUDIO: {
    title: "MLP AUDIO PROJECTION",
    what: "Projects audio dimension.",
    input: "Pooled Audio Component",
    process: "Linear projection into unified classification space.",
    output: "1024-d Vector",
    why: "Aligns representation scales.",
    color: "#b388ff"
  },
  MLP_HEAD: {
    title: "CLASSIFICATION HEAD",
    what: "The final PyTorch classification layer.",
    input: "Projected Vectors",
    process: "Passes vectors through a Multilayer Perceptron to compute log-probabilities.",
    output: "Logits [Fake, Real]",
    why: "Yields the definitive OpenAVFF model signal.",
    color: "#e0f7fa"
  },
  VISUAL_ANOMALY: {
    title: "VISUAL / TEMPORAL ANALYSIS",
    what: "Extracts traditional forensic signals.",
    input: "Raw Video Frames",
    process: "Computes Laplacian variance (blur), structural similarity, and noise gradients.",
    output: "Visual Anomaly Score",
    why: "Catches basic tampering that deep models might over-generalize.",
    color: "#ff4081" // magenta
  },
  METADATA: {
    title: "METADATA ANALYSIS",
    what: "Extracts technical characteristics.",
    input: "Video Container",
    process: "Parses atoms/moov data for unexpected codec strings or creation dates.",
    output: "Metadata Flags",
    why: "Reveals software signatures left by tampering tools.",
    color: "#ffb300" // amber
  },
  FUSION: {
    title: "MEDIADNA FUSION",
    what: "The deterministic fusion layer.",
    input: "Model Logits, Visual Score, Metadata",
    process: "Applies a weighted ensemble logic to combine deep and traditional signals.",
    output: "MediaDNA Score [0.0 - 1.0]",
    why: "Provides a robust, multi-faceted defense against varied attack vectors.",
    color: "#e0f7fa"
  },
  RESULT: {
    title: "FORENSIC ASSESSMENT",
    what: "The final verdict.",
    input: "MediaDNA Score",
    process: "Thresholds the score to classify the media.",
    output: "LIKELY FAKE / LIKELY REAL",
    why: "Translates complex forensic data into an actionable decision.",
    color: "#ef5350"
  }
};

const TRACE_SEQUENCE = [
  { step: 'MEDIA DETECTED', nodes: ['MEDIA'] },
  { step: 'VIDEO / AUDIO SPLIT', nodes: ['VIDEO_STREAM', 'AUDIO_STREAM', 'FRAME_SAMPLING', 'KALDI_FBANK'] },
  { step: 'REPRESENTATION', nodes: ['OPENAVFF', 'VISUAL_ENCODER', 'AUDIO_ENCODER'] },
  { step: 'CROSS-MODAL ANALYSIS', nodes: ['OPENAVFF', 'A2V', 'V2A'] },
  { step: 'FEATURE FUSION', nodes: ['OPENAVFF', 'FEATURE_CONCAT', 'MEAN_POOLING'] },
  { step: 'MODEL SIGNAL', nodes: ['OPENAVFF', 'MLP_VISION', 'MLP_AUDIO', 'MLP_HEAD'] },
  { step: 'MEDIADNA', nodes: ['VISUAL_ANOMALY', 'METADATA', 'FUSION'] },
  { step: 'FORENSIC ASSESSMENT', nodes: ['RESULT'] }
];

interface Connection {
  from: string;
  to: string;
  color?: string;
}

const CONNECTIONS: Connection[] = [
  { from: 'MEDIA', to: 'VIDEO_STREAM', color: '#00e5ff' },
  { from: 'MEDIA', to: 'AUDIO_STREAM', color: '#b388ff' },
  { from: 'VIDEO_STREAM', to: 'FRAME_SAMPLING', color: '#00e5ff' },
  { from: 'AUDIO_STREAM', to: 'KALDI_FBANK', color: '#b388ff' },
  { from: 'FRAME_SAMPLING', to: 'VISUAL_ENCODER', color: '#00e5ff' },
  { from: 'KALDI_FBANK', to: 'AUDIO_ENCODER', color: '#b388ff' },
  
  { from: 'VISUAL_ENCODER', to: 'A2V', color: '#00e5ff' },
  { from: 'VISUAL_ENCODER', to: 'V2A', color: '#00e5ff' },
  { from: 'AUDIO_ENCODER', to: 'A2V', color: '#b388ff' },
  { from: 'AUDIO_ENCODER', to: 'V2A', color: '#b388ff' },
  
  { from: 'A2V', to: 'FEATURE_CONCAT', color: '#e0f7fa' },
  { from: 'V2A', to: 'FEATURE_CONCAT', color: '#e0f7fa' },
  { from: 'FEATURE_CONCAT', to: 'MEAN_POOLING', color: '#e0f7fa' },
  
  { from: 'MEAN_POOLING', to: 'MLP_VISION', color: '#00e5ff' },
  { from: 'MEAN_POOLING', to: 'MLP_AUDIO', color: '#b388ff' },
  
  { from: 'MLP_VISION', to: 'MLP_HEAD', color: '#00e5ff' },
  { from: 'MLP_AUDIO', to: 'MLP_HEAD', color: '#b388ff' },
  
  // High level connections if OPENAVFF is collapsed
  { from: 'FRAME_SAMPLING', to: 'OPENAVFF', color: '#00e5ff' },
  { from: 'KALDI_FBANK', to: 'OPENAVFF', color: '#b388ff' },
  { from: 'OPENAVFF', to: 'FUSION', color: '#e0f7fa' },
  
  // If expanded, MLP_HEAD connects to fusion
  { from: 'MLP_HEAD', to: 'FUSION', color: '#e0f7fa' },
  
  // Branches
  { from: 'MEDIA', to: 'METADATA', color: '#ffb300' },
  { from: 'FRAME_SAMPLING', to: 'VISUAL_ANOMALY', color: '#ff4081' },
  
  { from: 'VISUAL_ANOMALY', to: 'FUSION', color: '#ff4081' },
  { from: 'METADATA', to: 'FUSION', color: '#ffb300' },
  { from: 'FUSION', to: 'RESULT', color: '#e0f7fa' }
];

export default function Architecture() {
  const [mode, setMode] = useState<Mode>('SYSTEM');
  const [activeNode, setActiveNode] = useState<NodeID | null>(null);
  const [inspectedNode, setInspectedNode] = useState<NodeID | null>(null);
  const [traceStep, setTraceStep] = useState(-1);
  const [anatomyExpanded, setAnatomyExpanded] = useState(false);
  
  const containerRef = useRef<HTMLDivElement>(null);
  const [nodeRects, setNodeRects] = useState<Record<string, DOMRect>>({});
  
  const isXRay = mode === 'X_RAY';
  const isDataFlow = mode === 'DATA_FLOW';

  useEffect(() => {
    if (mode === 'ANATOMY') {
      setAnatomyExpanded(true);
    } else {
      setAnatomyExpanded(false);
    }
  }, [mode]);

  useEffect(() => {
    let t: ReturnType<typeof setTimeout>;
    if (traceStep >= 0 && traceStep < TRACE_SEQUENCE.length) {
      t = setTimeout(() => setTraceStep(prev => prev + 1), 1500); // Slower, more cinematic timing
    } else if (traceStep >= TRACE_SEQUENCE.length) {
      t = setTimeout(() => setTraceStep(-1), 3000);
    }
    return () => clearTimeout(t);
  }, [traceStep]);

  // Dynamic layout measurement
  const updateRects = useCallback(() => {
    if (!containerRef.current) return;
    const containerRect = containerRef.current.getBoundingClientRect();
    const nodes = document.querySelectorAll('[data-arch-node]');
    
    const newRects: Record<string, DOMRect> = {};
    nodes.forEach(node => {
      const id = node.getAttribute('data-arch-node') as string;
      const rect = node.getBoundingClientRect();
      newRects[id] = new DOMRect(
        rect.left - containerRect.left,
        rect.top - containerRect.top,
        rect.width,
        rect.height
      );
    });
    setNodeRects(newRects);
  }, []);

  useLayoutEffect(() => {
    updateRects();
    window.addEventListener('resize', updateRects);
    
    const resizeObserver = new ResizeObserver(() => updateRects());
    if (containerRef.current) resizeObserver.observe(containerRef.current);
    
    return () => {
      window.removeEventListener('resize', updateRects);
      resizeObserver.disconnect();
    };
  }, [updateRects]);
  
  // Re-measure after transitions
  useEffect(() => {
    const timer = setTimeout(updateRects, 600);
    return () => clearTimeout(timer);
  }, [anatomyExpanded, updateRects]);

  const handleNodeClick = (id: NodeID) => {
    if (id === 'OPENAVFF') {
      setAnatomyExpanded(prev => !prev);
    }
    setInspectedNode(id);
  };

  // Hover isolation logic
  const isPathRelated = (nodeId: string) => {
    if (!activeNode) return true; // If nothing active, everything is related
    if (activeNode === nodeId) return true;
    
    // Check direct connections
    const isDirectlyConnected = CONNECTIONS.some(c => 
      (c.from === activeNode && c.to === nodeId) || 
      (c.to === activeNode && c.from === nodeId)
    );
    return isDirectlyConnected;
  };

  const isTraced = (nodeId: string) => {
    if (traceStep < 0 || traceStep >= TRACE_SEQUENCE.length) return false;
    return TRACE_SEQUENCE[traceStep].nodes.includes(nodeId);
  };

  const renderConnections = () => {
    return (
      <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
        <defs>
          {['cyan', 'violet', 'white', 'magenta', 'amber'].map(colorName => {
            const hex = colorName === 'cyan' ? '#00e5ff' : colorName === 'violet' ? '#b388ff' : colorName === 'magenta' ? '#ff4081' : colorName === 'amber' ? '#ffb300' : '#e0f7fa';
            return (
              <marker key={colorName} id={`arrow-${colorName}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
                <path d="M 0 0 L 10 5 L 0 10 z" fill={isXRay ? '#666' : hex} />
              </marker>
            );
          })}
        </defs>
        
        {CONNECTIONS.map((conn, idx) => {
          if (!anatomyExpanded) {
            const internalNodes = ['VISUAL_ENCODER', 'AUDIO_ENCODER', 'A2V', 'V2A', 'FEATURE_CONCAT', 'MEAN_POOLING', 'MLP_VISION', 'MLP_AUDIO', 'MLP_HEAD'];
            if (internalNodes.includes(conn.from) || internalNodes.includes(conn.to)) {
              if (conn.from === 'MLP_HEAD' && conn.to === 'FUSION') return null; 
              return null; 
            }
          } else {
            if ((conn.from === 'FRAME_SAMPLING' || conn.from === 'KALDI_FBANK') && conn.to === 'OPENAVFF') return null;
            if (conn.from === 'OPENAVFF' && conn.to === 'FUSION') return null;
          }

          const fromRect = nodeRects[conn.from];
          const toRect = nodeRects[conn.to];
          if (!fromRect || !toRect) return null;

          const startX = fromRect.x + fromRect.width / 2;
          const startY = fromRect.y + fromRect.height;
          let endX = toRect.x + toRect.width / 2;
          let endY = toRect.y;

          if (conn.to === 'VISUAL_ANOMALY' || conn.to === 'METADATA') {
             startX === endX ? endX = toRect.x + toRect.width / 2 : endX = startX;
          }

          const pathD = `M ${startX} ${startY} C ${startX} ${startY + (endY - startY)/2}, ${endX} ${startY + (endY - startY)/2}, ${endX} ${endY}`;
          
          let markerId = 'arrow-white';
          if (conn.color === '#00e5ff') markerId = 'arrow-cyan';
          if (conn.color === '#b388ff') markerId = 'arrow-violet';
          if (conn.color === '#ff4081') markerId = 'arrow-magenta';
          if (conn.color === '#ffb300') markerId = 'arrow-amber';

          // Hover Isolation & Trace highlights
          const isRelatedPath = !activeNode || activeNode === conn.from || activeNode === conn.to;
          const isTracePath = traceStep >= 0 && (isTraced(conn.from) || isTraced(conn.to));
          
          let strokeOpacity = 0.3;
          if (isXRay) strokeOpacity = 0.15;
          if (activeNode && isRelatedPath) strokeOpacity = 1;
          if (isTracePath) strokeOpacity = 1;
          if (activeNode && !isRelatedPath) strokeOpacity = 0.05;

          const strokeWidth = (activeNode && isRelatedPath) || isTracePath ? 3 : 2;

          return (
            <g key={idx}>
              <path 
                d={pathD} 
                fill="none" 
                stroke={isXRay ? '#666' : (conn.color || '#fff')} 
                strokeWidth={strokeWidth} 
                strokeOpacity={strokeOpacity}
                markerEnd={`url(#${markerId})`}
                className="transition-all duration-500 ease-out"
              />
              
              {/* Particle Animation in Data Flow mode */}
              {isDataFlow && !isXRay && (
                <path 
                  d={pathD} 
                  fill="none" 
                  stroke={conn.color || '#fff'} 
                  strokeWidth="4" 
                  strokeDasharray="2 150"
                  strokeLinecap="round"
                  className="animate-particles opacity-70"
                  style={{ animationDuration: '2.5s', animationTimingFunction: 'linear', animationIterationCount: 'infinite' }}
                />
              )}

              {/* Trace Path Blast */}
              {isTracePath && (
                <path 
                  d={pathD} 
                  fill="none" 
                  stroke={conn.color || '#fff'} 
                  strokeWidth="6" 
                  strokeDasharray="1000"
                  strokeDashoffset="1000"
                  className="animate-trace opacity-90"
                  style={{ animationDuration: '1s', animationTimingFunction: 'ease-out', animationFillMode: 'forwards' }}
                />
              )}
            </g>
          );
        })}
      </svg>
    );
  };

  const ArchNode = ({ id, label, hero = false }: { id: NodeID, label: string, hero?: boolean }) => {
    const isHover = activeNode === id;
    const isSelected = inspectedNode === id;
    const data = INSPECTOR_DB[id] || { color: '#ffffff' };
    const color = data.color;
    const activeTrace = isTraced(id);
    const related = isPathRelated(id);
    
    let bgStyle = isXRay ? '#050505' : (hero ? 'rgba(17, 24, 39, 0.8)' : 'rgba(31, 41, 55, 0.6)');
    if (activeTrace) bgStyle = color;
    
    let borderStyle = isHover || isSelected || activeTrace 
      ? `1px solid ${color}` 
      : `1px solid ${isXRay ? '#333' : 'rgba(255, 255, 255, 0.1)'}`;
      
    if (activeTrace) borderStyle = `2px solid ${color}`;

    const shadowStyle = (isHover || activeTrace) && !isXRay ? `0 0 30px ${color}60` : 'none';
    const textCol = isXRay ? '#ccc' : (activeTrace ? '#000' : color);
    
    const opacity = activeNode && !related ? 0.2 : 1;

    const sizeClasses = hero 
      ? 'w-full max-w-lg py-8 text-xl font-bold tracking-[0.2em] backdrop-blur-md' 
      : 'w-40 md:w-56 py-4 text-xs font-semibold tracking-wider backdrop-blur-sm';

    return (
      <div 
        data-arch-node={id}
        onClick={(e) => { e.stopPropagation(); handleNodeClick(id); }}
        onMouseEnter={() => setActiveNode(id)}
        onMouseLeave={() => setActiveNode(null)}
        data-hover="node"
        data-color={color}
        className={`relative z-10 flex flex-col items-center justify-center cursor-pointer transition-all duration-500 rounded-lg shadow-2xl ${sizeClasses}`}
        style={{
          background: bgStyle,
          border: borderStyle,
          boxShadow: shadowStyle,
          opacity,
          transform: activeTrace || isHover ? 'scale(1.05)' : 'scale(1)',
        }}
      >
        <span className="font-mono text-center px-4 leading-tight transition-colors duration-300" style={{ color: textCol }}>
          {label}
        </span>
        {hero && !anatomyExpanded && (
           <span className="font-mono text-[0.6rem] opacity-50 mt-2 transition-colors duration-300" style={{ color: textCol }}>[ CLICK TO EXPAND ANATOMY ]</span>
        )}
        
        {isXRay && id === 'OPENAVFF' && (
          <div className="absolute -top-4 -right-6 bg-red-950 border border-red-500 text-[0.6rem] font-mono text-white px-2 py-1 rotate-3 shadow-lg z-20 whitespace-nowrap opacity-90">
            KNOWN MODEL BLINDSPOT<br/>FV-RA: 27.20% ACC
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={`min-h-screen flex flex-col pt-16 transition-all duration-700 ${isXRay ? 'grayscale contrast-125 bg-black text-gray-300' : 'bg-[#06080d] text-gray-100'}`}>
      
      {/* ─── Header ─── */}
      <div className="px-6 lg:px-12 py-8 flex flex-col lg:flex-row justify-between items-start lg:items-end gap-6 border-b border-white/5 bg-[#0b0e14] z-20 shrink-0">
        <div>
          <h1 className="font-mono text-[0.65rem] font-bold tracking-[0.3em] mb-3 text-gray-500">
            MEDIA DNA — V6 CINEMATIC
          </h1>
          <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight" style={{ fontFamily: 'Inter, sans-serif' }}>
            <span className={isXRay ? 'text-gray-100' : 'text-[#00e5ff]'}>FORENSIC</span> ARCHITECTURE
          </h2>
          <p className="mt-4 max-w-2xl text-sm leading-relaxed text-gray-400">
            Explore the neural inference engine.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {[
            { id: 'SYSTEM', label: 'SYSTEM', icon: Database },
            { id: 'DATA_FLOW', label: 'DATA FLOW', icon: Activity },
            { id: 'ANATOMY', label: 'ANATOMY', icon: Layers },
            { id: 'X_RAY', label: 'X-RAY', icon: Hexagon },
          ].map(m => (
            <button 
              key={m.id}
              onClick={() => { setMode(m.id as Mode); setTraceStep(-1); if (m.id === 'ANATOMY') setAnatomyExpanded(true); }}
              data-hover="node"
              className={`flex items-center gap-2 text-[0.65rem] font-bold px-4 py-2 border rounded transition-all duration-300 font-mono tracking-wider
                ${mode === m.id ? 'border-[#00e5ff] text-[#00e5ff] bg-[#00e5ff]/10 shadow-[0_0_15px_rgba(0,229,255,0.15)]' : 'border-white/10 hover:border-white/30 text-gray-400 hover:text-white'}`}
              style={isXRay && mode === m.id ? { borderColor: 'white', color: 'white', background: 'rgba(255,255,255,0.1)' } : {}}
            >
              <m.icon className="w-3 h-3" /> {m.label}
            </button>
          ))}
          <div className="w-px h-6 mx-2 hidden sm:block bg-white/10"></div>
          <button 
            onClick={() => { setMode('SYSTEM'); setTraceStep(0); }}
            data-hover="node"
            className={`flex items-center gap-2 text-[0.65rem] font-bold px-6 py-2 rounded bg-[#00e5ff] text-black hover:bg-white hover:shadow-[0_0_20px_rgba(0,229,255,0.4)] transition-all duration-300 font-mono tracking-wider
              ${traceStep >= 0 ? 'animate-pulse bg-white' : ''}`}
          >
            <Play className="w-3 h-3" /> TRACE MEDIA
          </button>
        </div>
      </div>

      {/* ─── Trace Cinematic Overlay ─── */}
      <AnimatePresence>
        {traceStep >= 0 && traceStep < TRACE_SEQUENCE.length && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="fixed top-32 left-1/2 -translate-x-1/2 z-[100] pointer-events-none text-center"
          >
            <div className="font-mono text-[0.65rem] text-[#00e5ff] font-bold tracking-[0.3em] mb-1">VISUAL TRACE</div>
            <div className="bg-black/80 backdrop-blur-md border border-[#00e5ff]/50 px-8 py-3 rounded-full text-white font-mono text-lg shadow-[0_0_30px_rgba(0,229,255,0.3)] tracking-widest uppercase">
              {TRACE_SEQUENCE[traceStep].step}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ─── Main Viewport ─── */}
      <div className="flex-1 flex flex-col lg:flex-row relative overflow-hidden h-full">
        
        {/* Architecture Canvas */}
        <div 
          ref={containerRef}
          className="flex-1 relative w-full overflow-y-auto overflow-x-hidden p-8 lg:p-16 custom-scrollbar"
        >
          {renderConnections()}
          
          <div className="max-w-6xl mx-auto flex flex-col items-center gap-16 relative z-10 pb-32">
            
            <ArchNode id="MEDIA" label="MEDIA" />

            <div className="w-full flex justify-center gap-8 md:gap-32">
              <div className="flex flex-col items-center gap-16">
                <ArchNode id="VIDEO_STREAM" label="VIDEO STREAM" />
                <ArchNode id="FRAME_SAMPLING" label="FRAME SAMPLING" />
              </div>
              <div className="flex flex-col items-center gap-16">
                <ArchNode id="AUDIO_STREAM" label="AUDIO STREAM" />
                <ArchNode id="KALDI_FBANK" label="FFMPEG / KALDI FBANK" />
              </div>
            </div>

            {/* OPENAVFF CORE with Framer Motion expansion */}
            <motion.div 
              layout
              transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
              className={`w-full flex flex-col items-center p-8 rounded-2xl border ${anatomyExpanded ? 'bg-[#0d1017]/80 backdrop-blur-xl border-[#00e5ff]/20 shadow-[0_0_50px_rgba(0,229,255,0.05)]' : 'bg-transparent border-transparent'}`}
            >
              {!anatomyExpanded ? (
                <ArchNode id="OPENAVFF" label="OPENAVFF" hero />
              ) : (
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3, duration: 0.5 }}
                  className="w-full flex flex-col items-center gap-14"
                >
                  <div className="w-full flex items-center justify-between border-b border-white/5 pb-4 mb-2">
                     <span className="font-mono text-xs tracking-[0.2em] text-[#00e5ff]">OPENAVFF / INTERNAL ANATOMY</span>
                     <button onClick={(e) => { e.stopPropagation(); setAnatomyExpanded(false); }} className="text-gray-500 hover:text-white transition-colors p-2 bg-white/5 hover:bg-white/10 rounded-full">
                        <X className="w-4 h-4"/>
                     </button>
                  </div>
                  
                  <div className="w-full flex justify-center gap-8 md:gap-48">
                    <ArchNode id="VISUAL_ENCODER" label="VISUAL ENCODER" />
                    <ArchNode id="AUDIO_ENCODER" label="AUDIO ENCODER" />
                  </div>

                  <div className="w-full flex justify-center gap-8 md:gap-24">
                    <ArchNode id="A2V" label="A2V" />
                    <ArchNode id="V2A" label="V2A" />
                  </div>

                  <ArchNode id="FEATURE_CONCAT" label="FEATURE CONCATENATION" />
                  <ArchNode id="MEAN_POOLING" label="MEAN POOLING" />
                  
                  <div className="w-full flex justify-center gap-8 md:gap-48">
                    <ArchNode id="MLP_VISION" label="MLP VISION" />
                    <ArchNode id="MLP_AUDIO" label="MLP AUDIO" />
                  </div>

                  <ArchNode id="MLP_HEAD" label="MODEL SIGNAL (HEAD)" />
                </motion.div>
              )}
            </motion.div>

            <div className="w-full flex flex-col md:flex-row justify-center items-center gap-16 md:gap-8 lg:gap-32">
              <ArchNode id="VISUAL_ANOMALY" label="VISUAL ANALYSIS" />
              <div className="w-40 md:w-56 h-4 opacity-0 hidden md:block"></div>
              <ArchNode id="METADATA" label="METADATA" />
            </div>

            <ArchNode id="FUSION" label="MEDIADNA FUSION" />

            <ArchNode id="RESULT" label="FORENSIC ASSESSMENT" hero />
          </div>
        </div>

        {/* ─── Inspector Drawer ─── */}
        <AnimatePresence>
          {inspectedNode && (
            <motion.div 
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              className="absolute right-0 top-0 bottom-0 w-full lg:w-[450px] border-l border-white/5 bg-[#0b0e14]/95 backdrop-blur-2xl shadow-[-20px_0_50px_rgba(0,0,0,0.5)] flex flex-col z-50"
            >
              <div className="p-6 border-b border-white/5 flex justify-between items-center bg-black/20">
                <div className="flex items-center gap-3">
                  <Info className="w-4 h-4 text-[#00e5ff]" />
                  <h3 className="font-mono text-[0.65rem] font-bold text-gray-300 tracking-[0.2em]">NODE INSPECTOR</h3>
                </div>
                <button onClick={() => setInspectedNode(null)} className="p-2 bg-white/5 hover:bg-white/10 rounded-full text-gray-400 hover:text-white transition-colors">
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              <div className="flex-1 p-8 overflow-y-auto custom-scrollbar">
                <div className="space-y-10">
                  <div>
                    <div className="font-mono text-[0.6rem] mb-2 font-bold tracking-[0.2em] uppercase" style={{ color: INSPECTOR_DB[inspectedNode]?.color }}>
                      {inspectedNode.replace('_', ' ')}
                    </div>
                    <h4 className="text-3xl font-extrabold text-white leading-tight tracking-tight">
                      {INSPECTOR_DB[inspectedNode]?.title}
                    </h4>
                  </div>

                  <div className="space-y-3">
                    <div className="font-mono text-[0.6rem] text-gray-500 font-bold tracking-[0.2em]">WHAT</div>
                    <p className="text-sm text-gray-300 leading-relaxed font-light">
                      {INSPECTOR_DB[inspectedNode]?.what}
                    </p>
                  </div>

                  <div className="p-6 bg-black/40 rounded-xl border border-white/5 space-y-5">
                    <div>
                      <span className="text-[0.6rem] font-bold font-mono text-gray-500 block mb-2 tracking-[0.2em]">INPUT</span>
                      <code className="text-xs text-[#00e5ff] block font-mono">{INSPECTOR_DB[inspectedNode]?.input}</code>
                    </div>
                    <div className="w-full h-px bg-white/5"></div>
                    <div>
                      <span className="text-[0.6rem] font-bold font-mono text-gray-500 block mb-2 tracking-[0.2em]">PROCESS</span>
                      <p className="text-xs text-gray-300 leading-relaxed">{INSPECTOR_DB[inspectedNode]?.process}</p>
                    </div>
                    <div className="w-full h-px bg-white/5"></div>
                    <div>
                      <span className="text-[0.6rem] font-bold font-mono text-gray-500 block mb-2 tracking-[0.2em]">OUTPUT</span>
                      <code className="text-xs text-[#b388ff] block font-mono">{INSPECTOR_DB[inspectedNode]?.output}</code>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="font-mono text-[0.6rem] text-gray-500 font-bold tracking-[0.2em]">WHY IT MATTERS</div>
                    <p className="text-sm text-gray-300 leading-relaxed border-l-2 pl-4 py-1" style={{ borderColor: INSPECTOR_DB[inspectedNode]?.color }}>
                      {INSPECTOR_DB[inspectedNode]?.why}
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
      
      <style>{`
        @keyframes particles {
          0% { stroke-dashoffset: 152; }
          100% { stroke-dashoffset: 0; }
        }
        .animate-particles {
          animation: particles linear infinite;
        }
        @keyframes trace {
          0% { stroke-dashoffset: 1000; }
          100% { stroke-dashoffset: 0; }
        }
        .animate-trace {
          animation: trace ease-out forwards;
        }
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
      `}</style>
    </div>
  );
}
