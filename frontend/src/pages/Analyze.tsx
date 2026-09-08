import { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import axios from 'axios';
import { AlertTriangle, ArrowRight, RotateCcw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import UploadZone from '../components/UploadZone';
import ProcessingView from '../components/ProcessingView';
import ResultCard from '../components/ResultCard';
import type { AnalysisResponse } from '../types';

type AnalyzeState = 'upload' | 'preview' | 'processing' | 'result';

export default function Analyze() {
  const location = useLocation();
  const [state, setState] = useState<AnalyzeState>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [videoDuration, setVideoDuration] = useState<string>('—');
  const [videoResolution, setVideoResolution] = useState<string>('—');
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle demo or history navigation
  useEffect(() => {
    if (location.state?.historyItem) {
      setResult(location.state.historyItem);
      setState('result');
      window.history.replaceState({}, document.title);
    } else if (location.state?.demoType) {
      runDemo(location.state.demoType);
      window.history.replaceState({}, document.title);
    }
  }, [location]);

  // Clean up object URL
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileSelect = (selectedFile: File) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);

    setFile(selectedFile);
    setError(null);
    setResult(null);
    setVideoDuration('—');
    setVideoResolution('—');

    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    setState('preview');
  };

  const handleChangeMedia = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setVideoDuration('—');
    setVideoResolution('—');
    setState('upload');

    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleVideoLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    const dur = video.duration;
    const mins = Math.floor(dur / 60);
    const secs = Math.floor(dur % 60);
    setVideoDuration(`${mins}:${secs.toString().padStart(2, '0')}`);
    if (video.videoWidth && video.videoHeight) {
      setVideoResolution(`${video.videoWidth}×${video.videoHeight}`);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setState('processing');
    setError(null);

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await axios.post<AnalysisResponse>('/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
      setState('result');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to analyze video. Ensure the backend is running.');
      setState('preview');
    }
  };

  const runDemo = async (type: string) => {
    setState('processing');
    setError(null);
    setResult(null);

    try {
      const response = await axios.post<AnalysisResponse>(`/api/analyze-demo?type=${type}`);
      setResult(response.data);
      setState('result');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to run demo analysis.');
      setState('upload');
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
    if (bytes >= 1_000) return `${(bytes / 1_000).toFixed(0)} KB`;
    return `${bytes} B`;
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 min-h-[calc(100vh-10rem)] relative">
      <AnimatePresence mode="wait">
        
        {/* ─── UPLOAD STATE ─── */}
        {state === 'upload' && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <div className="mb-8">
              <div className="tech-label mb-3 text-cyan-accent tracking-[0.2em]">FORENSIC ANALYSIS ENGINE</div>
              <h1 className="editorial-headline text-white" style={{ fontSize: 'clamp(2rem, 4vw, 3rem)' }}>
                Begin Forensic Analysis
              </h1>
              <p className="mt-4 text-sm text-gray-400 max-w-xl leading-relaxed">
                Insert media into the forensic engine for deep audio-visual authenticity verification. The OpenAVFF core will evaluate synchronization, visual integrity, and metadata artifacts.
              </p>
            </div>

            <UploadZone onFileSelect={handleFileSelect} fileInputRef={fileInputRef} />

            {error && (
              <motion.div 
                initial={{ opacity: 0, height: 0 }} 
                animate={{ opacity: 1, height: 'auto' }} 
                className="mt-6 p-4 flex items-start gap-3 border border-red-500/30 bg-red-500/10 rounded"
              >
                <AlertTriangle className="w-4 h-4 shrink-0 text-red-500 mt-1" />
                <div>
                  <div className="font-mono text-[0.65rem] font-bold text-red-500 mb-1 tracking-wider">ANALYSIS ERROR</div>
                  <p className="text-sm text-gray-300">{error}</p>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {/* ─── PREVIEW STATE ─── */}
        {state === 'preview' && file && previewUrl && (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.02 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="flex flex-col gap-8"
          >
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6 pb-6 border-b border-gray-800">
              <div>
                <div className="font-mono text-[0.65rem] font-bold text-[#00e5ff] tracking-[0.2em] mb-2">MEDIA INSPECTION WORKSTATION</div>
                <div className="font-mono text-sm text-white flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)] animate-pulse"></div>
                  TARGET LOADED SUCCESSFULLY
                </div>
              </div>
              <div className="flex gap-4 w-full sm:w-auto">
                <button onClick={handleChangeMedia} className="btn-secondary flex-1 sm:flex-none justify-center" data-hover="node">
                  <RotateCcw className="w-4 h-4" /> ABORT
                </button>
                <button onClick={handleAnalyze} className="btn-primary flex-1 sm:flex-none justify-center" data-hover="node">
                  BEGIN INFERENCE <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Video Viewport */}
            <div className="relative w-full bg-[#06080d] border border-gray-800 rounded-lg overflow-hidden shadow-2xl reticle-container" data-hover="media">
              <div className="reticle-corner reticle-tl" />
              <div className="reticle-corner reticle-tr" />
              <div className="reticle-corner reticle-bl" />
              <div className="reticle-corner reticle-br" />

              <video
                ref={videoRef}
                src={previewUrl}
                controls
                controlsList="nodownload nofullscreen"
                onLoadedMetadata={handleVideoLoadedMetadata}
                className="w-full max-h-[60vh] object-contain opacity-90 hover:opacity-100 transition-opacity duration-300"
                style={{ display: 'block' }}
              />
              
              {/* Scanline overlay */}
              <div className="absolute inset-0 pointer-events-none overflow-hidden opacity-20 mix-blend-overlay">
                <div className="w-full h-[10%] bg-gradient-to-b from-transparent via-[#00e5ff] to-transparent animate-[scanline_4s_linear_infinite]" />
              </div>
            </div>

            {/* File Metadata */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {[
                { label: 'FILENAME', value: file.name },
                { label: 'SIZE', value: formatFileSize(file.size) },
                { label: 'MIME TYPE', value: file.type || 'UNKNOWN' },
                { label: 'DURATION', value: videoDuration },
                { label: 'RESOLUTION', value: videoResolution },
              ].map(({ label, value }) => (
                <div key={label} className="glass-card p-4 rounded-lg flex flex-col justify-center">
                  <div className="font-mono text-[0.6rem] text-gray-500 font-bold tracking-[0.1em] mb-2">{label}</div>
                  <div className="font-mono text-sm text-gray-200 truncate" title={value}>{value}</div>
                </div>
              ))}
            </div>

            {error && (
              <div className="mt-4 p-4 flex items-start gap-3 border border-red-500/30 bg-red-500/10 rounded">
                <AlertTriangle className="w-4 h-4 shrink-0 text-red-500 mt-1" />
                <p className="text-sm text-gray-300">{error}</p>
              </div>
            )}
          </motion.div>
        )}

        {/* ─── PROCESSING STATE ─── */}
        {state === 'processing' && (
          <motion.div
            key="processing"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.05 }}
            transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          >
            <ProcessingView filename={file?.name || 'DEMO_VIDEO.mp4'} />
          </motion.div>
        )}

        {/* ─── RESULT STATE ─── */}
        {state === 'result' && result && (
          <motion.div
            key="result"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
          >
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-10 pb-6 border-b border-gray-800">
              <motion.div 
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2, duration: 0.6 }}
              >
                <div className="font-mono text-[0.65rem] font-bold tracking-[0.2em] text-[#00e5ff] mb-2">FORENSIC ASSESSMENT COMPLETE</div>
                <h2 className="text-xl md:text-2xl font-bold text-white font-mono tracking-tight">
                  <span className="text-gray-500">TARGET:</span> {result.video_filename}
                </h2>
              </motion.div>
              <motion.button 
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.4, duration: 0.4 }}
                onClick={handleChangeMedia} 
                className="btn-secondary"
                data-hover="node"
              >
                <RotateCcw className="w-4 h-4" /> NEW ANALYSIS
              </motion.button>
            </div>

            {/* Cinematic Reveal Sequence handled internally by ResultCard if needed, or we just fade it in elegantly */}
            <motion.div
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            >
              <ResultCard result={result} />
            </motion.div>
          </motion.div>
        )}

      </AnimatePresence>
    </div>
  );
}
