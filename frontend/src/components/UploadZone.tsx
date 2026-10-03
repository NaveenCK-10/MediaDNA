import React from 'react';
import { Upload, ScanLine } from 'lucide-react';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  isLoading?: boolean;
  fileInputRef?: React.RefObject<HTMLInputElement | null>;
}

export default function UploadZone({ onFileSelect, isLoading, fileInputRef }: UploadZoneProps) {
  const [isDragOver, setIsDragOver] = React.useState(false);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (isLoading) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isLoading) setIsDragOver(true);
  };

  const handleDragLeave = () => setIsDragOver(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelect(e.target.files[0]);
    }
  };

  return (
    <div className="relative group">
      
      {/* Target Intake Header */}
      <div className="absolute -top-4 left-6 px-2 bg-[#06080d] z-30 font-mono text-[0.65rem] tracking-widest text-cyan-500 font-bold flex items-center gap-2">
         <ScanLine className="w-3 h-3" />
         TARGET INTAKE
      </div>

      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={clsx(
          'relative flex flex-col items-center justify-center w-full py-24 lg:py-32 transition-all duration-700 cursor-pointer overflow-hidden rounded-xl border',
          isLoading && 'opacity-50 pointer-events-none',
          isDragOver ? 'border-cyan-400/80 bg-cyan-900/10' : 'border-white/10 bg-black/40 hover:border-cyan-500/50 hover:bg-cyan-900/5'
        )}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4,video/x-m4v,video/*,audio/*"
          onChange={handleChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
          disabled={isLoading}
        />

        {/* Ambient glow */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className={`w-[600px] h-[600px] rounded-full transition-opacity duration-700 ${isDragOver ? 'opacity-30' : 'opacity-0 group-hover:opacity-10'}`}
            style={{ background: 'radial-gradient(circle, var(--color-cyan-glow), transparent 70%)' }}></div>
        </div>

        {/* Subtle animated border on drag over */}
        <AnimatePresence>
          {isDragOver && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-0 pointer-events-none"
            >
               <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent animate-[scanline_2s_linear_infinite]" />
               <div className="absolute bottom-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent animate-[scanline_2s_linear_infinite_reverse]" />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Reticle corners */}
        <div className={`reticle-corner reticle-tl transition-colors duration-500 ${isDragOver ? 'border-cyan-400' : 'border-white/20'}`}></div>
        <div className={`reticle-corner reticle-tr transition-colors duration-500 ${isDragOver ? 'border-cyan-400' : 'border-white/20'}`}></div>
        <div className={`reticle-corner reticle-bl transition-colors duration-500 ${isDragOver ? 'border-cyan-400' : 'border-white/20'}`}></div>
        <div className={`reticle-corner reticle-br transition-colors duration-500 ${isDragOver ? 'border-cyan-400' : 'border-white/20'}`}></div>

        <div className="relative z-10 flex flex-col items-center gap-6">
          {isDragOver ? (
            <motion.div 
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="flex flex-col items-center gap-4"
            >
              <Upload className="w-12 h-12 text-cyan-400 animate-bounce" />
              <div className="font-mono text-sm tracking-[0.2em] font-bold text-cyan-400 uppercase">Awaiting Target Payload</div>
            </motion.div>
          ) : (
            <>
              <div className="p-5 rounded-full border border-white/5 bg-white/5 group-hover:border-cyan-500/30 group-hover:bg-cyan-500/10 transition-colors shadow-xl">
                <Upload className="w-8 h-8 text-gray-400 group-hover:text-cyan-400 transition-colors" />
              </div>
              <div className="text-center">
                <h2 className="text-xl font-bold mb-3 text-white tracking-wide">INITIALIZE INTAKE</h2>
                <p className="font-mono text-[0.65rem] tracking-widest text-gray-500 uppercase">
                  DRAG & DROP MEDIA HERE OR CLICK TO BROWSE
                </p>
              </div>
            </>
          )}
        </div>
        
        {/* Media status line */}
        <div className="absolute bottom-4 left-6 right-6 flex justify-between items-end font-mono text-[0.55rem] tracking-widest text-white/30 uppercase pointer-events-none">
           <div className="flex gap-4">
              <span>SUPPORTED: MP4, MOV, WAV, AVI</span>
              <span className="hidden sm:inline">•</span>
              <span className="hidden sm:inline">MAX: 500MB</span>
           </div>
           <div className="flex items-center gap-2">
              <span className="w-1 h-1 rounded-full bg-cyan-500/50 animate-pulse"></span>
              SYSTEM READY
           </div>
        </div>
      </div>
    </div>
  );
}
