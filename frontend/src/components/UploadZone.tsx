import React from 'react';
import { Upload } from 'lucide-react';
import clsx from 'clsx';

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
    <div
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={clsx(
        'relative flex flex-col items-center justify-center w-full py-24 lg:py-32 transition-all duration-500 cursor-pointer group',
        isLoading && 'opacity-50 pointer-events-none'
      )}
      style={{
        border: `1px solid ${isDragOver ? 'var(--color-cyan-accent)' : 'var(--color-border-medium)'}`,
        background: isDragOver ? 'rgba(0, 212, 255, 0.03)' : 'rgba(255, 255, 255, 0.01)',
      }}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="video/mp4,video/x-m4v,video/*"
        onChange={handleChange}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-20"
        disabled={isLoading}
      />

      {/* Ambient glow */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <div className="w-48 h-48 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-700"
          style={{ background: 'radial-gradient(circle, var(--color-cyan-glow), transparent 70%)' }}></div>
      </div>

      {/* Reticle corners */}
      <div className="reticle-corner reticle-tl group-hover:!border-[var(--color-cyan-accent)]" style={{ borderColor: 'var(--color-border-strong)' }}></div>
      <div className="reticle-corner reticle-tr group-hover:!border-[var(--color-cyan-accent)]" style={{ borderColor: 'var(--color-border-strong)' }}></div>
      <div className="reticle-corner reticle-bl group-hover:!border-[var(--color-cyan-accent)]" style={{ borderColor: 'var(--color-border-strong)' }}></div>
      <div className="reticle-corner reticle-br group-hover:!border-[var(--color-cyan-accent)]" style={{ borderColor: 'var(--color-border-strong)' }}></div>

      <div className="relative z-10 flex flex-col items-center gap-5">
        {isDragOver ? (
          <>
            <Upload className="w-10 h-10" style={{ color: 'var(--color-cyan-accent)' }} />
            <h2 className="text-xl font-semibold text-cyan-accent">Drop to inspect</h2>
          </>
        ) : (
          <>
            <div className="p-4 rounded surface-3 group-hover:border-[var(--color-cyan-accent)] transition-colors" style={{ borderColor: 'var(--color-border-medium)' }}>
              <Upload className="w-8 h-8" style={{ color: 'var(--color-text-muted)' }} />
            </div>
            <div className="text-center">
              <h2 className="text-lg font-semibold mb-2" style={{ color: 'var(--color-text-primary)' }}>Drop media here</h2>
              <p className="tech-label">or click to select • video / audio input</p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
