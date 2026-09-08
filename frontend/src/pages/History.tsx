import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Trash2, ArrowRight, ShieldCheck, ShieldAlert, Archive, FileVideo } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import type { HistoryItem } from '../types';

export default function HistoryPage() {
  const navigate = useNavigate();
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await axios.get('/api/history');
        setHistory(res.data);
      } catch (err) {
        console.error('Failed to fetch history', err);
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  const clearHistory = async () => {
    try {
      await axios.delete('/api/history');
      setHistory([]);
    } catch (err) {
      console.error('Failed to clear history');
    }
  };

  const deleteItem = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await axios.delete(`/api/history/${id}`);
      setHistory(prev => prev.filter(h => h.id !== id));
    } catch (err) {
      console.error('Failed to delete item');
    }
  };

  const formatDate = (ts: number) => {
    const d = new Date(ts * 1000);
    return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }).toUpperCase()
      + ' · ' + d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end mb-12 pb-6 border-b border-white/10 gap-4">
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="font-mono text-[0.65rem] font-bold tracking-[0.3em] text-[#00e5ff] mb-2 flex items-center gap-2">
            <Archive className="w-4 h-4" /> ARCHIVE ACCESS
          </div>
          <h1 className="editorial-headline text-white" style={{ fontSize: 'clamp(2rem, 4vw, 3rem)' }}>
            Forensic Case <span className="text-[#00e5ff]">Archive</span>
          </h1>
          <p className="mt-2 text-sm text-gray-400 font-mono">Immutable record of all processed target media.</p>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
        >
          {history.length > 0 && (
            <button onClick={clearHistory} data-hover="node" className="btn-danger flex items-center gap-2 py-2 px-4 rounded border border-red-500/30 text-red-500 hover:bg-red-500/10 font-mono text-xs font-bold tracking-[0.2em] uppercase transition-all">
              <Trash2 className="w-3.5 h-3.5" /> PURGE ARCHIVE
            </button>
          )}
        </motion.div>
      </div>

      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div 
            key="loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center justify-center py-32 gap-6"
          >
            <div className="w-12 h-12 rounded-full border-2 border-white/10 border-t-[#00e5ff] animate-[spin_1s_linear_infinite]"></div>
            <div className="font-mono text-[0.65rem] font-bold tracking-[0.3em] text-gray-400">ACCESSING RECORDS</div>
          </motion.div>
        ) : history.length === 0 ? (
          <motion.div 
            key="empty"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center py-32 glass-card rounded-xl border border-white/5 flex flex-col items-center justify-center"
          >
            <Archive className="w-12 h-12 text-gray-700 mb-4" />
            <p className="font-mono text-sm font-bold tracking-[0.2em] text-gray-500">NO ARCHIVED CASES</p>
            <p className="text-xs mt-2 text-gray-600 font-mono">The forensic archive is currently empty.</p>
          </motion.div>
        ) : (
          <motion.div 
            key="list"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="relative ml-2 sm:ml-8 pb-12 before:absolute before:left-0 before:top-2 before:bottom-0 before:w-px before:bg-gradient-to-b before:from-[#00e5ff]/50 before:to-transparent"
          >
            {history.map((item, idx) => {
              const isFake = item.prediction === 'fake';
              const caseNum = String(history.length - idx).padStart(4, '0');
              const color = isFake ? '#ef4444' : '#10b981';

              return (
                <motion.div 
                  key={item.id} 
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05, duration: 0.4 }}
                  className="relative pl-8 sm:pl-12 pb-8 group"
                >
                  {/* Timeline Node */}
                  <div className="absolute left-0 top-3 w-4 h-4 rounded-full border-[3px] group-hover:scale-125 transition-transform bg-[#06080d] z-10"
                    style={{ borderColor: color, transform: 'translateX(-50%)' }}></div>

                  {/* Connecting Line Glow */}
                  <div className="absolute left-0 top-7 bottom-0 w-px transition-opacity opacity-0 group-hover:opacity-100 z-0"
                    style={{ background: `linear-gradient(to bottom, ${color}, transparent)`, transform: 'translateX(-50%)' }}></div>

                  <div
                    onClick={() => navigate(`/history/${item.id}`)}
                    data-hover="node"
                    className="w-full text-left glass-card p-6 flex flex-col md:flex-row md:items-center justify-between gap-6 cursor-pointer border border-white/5 hover:border-white/20 transition-all rounded-xl"
                  >
                    <div className="min-w-0 space-y-3">
                      <div className="flex items-center gap-3">
                        <div className="font-mono text-[0.6rem] font-bold tracking-[0.2em] bg-white/5 px-2 py-1 rounded text-white">CASE {caseNum}</div>
                        <div className="w-1 h-1 rounded-full bg-gray-700"></div>
                        <div className="font-mono text-[0.6rem] tracking-[0.1em] text-gray-500">{formatDate(item.timestamp)}</div>
                      </div>
                      <div className="flex items-center gap-3 text-white">
                        <FileVideo className="w-4 h-4 text-gray-500 group-hover:text-[#00e5ff] transition-colors" />
                        <span className="text-base font-bold truncate group-hover:text-[#00e5ff] transition-colors tracking-tight">
                          {item.video_filename || item.filename}
                        </span>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-8 shrink-0 bg-black/40 px-6 py-3 rounded-lg border border-white/5">
                      <div>
                        <div className="font-mono text-[0.55rem] font-bold tracking-[0.2em] text-gray-500 mb-1">MODEL SIGNAL</div>
                        <span className="font-mono text-sm font-bold text-white">{(item.openavff_fake_prob * 100).toFixed(1)}%</span>
                      </div>
                      
                      <div className="w-px h-8 bg-white/10"></div>
                      
                      <div className="flex flex-col items-start min-w-[100px]">
                        <div className="font-mono text-[0.55rem] font-bold tracking-[0.2em] text-gray-500 mb-1">ASSESSMENT</div>
                        <div className="flex items-center gap-2">
                          {isFake ? <ShieldAlert className="w-4 h-4" style={{ color }} /> : <ShieldCheck className="w-4 h-4" style={{ color }} />}
                          <span className="font-mono text-xs font-bold tracking-widest" style={{ color }}>
                            {item.prediction.toUpperCase()}
                          </span>
                        </div>
                      </div>

                      <button 
                        onClick={(e) => deleteItem(e, item.id)} 
                        className="p-2 opacity-0 group-hover:opacity-100 transition-opacity ml-4 rounded hover:bg-red-500/20 text-gray-600 hover:text-red-500"
                        title="Delete Record"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                      
                      <ArrowRight className="w-5 h-5 opacity-0 group-hover:opacity-100 transition-opacity text-gray-600 group-hover:text-white" />
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
