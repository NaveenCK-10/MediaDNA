import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { ArrowLeft, Plus, Archive } from 'lucide-react';
import { motion } from 'framer-motion';
import ResultCard from '../components/ResultCard';
import type { HistoryItem } from '../types';

export default function CaseDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [caseData, setCaseData] = useState<HistoryItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    const fetchCase = async () => {
      try {
        const res = await axios.get('/api/history');
        const items: HistoryItem[] = res.data;
        const found = items.find(item => item.id === id);
        if (found) {
          setCaseData(found);
        } else {
          setNotFound(true);
        }
      } catch (err) {
        console.error('Failed to fetch case', err);
        setNotFound(true);
      } finally {
        setLoading(false);
      }
    };
    fetchCase();
  }, [id]);

  const formatDate = (ts: number) => {
    const d = new Date(ts * 1000);
    return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }).toUpperCase()
      + ' · ' + d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-6">
        <div className="w-12 h-12 rounded-full border-2 border-white/10 border-t-[#00e5ff] animate-[spin_1s_linear_infinite]"></div>
        <div className="font-mono text-[0.65rem] font-bold tracking-[0.3em] text-gray-400">LOADING CASE FILE</div>
      </div>
    );
  }

  if (notFound || !caseData) {
    return (
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-3xl mx-auto px-6 py-32 text-center"
      >
        <div className="glass-card p-12 rounded-xl border border-white/5 flex flex-col items-center justify-center">
          <Archive className="w-12 h-12 text-gray-700 mb-4" />
          <h2 className="font-mono text-xl font-bold tracking-widest text-white mb-2">CASE NOT FOUND</h2>
          <p className="text-sm mb-8 text-gray-500 font-mono">The requested forensic record does not exist or has been purged.</p>
          <button 
            onClick={() => navigate('/history')} 
            data-hover="node"
            className="border border-white/10 hover:border-[#00e5ff]/50 bg-white/5 hover:bg-[#00e5ff]/10 text-white font-mono text-[0.65rem] font-bold tracking-[0.2em] transition-all px-6 py-3 flex items-center justify-center gap-2 uppercase rounded"
          >
            <ArrowLeft className="w-4 h-4" /> RETURN TO CASE ARCHIVE
          </button>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="max-w-5xl mx-auto px-6 py-12"
    >
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-6 mb-12 pb-6 border-b border-white/10">
        <div>
          <div className="font-mono text-[0.65rem] font-bold tracking-[0.3em] text-[#ffb300] mb-2 flex items-center gap-2">
            <Archive className="w-4 h-4" /> ARCHIVED FORENSIC CASE
          </div>
          <h1 className="editorial-headline text-white break-all" style={{ fontSize: 'clamp(1.5rem, 3vw, 2rem)' }}>
            <span className="text-[#00e5ff] mr-2">TARGET:</span> 
            {caseData.video_filename || caseData.filename}
          </h1>
          <div className="font-mono text-[0.65rem] tracking-[0.1em] text-gray-500 mt-2">{formatDate(caseData.timestamp)}</div>
        </div>
        
        <div className="flex flex-wrap gap-3 shrink-0">
          <button 
            onClick={() => navigate('/history')} 
            data-hover="node"
            className="border border-white/10 hover:border-white/30 bg-white/5 hover:bg-white/10 text-gray-300 hover:text-white font-mono text-[0.65rem] font-bold tracking-[0.2em] transition-all px-4 py-2 flex items-center justify-center gap-2 uppercase rounded"
          >
            <ArrowLeft className="w-3 h-3" /> ARCHIVE
          </button>
          <button 
            onClick={() => navigate('/analyze')} 
            data-hover="node"
            className="bg-[#00e5ff]/10 hover:bg-[#00e5ff]/20 text-[#00e5ff] border border-[#00e5ff]/30 hover:border-[#00e5ff]/60 font-mono text-[0.65rem] font-bold tracking-[0.2em] transition-all px-4 py-2 flex items-center justify-center gap-2 uppercase rounded"
          >
            <Plus className="w-3 h-3" /> NEW ANALYSIS
          </button>
        </div>
      </div>

      <ResultCard result={caseData} isArchived />
    </motion.div>
  );
}
