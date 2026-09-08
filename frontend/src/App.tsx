import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import axios from 'axios';
import { ServerCrash } from 'lucide-react';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import Analyze from './pages/Analyze';
import HistoryPage from './pages/History';
import CaseDetail from './pages/CaseDetail';
import Architecture from './pages/Architecture';
import Research from './pages/Research';

function AnimatedRoutes() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<PageWrapper><Dashboard /></PageWrapper>} />
        <Route path="/analyze" element={<PageWrapper><Analyze /></PageWrapper>} />
        <Route path="/history" element={<PageWrapper><HistoryPage /></PageWrapper>} />
        <Route path="/history/:id" element={<PageWrapper><CaseDetail /></PageWrapper>} />
        <Route path="/architecture" element={<PageWrapper><Architecture /></PageWrapper>} />
        <Route path="/research" element={<PageWrapper><Research /></PageWrapper>} />
      </Routes>
    </AnimatePresence>
  );
}

function PageWrapper({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10, filter: 'blur(4px)' }}
      animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      exit={{ opacity: 0, y: -10, filter: 'blur(4px)' }}
      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
      className="h-full"
    >
      {children}
    </motion.div>
  );
}

function SystemOffline() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#06080d] bg-opacity-95 backdrop-blur-md">
      <div className="text-center space-y-6 p-8 border border-red-500/30 bg-red-500/5 rounded-xl max-w-lg shadow-2xl">
        <ServerCrash className="w-16 h-16 text-red-500 mx-auto animate-pulse" />
        <div>
          <h2 className="font-mono text-2xl font-bold tracking-widest text-white mb-2 uppercase">System Offline</h2>
          <div className="w-12 h-1 bg-red-500 mx-auto mb-4"></div>
          <p className="text-sm text-gray-400 font-mono leading-relaxed">
            The MediaDNA backend inference engine is currently unreachable or initializing. 
          </p>
          <p className="text-xs text-red-400/80 font-mono mt-4">
            Awaiting connection...
          </p>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await axios.get('/api/health', { timeout: 3000 });
        setIsOnline(true);
      } catch (err) {
        setIsOnline(false);
      }
    };
    
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col relative z-10">
        {!isOnline && <SystemOffline />}
        
        <Header />

        <main className="flex-1 pt-20">
          <AnimatedRoutes />
        </main>

        <footer className="border-t border-gray-800 py-8 text-center bg-[#06080d]">
          <p className="tech-mono text-[0.6rem] text-gray-500">
            MEDIA DNA — AI-POWERED AUDIO-VISUAL DEEPFAKE DETECTION — POWERED BY OPENAVFF
          </p>
        </footer>
      </div>
    </BrowserRouter>
  );
}

export default App;
