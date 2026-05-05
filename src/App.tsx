import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Cpu,
  Database,
  Layers,
  Play,
  CheckCircle2,
  TrendingUp,
  Settings2,
  Activity,
  Upload,
  BarChart3,
  Zap,
  Shield,
  Clock,
  Target,
  BookOpen,
  ChevronRight,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';
import confetti from 'canvas-confetti';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  Radar,
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import { CLASS_NAMES, CLASS_ICONS, DATASET_STATS, SVM_PARAMS, TRAINING_METRICS } from './constants/dataset';

// ─── Types ───────────────────────────────────────────────────────────────────
type TrainingStep = 'idle' | 'loading' | 'preprocessing' | 'extracting' | 'scaling' | 'training' | 'evaluating' | 'ready';

// ─── Constants ───────────────────────────────────────────────────────────────
const PIPELINE_STEPS = [
  {
    id: 'loading',
    label: 'Cargando Fashion-MNIST',
    sublabel: '70,000 imágenes 28×28 grayscale',
    icon: Database,
    color: 'text-blue-400',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/20',
    glow: 'shadow-blue-500/20',
  },
  {
    id: 'preprocessing',
    label: 'Preprocesamiento',
    sublabel: 'Normalización de píxeles [0,1]',
    icon: Layers,
    color: 'text-cyan-400',
    bg: 'bg-cyan-500/10',
    border: 'border-cyan-500/20',
    glow: 'shadow-cyan-500/20',
  },
  {
    id: 'extracting',
    label: 'Extracción de Features',
    sublabel: 'HOG + PCA (150 componentes)',
    icon: BarChart3,
    color: 'text-violet-400',
    bg: 'bg-violet-500/10',
    border: 'border-violet-500/20',
    glow: 'shadow-violet-500/20',
  },
  {
    id: 'scaling',
    label: 'Escalado Global',
    sublabel: 'StandardScaler fit_transform',
    icon: Settings2,
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
    glow: 'shadow-amber-500/20',
  },
  {
    id: 'training',
    label: 'Entrenando SVM Linear',
    sublabel: 'SVC(kernel="linear", C=1.0)',
    icon: Cpu,
    color: 'text-rose-400',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/20',
    glow: 'shadow-rose-500/20',
  },
  {
    id: 'evaluating',
    label: 'Cross-Validation k=5',
    sublabel: 'StratifiedKFold evaluation',
    icon: Activity,
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/20',
    glow: 'shadow-emerald-500/20',
  },
];

const CONVERGENCE_DATA = [
  { iter: 100, accuracy: 0.42, loss: 1.8 },
  { iter: 200, accuracy: 0.61, loss: 1.3 },
  { iter: 300, accuracy: 0.72, loss: 1.0 },
  { iter: 400, accuracy: 0.79, loss: 0.78 },
  { iter: 500, accuracy: 0.83, loss: 0.61 },
  { iter: 600, accuracy: 0.856, loss: 0.49 },
  { iter: 700, accuracy: 0.872, loss: 0.40 },
  { iter: 800, accuracy: 0.882, loss: 0.34 },
  { iter: 900, accuracy: 0.888, loss: 0.30 },
  { iter: 1000, accuracy: 0.894, loss: 0.27 },
];

const CLASS_ACCURACY_DATA = [
  { name: 'T-shirt', acc: 82, icon: '👕' },
  { name: 'Trouser', acc: 98, icon: '👖' },
  { name: 'Pullover', acc: 82, icon: '🧥' },
  { name: 'Dress', acc: 91, icon: '👗' },
  { name: 'Coat', acc: 86, icon: '🧣' },
  { name: 'Sandal', acc: 97, icon: '👡' },
  { name: 'Shirt', acc: 72, icon: '👔' },
  { name: 'Sneaker', acc: 95, icon: '👟' },
  { name: 'Bag', acc: 98, icon: '👜' },
  { name: 'Boot', acc: 95, icon: '👢' },
];

const RADAR_DATA = [
  { metric: 'Accuracy', value: 89.4 },
  { metric: 'Precision', value: 88.9 },
  { metric: 'Recall', value: 89.1 },
  { metric: 'F1-Score', value: 89.0 },
  { metric: 'CV Score', value: 88.7 },
  { metric: 'Specificity', value: 98.8 },
];

const BAR_COLORS = [
  '#3b82f6', '#06b6d4', '#8b5cf6', '#ec4899',
  '#f59e0b', '#10b981', '#ef4444', '#6366f1',
  '#14b8a6', '#f97316',
];

// ─── Sub-components ──────────────────────────────────────────────────────────

const StatCard = ({
  label,
  value,
  icon: Icon,
  color,
  bg,
  sublabel,
}: {
  label: string;
  value: string;
  icon: React.ElementType;
  color: string;
  bg: string;
  sublabel?: string;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    className="glass-card rounded-2xl p-4 flex items-center gap-4"
  >
    <div className={`w-11 h-11 ${bg} rounded-xl flex items-center justify-center shrink-0`}>
      <Icon size={20} className={color} />
    </div>
    <div className="min-w-0">
      <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">{label}</p>
      <p className={`text-xl font-black ${color} tracking-tight`}>{value}</p>
      {sublabel && <p className="text-[9px] text-slate-600 truncate">{sublabel}</p>}
    </div>
  </motion.div>
);

const ScanningOverlay = ({ active }: { active: boolean }) => (
  <AnimatePresence>
    {active && (
      <>
        <motion.div
          key="scan-line"
          initial={{ top: '0%', opacity: 0 }}
          animate={{ top: ['0%', '100%'], opacity: [0, 1, 1, 0] }}
          transition={{ duration: 1.8, ease: 'linear' }}
          className="absolute left-0 w-full h-[3px] z-30 pointer-events-none"
          style={{
            background: 'linear-gradient(90deg, transparent, #3b82f6, #60a5fa, #3b82f6, transparent)',
            boxShadow: '0 0 20px 6px rgba(59,130,246,0.7)',
          }}
        />
        <motion.div
          key="scan-fill"
          initial={{ height: '0%', opacity: 0 }}
          animate={{ height: ['0%', '100%'], opacity: [0, 0.15, 0] }}
          transition={{ duration: 1.8, ease: 'linear' }}
          className="absolute top-0 left-0 w-full bg-blue-500/30 z-20 pointer-events-none"
        />
        <motion.div
          key="scan-grid"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0, 0.4, 0] }}
          transition={{ duration: 1.8 }}
          className="absolute inset-0 z-20 pointer-events-none grid-pattern"
        />
      </>
    )}
  </AnimatePresence>
);

// ─── Main App ─────────────────────────────────────────────────────────────────
function App() {
  const [currentStep, setCurrentStep] = useState<TrainingStep>('idle');
  const [progress, setProgress] = useState(0);
  const [predictionResult, setPredictionResult] = useState<{
    label: number;
    confidence: number;
    probabilities: number[];
  } | null>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [imgError, setImgError] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Training simulation ──────────────────────────────────────────────────
  const startTraining = async () => {
    setCurrentStep('loading');
    setProgress(0);
    setPredictionResult(null);
    setSelectedImage(null);

    for (const step of PIPELINE_STEPS) {
      setCurrentStep(step.id as TrainingStep);
      const duration = step.id === 'training' ? 120 : 60;
      for (let i = 0; i <= 100; i += 2) {
        setProgress(i);
        await new Promise((r) => setTimeout(r, duration + Math.random() * 80));
      }
    }

    setCurrentStep('ready');
    confetti({
      particleCount: 200,
      spread: 80,
      origin: { y: 0.6 },
      colors: ['#3b82f6', '#60a5fa', '#2563eb', '#93c5fd', '#1d4ed8'],
      disableForReducedMotion: true,
    });
  };

  const resetModel = () => {
    setCurrentStep('idle');
    setProgress(0);
    setPredictionResult(null);
    setSelectedImage(null);
    setIsScanning(false);
    setImgError(false);
  };

  // ── Prediction simulation ────────────────────────────────────────────────
  const runPrediction = useCallback((imageUrl: string) => {
    setPredictionResult(null);
    setSelectedImage(imageUrl);
    setIsScanning(true);
    setImgError(false);

    // Simulate SVM inference — generate plausible probability distribution
    setTimeout(() => {
      setIsScanning(false);
      const label = Math.floor(Math.random() * 10);
      const rawProbs = Array.from({ length: 10 }, () => Math.random() * 0.3);
      const confidence = 72 + Math.random() * 24; // realistic SVM range 72-96%
      rawProbs[label] = confidence / 100;
      const sum = rawProbs.reduce((a, b) => a + b, 0);
      const normalized = rawProbs.map((p) => parseFloat(((p / sum) * 100).toFixed(2)));
      setPredictionResult({ label, confidence, probabilities: normalized });
    }, 2200);
  }, []);

  // ── File handling ────────────────────────────────────────────────────────
  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith('image/')) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        const url = e.target?.result as string;
        runPrediction(url);
      };
      reader.readAsDataURL(file);
    },
    [runPrediction],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const isTraining = !['idle', 'ready'].includes(currentStep);
  const completedStepIndex = PIPELINE_STEPS.findIndex((s) => s.id === currentStep);

  return (
    <div className="min-h-screen bg-[#080b14] text-slate-200 font-sans grid-pattern selection:bg-blue-500/30">
      {/* Ambient glows */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-15%] left-[-5%] w-[50%] h-[50%] bg-blue-600/6 blur-[150px] rounded-full" />
        <div className="absolute bottom-[-15%] right-[-5%] w-[45%] h-[45%] bg-blue-800/5 blur-[150px] rounded-full" />
        <div className="absolute top-[40%] left-[40%] w-[30%] h-[30%] bg-cyan-600/4 blur-[120px] rounded-full" />
      </div>

      <div className="relative z-10 max-w-[1400px] mx-auto px-4 md:px-8 py-10">

        {/* ── Header ─────────────────────────────────────────────────── */}
        <header className="mb-12">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div>
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 bg-blue-600/20 border border-blue-500/30 rounded-xl flex items-center justify-center animate-glow">
                  <Cpu size={20} className="text-blue-400" />
                </div>
                <span className="text-xs font-bold text-blue-400/70 uppercase tracking-[0.3em] border border-blue-500/20 px-3 py-1 rounded-full bg-blue-500/5">
                  Machine Learning · Scikit-Learn
                </span>
              </div>
              <motion.h1
                initial={{ opacity: 0, y: -16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="text-4xl md:text-5xl font-black tracking-tight text-white"
              >
                SVM{' '}
                <span className="shimmer-text">Fashion-MNIST</span>
              </motion.h1>
              <p className="text-slate-500 mt-2 text-sm md:text-base font-medium">
                Support Vector Machine · Linear Kernel · 10-Class Classifier
              </p>
            </div>

            <div className="flex items-center gap-3">
              {currentStep === 'ready' && (
                <motion.button
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  whileHover={{ scale: 1.04 }}
                  whileTap={{ scale: 0.96 }}
                  onClick={resetModel}
                  className="px-5 py-3 bg-slate-800/80 hover:bg-slate-700/80 border border-white/8 rounded-xl font-bold text-sm flex items-center gap-2 text-slate-300 transition-all"
                >
                  <RefreshCw size={16} />
                  Resetear
                </motion.button>
              )}
              {currentStep === 'idle' && (
                <motion.button
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  whileHover={{ scale: 1.04, boxShadow: '0 0 30px rgba(37,99,235,0.4)' }}
                  whileTap={{ scale: 0.96 }}
                  onClick={startTraining}
                  className="px-7 py-3.5 bg-blue-600 hover:bg-blue-500 rounded-xl font-bold flex items-center gap-3 text-white shadow-lg shadow-blue-600/25 transition-all text-sm"
                >
                  <Play size={18} className="fill-current" />
                  Iniciar Entrenamiento
                </motion.button>
              )}
              {isTraining && (
                <div className="flex items-center gap-3 px-5 py-3 bg-blue-600/10 border border-blue-500/20 rounded-xl">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                  <span className="text-blue-300 font-bold text-sm uppercase tracking-widest">Procesando...</span>
                </div>
              )}
            </div>
          </div>

          {/* ── Top stat bar ── */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-8">
            <StatCard label="Dataset" value="Fashion-MNIST" icon={Database} color="text-blue-400" bg="bg-blue-500/10" sublabel="Zalando Research" />
            <StatCard label="Algoritmo" value="SVM Linear" icon={Cpu} color="text-cyan-400" bg="bg-cyan-500/10" sublabel='SVC(kernel="linear")' />
            <StatCard label="Accuracy" value={`${TRAINING_METRICS.accuracy}%`} icon={Target} color="text-emerald-400" bg="bg-emerald-500/10" sublabel="Test set 10,000 samples" />
            <StatCard label="Train Time" value={TRAINING_METRICS.trainTime} icon={Clock} color="text-amber-400" bg="bg-amber-500/10" sublabel="60,000 train samples" />
          </div>
        </header>

        {/* ── Main Grid ──────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">

          {/* ── Left column ───────────────────────────────────────────── */}
          <div className="xl:col-span-4 space-y-6">

            {/* Pipeline */}
            <section className="glass-card rounded-2xl p-6 shadow-xl">
              <h2 className="text-xs font-black uppercase tracking-[0.2em] text-slate-400 mb-5 flex items-center gap-2">
                <Layers size={14} className="text-blue-400" />
                Pipeline SVM
              </h2>
              <div className="space-y-2">
                {PIPELINE_STEPS.map((step, idx) => {
                  const isCompleted = completedStepIndex > idx || currentStep === 'ready';
                  const isActive = currentStep === step.id;

                  return (
                    <div key={step.id} className="relative">
                      <div
                        className={`flex items-center gap-3 p-3.5 rounded-xl transition-all duration-300 ${
                          isActive
                            ? `${step.bg} border ${step.border}`
                            : isCompleted
                            ? 'bg-emerald-500/5 border border-emerald-500/10'
                            : 'bg-transparent border border-transparent'
                        }`}
                      >
                        <div
                          className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 transition-all ${
                            isCompleted
                              ? 'bg-emerald-500/15 text-emerald-400'
                              : isActive
                              ? `${step.bg} ${step.color}`
                              : 'bg-slate-800/60 text-slate-600'
                          }`}
                        >
                          {isCompleted ? <CheckCircle2 size={16} /> : <step.icon size={16} />}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p
                            className={`text-xs font-bold truncate ${
                              isActive ? step.color : isCompleted ? 'text-slate-300' : 'text-slate-600'
                            }`}
                          >
                            {step.label}
                          </p>
                          <p className="text-[9px] text-slate-600 truncate font-mono">{step.sublabel}</p>
                          {isActive && (
                            <div className="mt-2 h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                              <motion.div
                                className={`h-full rounded-full ${
                                  step.id === 'loading'
                                    ? 'bg-blue-500'
                                    : step.id === 'preprocessing'
                                    ? 'bg-cyan-500'
                                    : step.id === 'extracting'
                                    ? 'bg-violet-500'
                                    : step.id === 'scaling'
                                    ? 'bg-amber-500'
                                    : step.id === 'training'
                                    ? 'bg-rose-500'
                                    : 'bg-emerald-500'
                                }`}
                                initial={{ width: 0 }}
                                animate={{ width: `${progress}%` }}
                                transition={{ ease: 'linear' }}
                              />
                            </div>
                          )}
                        </div>
                        {isCompleted && (
                          <ChevronRight size={12} className="text-emerald-500 shrink-0" />
                        )}
                      </div>
                      {idx < PIPELINE_STEPS.length - 1 && (
                        <div className="absolute left-[30px] top-[52px] w-px h-2 bg-slate-800" />
                      )}
                    </div>
                  );
                })}
              </div>
            </section>

            {/* SVM Hyperparameters */}
            <section className="glass-card rounded-2xl p-6 shadow-xl">
              <h2 className="text-xs font-black uppercase tracking-[0.2em] text-slate-400 mb-5 flex items-center gap-2">
                <Settings2 size={14} className="text-blue-400" />
                SVC Hyperparameters
              </h2>
              <div className="space-y-1 font-mono text-[11px]">
                {Object.entries(SVM_PARAMS).map(([key, val]) => (
                  <div key={key} className="flex justify-between items-center py-2.5 border-b border-white/4">
                    <span className="text-slate-500 font-mono">{key}</span>
                    <span className="text-blue-300 font-bold truncate ml-2 max-w-[55%] text-right">
                      {String(val)}
                    </span>
                  </div>
                ))}
              </div>
            </section>

            {/* Dataset Info */}
            <section className="glass-card rounded-2xl p-6 shadow-xl">
              <h2 className="text-xs font-black uppercase tracking-[0.2em] text-slate-400 mb-5 flex items-center gap-2">
                <BookOpen size={14} className="text-blue-400" />
                Dataset · Fashion-MNIST
              </h2>
              <div className="space-y-1 font-mono text-[11px]">
                {Object.entries(DATASET_STATS).map(([key, val]) => (
                  <div key={key} className="flex justify-between items-center py-2.5 border-b border-white/4">
                    <span className="text-slate-500 font-mono">{key}</span>
                    <span className="text-cyan-300 font-bold">{String(val)}</span>
                  </div>
                ))}
              </div>
            </section>
          </div>

          {/* ── Right column ──────────────────────────────────────────── */}
          <div className="xl:col-span-8 space-y-6">

            {/* ── Inference Panel ────────────────────────────────────── */}
            <section className="glass-card rounded-2xl p-6 shadow-xl">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h2 className="text-lg font-black text-white">Clasificador SVM</h2>
                  <p className="text-slate-500 text-xs mt-0.5">Carga una imagen para inferencia · Fashion-MNIST Labels</p>
                </div>
                <div
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest border ${
                    currentStep === 'ready'
                      ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                      : currentStep === 'idle'
                      ? 'bg-slate-700/40 border-slate-600/20 text-slate-500'
                      : 'bg-blue-500/10 border-blue-500/20 text-blue-400'
                  }`}
                >
                  <div
                    className={`w-1.5 h-1.5 rounded-full ${
                      currentStep === 'ready'
                        ? 'bg-emerald-400 animate-pulse'
                        : currentStep === 'idle'
                        ? 'bg-slate-600'
                        : 'bg-blue-400 animate-pulse'
                    }`}
                  />
                  {currentStep === 'ready' ? 'Modelo Listo' : currentStep === 'idle' ? 'Sin Entrenar' : 'Entrenando'}
                </div>
              </div>

              {/* Idle State */}
              {currentStep === 'idle' && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col items-center justify-center py-20 text-center"
                >
                  <div className="w-20 h-20 bg-slate-800/60 rounded-2xl flex items-center justify-center mb-5 border border-slate-700/40">
                    <Cpu size={36} className="text-slate-600" />
                  </div>
                  <p className="text-slate-600 font-bold uppercase tracking-[0.2em] text-xs mb-2">
                    Entrenamiento Requerido
                  </p>
                  <p className="text-slate-700 text-xs max-w-xs">
                    Haz clic en "Iniciar Entrenamiento" para ejecutar el pipeline SVM completo sobre Fashion-MNIST.
                  </p>
                </motion.div>
              )}

              {/* Training State */}
              {isTraining && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col items-center justify-center py-16 text-center"
                >
                  <div className="relative w-20 h-20 mb-6">
                    <div className="absolute inset-0 border-[3px] border-blue-500/20 border-t-blue-500 rounded-full animate-spin" />
                    <div className="absolute inset-3 border-[2px] border-cyan-500/20 border-b-cyan-500 rounded-full animate-spin" style={{ animationDirection: 'reverse', animationDuration: '0.8s' }} />
                    <Cpu className="absolute inset-0 m-auto text-blue-400" size={26} />
                  </div>
                  <h3 className="text-lg font-black text-white mb-1">Procesando vectores...</h3>
                  <p className="text-slate-500 text-sm max-w-xs">
                    Optimizando hiperparámetros con kernel lineal sobre {DATASET_STATS.trainSamples.toLocaleString()} muestras.
                  </p>
                  <div className="mt-5 w-64 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-blue-600 via-cyan-500 to-blue-400 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="text-blue-400/60 font-mono text-xs mt-2">{progress}%</p>
                </motion.div>
              )}

              {/* Ready State — Upload only */}
              {currentStep === 'ready' && (
                <div className="flex flex-col lg:flex-row gap-6">
                  {/* Drop zone */}
                  <div className="lg:w-72 shrink-0">
                    <input
                      type="file"
                      ref={fileInputRef}
                      className="hidden"
                      accept="image/*"
                      onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                    />
                    <motion.div
                      whileHover={{ scale: 1.01 }}
                      whileTap={{ scale: 0.99 }}
                      onClick={() => fileInputRef.current?.click()}
                      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                      onDragLeave={() => setIsDragging(false)}
                      onDrop={handleDrop}
                      className={`cursor-pointer h-52 rounded-2xl border-2 border-dashed flex flex-col items-center justify-center gap-4 transition-all duration-300 ${
                        isDragging
                          ? 'border-blue-400 bg-blue-500/10 scale-[1.02]'
                          : 'border-slate-700/60 hover:border-blue-500/50 hover:bg-blue-500/5 bg-slate-900/30'
                      }`}
                    >
                      <div
                        className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-all ${
                          isDragging ? 'bg-blue-500/20 text-blue-300 scale-110' : 'bg-slate-800/60 text-slate-500'
                        }`}
                      >
                        <Upload size={26} />
                      </div>
                      <div className="text-center px-4">
                        <p className="text-xs font-black text-slate-400 uppercase tracking-[0.2em] mb-1">
                          Cargar Imagen
                        </p>
                        <p className="text-[10px] text-slate-600 font-medium">
                          Arrastra o haz clic · JPG, PNG, WEBP
                        </p>
                      </div>
                    </motion.div>

                    {/* Label reference */}
                    <div className="mt-4 glass-card rounded-xl p-3">
                      <p className="text-[9px] text-slate-600 uppercase tracking-widest font-black mb-2">
                        Clases Fashion-MNIST
                      </p>
                      <div className="grid grid-cols-2 gap-1">
                        {Object.entries(CLASS_NAMES).map(([id, name]) => (
                          <div key={id} className="flex items-center gap-1.5">
                            <span className="text-sm">{CLASS_ICONS[Number(id)]}</span>
                            <span className="text-[9px] text-slate-500 truncate font-mono">{name}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Prediction result */}
                  <div className="flex-1 min-w-0">
                    <AnimatePresence mode="wait">
                      {!selectedImage ? (
                        <motion.div
                          key="empty"
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          className="h-full min-h-[300px] flex flex-col items-center justify-center text-center border-2 border-dashed border-slate-800 rounded-2xl bg-slate-900/10"
                        >
                          <Upload size={40} className="text-slate-700 mb-3" />
                          <p className="text-[10px] text-slate-700 font-black uppercase tracking-[0.2em]">
                            Esperando imagen...
                          </p>
                        </motion.div>
                      ) : (
                        <motion.div
                          key={selectedImage}
                          initial={{ opacity: 0, y: 12 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -12 }}
                          className="space-y-4"
                        >
                          {/* Image preview */}
                          <div className="relative w-full aspect-video max-h-56 rounded-2xl overflow-hidden bg-slate-900 border border-blue-500/15">
                            <ScanningOverlay active={isScanning} />
                            {!imgError ? (
                              <img
                                src={selectedImage}
                                alt="Input"
                                className="w-full h-full object-contain"
                                onError={() => setImgError(true)}
                              />
                            ) : (
                              <div className="flex flex-col items-center justify-center h-full gap-2 text-slate-600">
                                <AlertCircle size={32} />
                                <p className="text-xs font-bold">Error al cargar imagen</p>
                              </div>
                            )}
                            {isScanning && (
                              <div className="absolute inset-0 flex items-center justify-center z-40">
                                <div className="flex items-center gap-3 bg-blue-600/20 backdrop-blur-md border border-blue-500/30 px-5 py-2.5 rounded-full">
                                  <Activity size={14} className="text-blue-400 animate-pulse" />
                                  <span className="text-blue-300 text-[11px] font-black uppercase tracking-[0.2em]">
                                    Analizando vectores HOG...
                                  </span>
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Result */}
                          <AnimatePresence>
                            {predictionResult && (
                              <motion.div
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="glass-card rounded-2xl p-5 border border-blue-500/15"
                              >
                                <div className="flex items-start justify-between mb-4">
                                  <div>
                                    <p className="text-[9px] text-blue-400/70 uppercase tracking-[0.3em] font-black mb-1">
                                      SVM · Inference Result
                                    </p>
                                    <div className="flex items-center gap-3">
                                      <span className="text-3xl">{CLASS_ICONS[predictionResult.label]}</span>
                                      <h3 className="text-2xl font-black text-white tracking-tight">
                                        {CLASS_NAMES[predictionResult.label]}
                                      </h3>
                                    </div>
                                  </div>
                                  <div className="text-right">
                                    <p className="text-[9px] text-slate-600 uppercase tracking-widest font-black">Confianza</p>
                                    <p className="text-2xl font-black text-blue-400 font-mono">
                                      {predictionResult.confidence.toFixed(1)}%
                                    </p>
                                  </div>
                                </div>

                                {/* Confidence bar */}
                                <div className="mb-4">
                                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                    <motion.div
                                      initial={{ width: 0 }}
                                      animate={{ width: `${predictionResult.confidence}%` }}
                                      transition={{ duration: 0.8, ease: 'easeOut' }}
                                      className="h-full rounded-full bg-gradient-to-r from-blue-700 via-blue-500 to-cyan-400"
                                      style={{ boxShadow: '0 0 12px rgba(59,130,246,0.5)' }}
                                    />
                                  </div>
                                </div>

                                {/* Per-class probabilities mini bars */}
                                <p className="text-[9px] text-slate-600 uppercase tracking-widest font-black mb-2">
                                  Distribución por clase
                                </p>
                                <div className="space-y-1.5">
                                  {predictionResult.probabilities
                                    .map((prob, i) => ({ prob, i }))
                                    .sort((a, b) => b.prob - a.prob)
                                    .slice(0, 5)
                                    .map(({ prob, i }) => (
                                      <div key={i} className="flex items-center gap-2">
                                        <span className="text-xs w-5">{CLASS_ICONS[i]}</span>
                                        <span className="text-[9px] text-slate-500 font-mono w-20 truncate">{CLASS_NAMES[i]}</span>
                                        <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                          <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${prob}%` }}
                                            transition={{ delay: 0.1, duration: 0.5 }}
                                            className="h-full rounded-full"
                                            style={{
                                              backgroundColor: i === predictionResult.label ? '#3b82f6' : '#334155',
                                            }}
                                          />
                                        </div>
                                        <span className="text-[9px] font-mono text-slate-500 w-10 text-right">
                                          {prob.toFixed(1)}%
                                        </span>
                                      </div>
                                    ))}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </div>
              )}
            </section>

            {/* ── Charts ─────────────────────────────────────────────── */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

              {/* Convergence curve */}
              <section className="glass-card rounded-2xl p-5 shadow-xl">
                <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-400 mb-1 flex items-center gap-2">
                  <TrendingUp size={14} className="text-blue-400" />
                  Curva de Convergencia
                </h3>
                <p className="text-[9px] text-slate-600 mb-4 font-mono">Accuracy vs. Iteraciones (SVM dual)</p>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={CONVERGENCE_DATA} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                      <defs>
                        <linearGradient id="gradAcc" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#0f172a" vertical={false} />
                      <XAxis dataKey="iter" tick={{ fontSize: 9, fill: '#475569' }} tickLine={false} axisLine={false} />
                      <YAxis domain={[0.3, 1]} tick={{ fontSize: 9, fill: '#475569' }} tickLine={false} axisLine={false} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0d1829', border: '1px solid #1e3a5f', borderRadius: '10px', fontSize: '11px' }}
                        itemStyle={{ color: '#60a5fa' }}
                        labelStyle={{ color: '#64748b', fontSize: '10px' }}
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        formatter={(val: any) => [`${(Number(val) * 100).toFixed(1)}%`, 'Accuracy']}
                      />
                      <Area type="monotone" dataKey="accuracy" stroke="#3b82f6" strokeWidth={2} fill="url(#gradAcc)" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </section>

              {/* Radar metrics */}
              <section className="glass-card rounded-2xl p-5 shadow-xl">
                <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-400 mb-1 flex items-center gap-2">
                  <Shield size={14} className="text-cyan-400" />
                  Métricas de Evaluación
                </h3>
                <p className="text-[9px] text-slate-600 mb-4 font-mono">Performance global del clasificador SVM</p>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={RADAR_DATA} margin={{ top: 0, right: 20, bottom: 0, left: 20 }}>
                      <PolarGrid stroke="#1e293b" />
                      <PolarAngleAxis dataKey="metric" tick={{ fontSize: 9, fill: '#475569' }} />
                      <Radar dataKey="value" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.15} strokeWidth={2} dot={{ fill: '#06b6d4', r: 3 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0d1829', border: '1px solid #1e3a5f', borderRadius: '10px', fontSize: '11px' }}
                        itemStyle={{ color: '#67e8f9' }}
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        formatter={(val: any) => [`${Number(val).toFixed(1)}%`, '']}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </section>
            </div>

            {/* Per-class accuracy bar chart */}
            <section className="glass-card rounded-2xl p-5 shadow-xl">
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-400 flex items-center gap-2">
                  <BarChart3 size={14} className="text-violet-400" />
                  Accuracy por Clase
                </h3>
                <span className="text-[9px] font-mono text-slate-600">Fashion-MNIST Test Set</span>
              </div>
              <p className="text-[9px] text-slate-600 mb-4 font-mono">Precision individual de cada categoría · SVM Linear</p>
              <div className="h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={CLASS_ACCURACY_DATA} margin={{ top: 4, right: 4, bottom: 20, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#0f172a" vertical={false} />
                    <XAxis
                      dataKey="name"
                      tick={{ fontSize: 9, fill: '#475569' }}
                      tickLine={false}
                      axisLine={false}
                      angle={-30}
                      textAnchor="end"
                      interval={0}
                    />
                    <YAxis
                      domain={[60, 100]}
                      tick={{ fontSize: 9, fill: '#475569' }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v) => `${v}%`}
                    />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#0d1829', border: '1px solid #1e3a5f', borderRadius: '10px', fontSize: '11px' }}
                        itemStyle={{ color: '#a78bfa' }}
                        // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        formatter={(val: any) => [`${Number(val)}%`, 'Accuracy']}
                      />
                    <Bar dataKey="acc" radius={[4, 4, 0, 0]}>
                      {CLASS_ACCURACY_DATA.map((_, index) => (
                        <Cell key={index} fill={BAR_COLORS[index % BAR_COLORS.length]} fillOpacity={0.85} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>

            {/* Bottom info row */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="glass-card rounded-2xl p-4 flex items-center gap-3">
                <div className="w-9 h-9 bg-blue-500/10 rounded-xl flex items-center justify-center">
                  <Database size={16} className="text-blue-400" />
                </div>
                <div>
                  <p className="text-[9px] text-slate-600 uppercase tracking-widest font-black">Train Split</p>
                  <p className="text-sm font-black text-white">60,000 muestras</p>
                  <p className="text-[9px] text-slate-600 font-mono">X_train.shape = (60000, 784)</p>
                </div>
              </div>
              <div className="glass-card rounded-2xl p-4 flex items-center gap-3">
                <div className="w-9 h-9 bg-cyan-500/10 rounded-xl flex items-center justify-center">
                  <Zap size={16} className="text-cyan-400" />
                </div>
                <div>
                  <p className="text-[9px] text-slate-600 uppercase tracking-widest font-black">Test Split</p>
                  <p className="text-sm font-black text-white">10,000 muestras</p>
                  <p className="text-[9px] text-slate-600 font-mono">X_test.shape = (10000, 784)</p>
                </div>
              </div>
              <div className="glass-card rounded-2xl p-4 flex items-center gap-3">
                <div className="w-9 h-9 bg-violet-500/10 rounded-xl flex items-center justify-center">
                  <Activity size={16} className="text-violet-400" />
                </div>
                <div>
                  <p className="text-[9px] text-slate-600 uppercase tracking-widest font-black">CV Score</p>
                  <p className="text-sm font-black text-white">{TRAINING_METRICS.cvMean}% ± {TRAINING_METRICS.cvStd}%</p>
                  <p className="text-[9px] text-slate-600 font-mono">StratifiedKFold(n_splits=5)</p>
                </div>
              </div>
            </div>

          </div>
        </div>

        {/* Footer */}
        <footer className="mt-10 pt-6 border-t border-white/4 flex flex-col md:flex-row items-center justify-between gap-3">
          <p className="text-slate-700 text-[10px] font-mono">
            SVM Fashion-MNIST Classifier · scikit-learn SVC · Kernel Linear · Fashion-MNIST Dataset by Zalando Research
          </p>
          <div className="flex items-center gap-4">
            <span className="text-[10px] text-slate-700 font-mono">kernel="linear"</span>
            <span className="text-[10px] text-slate-700 font-mono">C=1.0</span>
            <span className="text-[10px] text-slate-700 font-mono">acc=89.4%</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
