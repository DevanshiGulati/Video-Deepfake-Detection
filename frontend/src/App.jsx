import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { AlertCircle, ArrowLeft, ArrowRight, CheckCircle2, Shield, Target, Upload, Video, XCircle, Zap } from 'lucide-react';
import { predictVideo } from './api';

const MAX_SIZE = 100 * 1024 * 1024;
const ACCEPTED = { 'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm'] };

function App() {
  const [page, setPage] = useState('landing');
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [result, setResult] = useState(null);
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const onDrop = useCallback((files, rejected) => {
    setError('');
    setResult(null);
    const next = files[0];
    if (rejected.length || !next) {
      const reason = rejected[0]?.errors?.[0]?.code;
      setError(reason === 'file-too-large' ? 'Video must be 100 MB or smaller.' : 'Please select a supported video file.');
      return;
    }
    setFile(next);
    setPreview(URL.createObjectURL(next));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    maxSize: MAX_SIZE,
    multiple: false,
  });

  const clearFile = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview('');
    setResult(null);
    setError('');
  };

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    setProgress(0);
    try {
      setResult(await predictVideo(file, setProgress));
    } catch (err) {
      setError(err?.response?.data?.detail || err?.message || 'Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const start = () => { setPage('analyze'); setError(''); };
  const home = () => { clearFile(); setPage('landing'); };

  if (page === 'landing') {
    return (
      <main className="landing-page">
        <div className="orb orb-one" /><div className="orb orb-two" />
        <section className="hero">
          <div className="hero-visual"><div className="scan-ring"><Shield size={92} /></div><div className="scan-line" /></div>
          <div className="hero-content">
            <div className="badge"><Shield size={15} /> AI-POWERED DETECTION</div>
            <h1>Welcome to <span>ByteBuster</span></h1>
            <p>Your intelligent companion for detecting deepfake content. Analyze videos with a trained EfficientNet-B3 + BiLSTM + Attention model.</p>
            <div className="feature-grid">
              <Feature icon={<Zap />} title="Fast Analysis" text="Automated frame extraction" />
              <Feature icon={<Target />} title="Temporal AI" text="Spatial + sequence analysis" />
              <Feature icon={<Shield />} title="Private" text="Uploads stay on your API" />
            </div>
            <button className="primary" onClick={start}>Get Started <ArrowRight size={19} /></button>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="analyze-page">
      <button className="back" onClick={home}><ArrowLeft size={18} /> Back to Home</button>
      <section className="analyze-shell">
        <header><div className="badge"><Shield size={15} /> AI-POWERED DETECTION</div><h1>Deepfake <span>Detector</span></h1><p>Upload a video to analyze for manipulation.</p></header>
        {!result ? (
          <div className="analyze-grid">
            <aside className="info-card"><h2>How It Works</h2><Step n="1" title="Upload your video" text="MP4, MOV, AVI, MKV or WEBM up to 100 MB." /><Step n="2" title="AI analyzes frames" text="Faces are detected, cropped and sampled across the video." /><Step n="3" title="Get a prediction" text="The temporal model returns real/fake probabilities and evidence frames." /></aside>
            <section className="upload-card">
              {!file ? (
                <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}><input {...getInputProps()} /><Upload size={48} /><h2>Drag & drop your video</h2><p>or click to browse</p><small>MP4 · MOV · AVI · MKV · WEBM · max 100 MB</small></div>
              ) : (
                <div className="selected">
                  <div className="file-row"><span><Video size={20} /> {file.name}</span><button onClick={clearFile}>Remove</button></div>
                  <video src={preview} controls />
                  {!loading && <button className="primary" onClick={analyze}>Analyze Video <Target size={19} /></button>}
                  {loading && <div className="progress-wrap"><div className="progress-label"><span>Analyzing video…</span><b>{progress}%</b></div><div className="progress"><i style={{ width: `${Math.max(progress, 3)}%` }} /></div><small>Frame extraction and model inference can take a few minutes on CPU.</small></div>}
                </div>
              )}
              {error && <div className="error"><AlertCircle size={20} /> <span>{error}</span></div>}
            </section>
          </div>
        ) : <Result result={result} onReset={clearFile} />}
      </section>
    </main>
  );
}

function Feature({ icon, title, text }) { return <div className="feature"><div>{icon}</div><section><b>{title}</b><small>{text}</small></section></div>; }
function Step({ n, title, text }) { return <div className="step"><strong>{n}</strong><section><b>{title}</b><p>{text}</p></section></div>; }
function Result({ result, onReset }) {
  const fake = result.prediction === 'fake';
  return <section className="result-card">
    <div className={`result-icon ${fake ? 'fake' : 'real'}`}>{fake ? <XCircle size={68} /> : <CheckCircle2 size={68} />}</div>
    <div className="result-copy"><div className="badge">ANALYSIS COMPLETE</div><h2>{fake ? 'Deepfake Detected' : 'Appears Authentic'}</h2><p>{fake ? 'The model found patterns associated with manipulated video content.' : 'The model found stronger evidence for authentic video content.'}</p></div>
    <div className="confidence"><div><span>Model confidence</span><b>{(result.confidence * 100).toFixed(1)}%</b></div><div className="bar"><i style={{ width: `${result.confidence * 100}%` }} /></div></div>
    <div className="probabilities"><div><span>Real</span><b>{(result.realConfidence * 100).toFixed(2)}%</b></div><div><span>Fake</span><b>{(result.fakeConfidence * 100).toFixed(2)}%</b></div></div>
    {result.frameUrls.length > 0 && <div className="frames"><h3>Evidence Frames</h3><div>{result.frameUrls.map((url, i) => <img key={url} src={url} alt={`Analyzed frame ${i + 1}`} />)}</div></div>}
    <button className="secondary" onClick={onReset}>Analyze Another Video</button>
  </section>;
}

export default App;
