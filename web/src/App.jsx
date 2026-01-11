import { useState } from 'react'
import axios from 'axios'
import Panel from './components/Panel'

export default function App() {
  const [file, setFile] = useState()
  const [pred, setPred] = useState(null)
  const [probs, setProbs] = useState(null)
  const [predFrames, setPredFrames] = useState([])
  const [predictLoading, setPredictLoading] = useState(false)
  const [predictError, setPredictError] = useState('')
  const api = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

  const fileName = file?.name || 'No file selected'

  const btnBase = {
    appearance: 'none',
    border: 'none',
    outline: 'none',
    cursor: 'pointer',
    padding: '10px 16px',
    borderRadius: 10,
    fontWeight: 600,
    color: '#0b0b0b',
    background: 'linear-gradient(135deg, #ffffff, #e5e7eb)',
    boxShadow: '0 2px 8px rgba(255,255,255,0.2), 0 4px 24px rgba(255,255,255,0.12)',
    transform: 'translateY(0px)',
    transition: 'transform 120ms ease, box-shadow 120ms ease, opacity 120ms ease',
  }

  const btnHover = {
    boxShadow: '0 6px 18px rgba(255,255,255,0.28), 0 12px 36px rgba(255,255,255,0.16)',
    transform: 'translateY(-1px)'
  }

  const predictBtnStyles = predictLoading
    ? { ...btnBase, opacity: 0.85, boxShadow: '0 0 18px rgba(255,255,255,0.22), 0 0 48px rgba(255,255,255,0.12)' }
    : btnBase

  const onPredict = async () => {
    if (!file) return
    const form = new FormData()
    form.append('file', file)
    setPredictLoading(true)
    setPredictError('')
    try {
      const { data } = await axios.post(`${api}/api/predict-video`, form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setPred(data.pred)
      setProbs(data.probs)
      setPredFrames(data.frameUrls || [])
    } catch (e) {
      setPredictError(e?.response?.data?.detail || 'Prediction failed')
    } finally {
      setPredictLoading(false)
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, sans-serif', background: '#000', minHeight: '100vh', color: '#e5e7eb' }}>
      <Panel title="Deepfake Pipeline: Upload Video Predict">
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        <input id="video-input" type="file" accept="video/mp4" onChange={e => setFile(e.target.files?.[0])} style={{ display: 'none' }} />
        <label htmlFor="video-input"
               style={btnBase}
               onMouseEnter={e => Object.assign(e.currentTarget.style, btnHover)}
               onMouseLeave={e => Object.assign(e.currentTarget.style, { boxShadow: btnBase.boxShadow, transform: btnBase.transform })}>
          Choose Video
        </label>
        <div style={{ opacity: 0.85 }}>{fileName}</div>

        <button
          onClick={onPredict}
          disabled={!file || predictLoading}
          style={{
            ...predictBtnStyles,
            background: 'linear-gradient(135deg, #ffffff, #e5e7eb)',
            color: '#0b0b0b'
          }}
          onMouseEnter={e => !predictLoading && Object.assign(e.currentTarget.style, btnHover)}
          onMouseLeave={e => !predictLoading && Object.assign(e.currentTarget.style, { boxShadow: btnBase.boxShadow, transform: btnBase.transform })}
        >
          {predictLoading ? 'Predicting…' : 'Predict'}
        </button>
      </div>
      {predictError && <div style={{ color: '#fda4af', marginTop: 8 }}>{predictError}</div>}
      {(pred !== null || probs) && (
        <div style={{ marginTop: 16, padding: 12, border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10, background: '#0f0f10' }}>
          <div style={{ fontWeight: 700, marginBottom: 8, color: '#f8fafc' }}>Prediction</div>
          <div>Label: <span style={{ fontWeight: 600 }}>{pred === 1 ? 'fake' : 'real'}</span></div>
          {Array.isArray(probs) && probs.length === 2 && (
            <div style={{ marginTop: 4 }}>Probs: real={probs[0]?.toFixed?.(3)} fake={probs[1]?.toFixed?.(3)}</div>
          )}
          {!!predFrames.length && (
            <div style={{ marginTop: 12 }}>
              <div style={{ marginBottom: 6 }}>Sampled frames:</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, 120px)', gap: 8 }}>
                {predFrames.map(u => (
                  <img key={u} src={`${api}${u}`} style={{ width: 120, height: 120, objectFit: 'cover', borderRadius: 6, boxShadow: '0 0 10px rgba(255,255,255,0.12)' }} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      </Panel>
    </div>
  )
}


