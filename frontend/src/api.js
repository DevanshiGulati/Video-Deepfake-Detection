import axios from 'axios';

const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');

export const apiBaseUrl = API_BASE_URL;

export async function predictVideo(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post(`${API_BASE_URL}/api/predict-video`, formData, {
    onUploadProgress: (event) => {
      if (event.total) onProgress?.(Math.round((event.loaded / event.total) * 100));
    },
    timeout: 10 * 60 * 1000,
  });

  const data = response.data;
  const pred = Number(data.pred);
  const probs = Array.isArray(data.probs) ? data.probs : [];

  if (![0, 1].includes(pred) || probs.length < 2) {
    throw new Error('Backend returned an invalid prediction response.');
  }

  return {
    prediction: pred === 1 ? 'fake' : 'real',
    confidence: Number(probs[pred]),
    realConfidence: Number(probs[0]),
    fakeConfidence: Number(probs[1]),
    frameUrls: (data.frameUrls || []).map((url) => `${API_BASE_URL}${url}`),
    message: data.message || 'Video analysis complete',
  };
}
