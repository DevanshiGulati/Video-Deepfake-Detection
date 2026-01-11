export default function Panel({ title, children }) {
  return (
    <div style={{
      maxWidth: 960,
      margin: '24px auto',
      padding: 24,
      borderRadius: 14,
      background: '#0b0b0b',
      color: '#e5e7eb',
      boxShadow: '0 0 28px rgba(255,255,255,0.20), 0 0 80px rgba(255,255,255,0.08), 0 4px 24px rgba(0,0,0,0.6)',
      border: '1px solid rgba(255,255,255,0.12)',
      transition: 'box-shadow 200ms ease, transform 200ms ease',
    }}>
      {title && (
        <div style={{
          fontSize: 20,
          fontWeight: 700,
          marginBottom: 12,
          color: '#f8fafc'
        }}>
          {title}
        </div>
      )}
      <div>
        {children}
      </div>
    </div>
  )
}


