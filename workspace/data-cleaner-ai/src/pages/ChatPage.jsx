import { useState, useRef, useEffect } from 'react'
import { useDataStore } from '../store/useDataStore'

function answerQuestion(question, rows) {
  const q = question.toLowerCase()
  if (!rows?.length) return 'Chargez des données pour obtenir des réponses contextuelles.'
  if (q.includes('combien') || q.includes('how many')) return `Il y a ${rows.length} enregistrements.`
  if (q.includes('qualité') || q.includes('quality')) return `Score qualité estimé: ${(useDataStore.getState().qualityScore ?? 0)}%.`
  return 'Je recherche des patterns... Essayez: "Combien d’enregistrements ?" ou "Score qualité ?"'
}

export default function ChatPage() {
  const rows = useDataStore(s=>s.cleanedData)
  const [messages, setMessages] = useState([{ role: 'ai', text: 'Bonjour ! Posez-moi des questions sur vos données.' }])
  const [input, setInput] = useState('')
  const endRef = useRef(null)
  useEffect(()=>{ endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  function send() {
    if (!input.trim()) return
    const userMsg = { role: 'user', text: input.trim() }
    setMessages(m=>[...m, userMsg, { role: 'typing' }])
    setInput('')
    setTimeout(()=>{
      const reply = answerQuestion(userMsg.text, rows)
      setMessages(m=>[...m.filter(x=>x.role!=='typing'), { role: 'ai', text: reply }])
    }, 600)
  }

  return (
    <div className="flex flex-col h-[70vh] border border-base-300 rounded-xl overflow-hidden bg-base-200/40">
      <div className="flex-1 p-4 space-y-3 overflow-y-auto">
        {messages.map((m, i)=> (
          <div key={i} className={`chat ${m.role==='user'?'chat-end':'chat-start'}`}>
            {m.role==='typing' ? (
              <div className="chat-bubble"><span className="loading loading-dots loading-sm" /></div>
            ) : (
              <div className="chat-bubble">{m.text}</div>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <div className="p-3 bg-base-200 border-t border-base-300 flex gap-2">
        <input className="input input-bordered flex-1" placeholder="Posez une question..." value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>e.key==='Enter' && send()} />
        <button className="btn btn-secondary" onClick={send}>Envoyer</button>
      </div>
    </div>
  )
}

