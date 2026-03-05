import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import Login from './Login';
import { loadUser, exchangeCodeForToken, getUserInfo, saveUser, logoutUser } from './auth';
const API_URL = import.meta.env.VITE_API_URL || '';

// Helper for resilient fetching
const fetchWithRetry = async (url, options = {}, retries = 3, backoff = 1000) => {
  try {
    const response = await fetch(url, options);
    if (!response.ok && retries > 0) {
      const errorData = await response.json().catch(() => ({}));
      console.warn(`Fetch failed (${response.status}), retrying in ${backoff}ms...`, errorData);
      await new Promise(resolve => setTimeout(resolve, backoff));
      return fetchWithRetry(url, options, retries - 1, backoff * 2);
    }
    return response;
  } catch (error) {
    if (retries > 0) {
      console.warn(`Fetch error (${error.message}), retrying in ${backoff}ms...`);
      await new Promise(resolve => setTimeout(resolve, backoff));
      return fetchWithRetry(url, options, retries - 1, backoff * 2);
    }
    throw error;
  }
};

// Helper function to render markdown-style text
const renderFormattedText = (text, isDark = false) => {
  const lines = text.split('\n');
  const elements = [];
  let inTable = false;
  let tableRows = [];

  lines.forEach((line, idx) => {
    if (line.trim().startsWith('|')) {
      inTable = true;
      tableRows.push(line);
    } else {
      if (inTable && tableRows.length > 0) {
        elements.push(renderTable(tableRows, elements.length, isDark));
        tableRows = [];
        inTable = false;
      }

      if (line.trim()) {
        elements.push(renderLine(line, idx, isDark));
      } else if (idx > 0 && lines[idx - 1].trim()) {
        // Only add a small break for empty lines if the previous line wasn't empty
        // This prevents "huge line space"
        elements.push(<div key={idx} style={{ height: '8px' }} />);
      }
    }
  });

  if (inTable && tableRows.length > 0) {
    elements.push(renderTable(tableRows, elements.length, isDark));
  }

  return elements;
};

const renderTable = (rows, key, isDark) => {
  const dataRows = rows.filter(row => !row.match(/^\|[\s-:|]+\|$/));

  if (dataRows.length === 0) return null;

  const parseRow = (row) => {
    return row.split('|')
      .filter(cell => cell.trim())
      .map(cell => cell.trim());
  };

  const headerCells = parseRow(dataRows[0]);
  const bodyRows = dataRows.slice(1).map(parseRow);

  return (
    <div key={key} style={{ overflowX: 'auto', margin: '16px 0' }}>
      <table style={{
        width: '100%',
        borderCollapse: 'collapse',
        border: `1px solid ${isDark ? '#4b5563' : '#e5e7eb'}`,
        borderRadius: '8px',
        overflow: 'hidden'
      }}>
        <thead style={{ backgroundColor: isDark ? '#1f2937' : '#f9fafb' }}>
          <tr>
            {headerCells.map((cell, i) => (
              <th key={i} style={{
                padding: '12px',
                textAlign: 'left',
                fontWeight: '600',
                borderBottom: `2px solid ${isDark ? '#4b5563' : '#e5e7eb'}`,
                color: isDark ? '#f3f4f6' : '#374151'
              }}>{cell}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bodyRows.map((row, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${isDark ? '#374151' : '#f3f4f6'}` }}>
              {row.map((cell, j) => (
                <td key={j} style={{
                  padding: '12px',
                  color: isDark ? '#d1d5db' : '#6b7280'
                }}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const renderLine = (line, key, isDark) => {
  const trimmed = line.trim();
  const textColor = isDark ? '#e5e7eb' : '#374151';
  const headingColor = isDark ? '#f3f4f6' : '#111827';

  if (trimmed.startsWith('•') || trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
    const content = trimmed.startsWith('•')
      ? trimmed.substring(1).trim()
      : trimmed.substring(2).trim();

    return (
      <div key={key} style={{ display: 'flex', marginBottom: '6px', alignItems: 'flex-start', color: textColor }}>
        <span style={{ marginRight: '8px', fontWeight: '900', flexShrink: 0, fontSize: '18px', color: headingColor }}>•</span>
        <span style={{ flex: 1 }}>{formatInlineStyles(content)}</span>
      </div>
    );
  }

  if (/^\d+\./.test(trimmed)) {
    const match = trimmed.match(/^(\d+\.)\s*(.+)$/);
    if (match) {
      return (
        <div key={key} style={{ display: 'flex', marginBottom: '8px', marginLeft: '16px', alignItems: 'flex-start', color: textColor }}>
          <span style={{ marginRight: '8px', fontWeight: '500', flexShrink: 0 }}>{match[1]}</span>
          <span style={{ flex: 1 }}>{formatInlineStyles(match[2])}</span>
        </div>
      );
    }
  }

  if (trimmed.startsWith('###')) {
    return <h4 key={key} style={{ fontSize: '17px', fontWeight: '600', margin: '12px 0 8px', color: headingColor }}>{trimmed.substring(3).trim()}</h4>;
  }
  if (trimmed.startsWith('##')) {
    return <h3 key={key} style={{ fontSize: '18px', fontWeight: '600', margin: '16px 0 8px', color: headingColor }}>{trimmed.substring(2).trim()}</h3>;
  }
  if (trimmed.startsWith('#')) {
    return <h2 key={key} style={{ fontSize: '21px', fontWeight: '700', margin: '18px 0 12px', color: headingColor }}>{trimmed.substring(1).trim()}</h2>;
  }

  return <p key={key} style={{ marginBottom: '10px', lineHeight: '1.5', color: textColor }}>{formatInlineStyles(trimmed)}</p>;
};

const formatInlineStyles = (text) => {
  const parts = [];
  let lastIndex = 0;
  const boldRegex = /(\*\*|__)(.*?)\1/g;
  let match;

  while ((match = boldRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }
    parts.push(<strong key={match.index}>{match[2]}</strong>);
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  return parts.length > 0 ? parts : text;
};

function App() {
  const [user, setUser] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const isLoadingRef = useRef(false); // Ref to avoid stale closure in watchdog
  const [darkMode, setDarkMode] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [uploadedDocs, setUploadedDocs] = useState([]);
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(`session-${Date.now()}`);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [selectedModel, setSelectedModel] = useState('deepseek-v3');
  const [availableModels, setAvailableModels] = useState([]);
  const [systemStats, setSystemStats] = useState(null);
  const [showSourcesPanel, setShowSourcesPanel] = useState(false);
  const [currentSources, setCurrentSources] = useState([]);
  const [useQueryExpansion, setUseQueryExpansion] = useState(true);
  const [searchMode, setSearchMode] = useState('thinking'); // 'fast' or 'thinking'
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const forceClearLoading = async () => {
    console.log("🛠️ Emergency Reset: Clearing isLoading and Refreshing State");
    setIsLoading(false);
    setConnectionStatus('connected');

    // Auto-refresh core app state
    try {
      await checkConnection();
      await loadSessions();
      if (currentSessionId) {
        await loadSession(currentSessionId);
      }
    } catch (e) {
      console.error("Reset refresh failed:", e);
    }

    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.disabled = false;
        inputRef.current.focus();
      }
    }, 200);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const initAuth = async () => {
      // Check for callback
      const urlParams = new URLSearchParams(window.location.search);
      const code = urlParams.get('code');

      if (code) {
        try {
          // Clear query params from URL
          window.history.replaceState({}, document.title, window.location.pathname);

          setIsLoading(true);
          const tokenData = await exchangeCodeForToken(code);
          const userData = await getUserInfo(tokenData.access_token);

          const userWithTokens = { ...userData, ...tokenData };
          saveUser(userWithTokens);
          setUser(userWithTokens);
        } catch (error) {
          console.error('Auth error:', error);
        } finally {
          setIsLoading(false);
        }
      } else {
        const storedUser = loadUser();
        if (storedUser) {
          setUser(storedUser);
        }
      }
    };

    initAuth();
  }, []);

  useEffect(() => {
    if (user) {
      initializeApp();
    }
  }, [user]);

  const initializeApp = async () => {
    await checkConnection();
    await fetchAvailableModels();
    await fetchSystemStats();
    await fetchSources();
    await loadSessions();

    // Load or create current session
    await loadSession(currentSessionId);
  };

  const checkConnection = async () => {
    try {
      const response = await fetch(`${API_URL}/api/health`);
      const data = await response.json();

      if (data.status === 'healthy') {
        setConnectionStatus('connected');
        console.log('✅ Connected to backend:', data);
      } else {
        setConnectionStatus('error');
      }
    } catch (error) {
      console.error('❌ Backend connection failed:', error);
      setConnectionStatus('disconnected');
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await fetch(`${API_URL}/api/models`);
      const data = await response.json();

      if (data.available_models) {
        setAvailableModels(data.available_models);
        if (data.current_model) {
          setSelectedModel(data.current_model);
        }
      }
    } catch (error) {
      console.error('Error fetching models:', error);
      // UPDATED: Changed to DeepSeek-V3 and GPT-OSS-20B
      setAvailableModels([
        { key: 'deepseek-v3', name: 'DeepSeek-V3', type: 'deepseek', is_current: true },
        { key: 'gpt-oss-20b', name: 'GPT-OSS-20B', type: 'openai', is_current: false }
      ]);
    }
  };

  const fetchSystemStats = async () => {
    try {
      const response = await fetch(`${API_URL}/api/stats`);
      const data = await response.json();
      setSystemStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const fetchSources = async () => {
    try {
      const response = await fetch(`${API_URL}/api/sources`);
      const data = await response.json();

      if (data.sources && data.sources.length > 0) {
        setUploadedDocs(data.sources);
      }
    } catch (error) {
      console.error('Error fetching sources:', error);
    }
  };

  const loadSessions = async () => {
    try {
      const response = await fetch(`${API_URL}/api/sessions`);
      const data = await response.json();

      if (data.sessions) {
        setSessions(data.sessions);
      }
    } catch (error) {
      console.error('Error loading sessions:', error);
    }
  };

  const loadSession = async (sessionId) => {
    try {
      const response = await fetch(`${API_URL}/api/session/${sessionId}`);
      const data = await response.json();

      if (data.messages) {
        const uiMessages = data.messages.map(msg => ({
          id: msg.id,
          text: msg.content,
          isUser: msg.role === 'user',
          sources: msg.sources || [],
          timestamp: msg.timestamp
        }));

        setMessages(uiMessages);
        setCurrentSessionId(sessionId);
      }
    } catch (error) {
      console.error('Error loading session:', error);
      if (messages.length === 0) {
        setMessages([
          {
            id: 1,
            text: "Welcome to the Research Assistant! 🌊 I'm here to help you with aquatic research, conservation strategies, hydrological analysis, and ecosystem management. Upload your research documents and ask me anything!",
            isUser: false
          }
        ]);
      }
    }
  };

  const switchModel = async (modelKey) => {
    try {
      const response = await fetch(`${API_URL}/api/switch-model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: modelKey,
          previous_model: selectedModel
        })
      });

      const data = await response.json();

      if (response.ok) {
        setSelectedModel(modelKey);
        await fetchAvailableModels();
        await fetchSystemStats();

        const systemMessage = {
          id: Date.now(),
          text: `✅ Switched to ${data.model_info.name} model`,
          isUser: false
        };
        setMessages(prev => [...prev, systemMessage]);
      }
    } catch (error) {
      console.error('Error switching model:', error);
    }
  };

  const handleSendMessage = async () => {
    console.log("💬 Sending message...", { inputText, isLoading, connectionStatus });
    if (!inputText.trim()) {
      console.warn("⚠️ Empty input, aborting send.");
      return;
    }
    if (isLoading) {
      console.warn("⚠️ Already loading, aborting send.");
      return;
    }

    const userMessageCount = messages.filter(m => m.isUser).length;
    if (userMessageCount >= 15) {
      console.warn("⚠️ Message limit reached (15 questions).");
      return;
    }

    const currentMode = searchMode; // Capture mode at send time
    const userMessage = {
      id: Date.now(),
      text: inputText,
      isUser: true,
      mode: currentMode
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputText;
    setInputText('');
    isLoadingRef.current = true;
    setIsLoading(true);

    // Create a placeholder for the AI message
    const aiMessageId = Date.now() + 1;
    const initialAiMessage = {
      id: aiMessageId,
      text: '',
      isUser: false,
      sources: [],
      isThinking: true,
      mode: currentMode,
      thinkingText: currentMode === 'fast' ? "⚡ Searching for immediate answers..." : "🧠 Analyzing wetland data and research records..."
    };

    setMessages(prev => [...prev, initialAiMessage]);

    // Safety watchdog: Force unlock after 45 seconds if everything else fails
    const watchdog = setTimeout(() => {
      if (isLoadingRef.current) {
        console.warn("🛡️ Watchdog: Chat taking too long, forcing automatic refresh...");
        forceClearLoading();
      }
    }, 45000);

    try {
      const response = await fetchWithRetry(`${API_URL}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentInput,
          session_id: currentSessionId,
          use_query_expansion: useQueryExpansion,
          mode: currentMode
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to connect to backend');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullText = '';
      let aiSources = [];
      // SSE buffer: network packets may arrive mid-line; accumulate until \n\n
      let sseBuffer = '';
      let thinkingStepCount = 0;

      let isDone = false;
      while (!isDone) {
        const { done, value } = await reader.read();
        if (done) {
          console.log("➡️ Stream reader finished (done: true)");
          break;
        }

        // Append to buffer and split on complete lines only
        sseBuffer += decoder.decode(value, { stream: true });
        const lines = sseBuffer.split('\n');
        // Last element may be incomplete — keep it in the buffer
        sseBuffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine.startsWith('data: ')) continue;

          const dataStr = trimmedLine.slice(6).trim();
          if (dataStr === '[DONE]') {
            console.log("➡️ Received [DONE] signal");
            isDone = true;
            try { await reader.cancel(); } catch (e) { }
            break;
          }
          if (!dataStr) continue;

          try {
            const data = JSON.parse(dataStr);

            if (data.type === 'thought') {
              // Show live thinking progress in the spinner
              thinkingStepCount += 1;
              const label = currentMode === 'fast'
                ? `⚡ Searching... (step ${thinkingStepCount})`
                : `🧠 Thinking deeply... (step ${thinkingStepCount})`;
              setMessages(prev => prev.map(msg =>
                msg.id === aiMessageId
                  ? { ...msg, isThinking: true, thinkingText: label }
                  : msg
              ));
            } else if (data.type === 'observation') {
              // Tool result received — show intermediate status
              setMessages(prev => prev.map(msg =>
                msg.id === aiMessageId
                  ? { ...msg, isThinking: true, thinkingText: currentMode === 'fast' ? '⚡ Processing results...' : '🔍 Reviewing research sources...' }
                  : msg
              ));
            } else if (data.type === 'answer') {
              fullText += data.content;
              setMessages(prev => prev.map(msg =>
                msg.id === aiMessageId
                  ? { ...msg, text: fullText, isThinking: false }
                  : msg
              ));
            } else if (data.type === 'sources') {
              aiSources = data.content;
              setMessages(prev => prev.map(msg =>
                msg.id === aiMessageId
                  ? { ...msg, sources: aiSources }
                  : msg
              ));
            } else if (data.type === 'error') {
              // Server-side error sent through the stream
              const errMsg = `❌ Server error: ${data.content}`;
              setMessages(prev => prev.map(msg =>
                msg.id === aiMessageId
                  ? { ...msg, text: errMsg, isThinking: false }
                  : msg
              ));
              isDone = true;
              try { await reader.cancel(); } catch (e) { }
              break;
            } else if (data.type === 'session_id') {
              if (!currentSessionId) setCurrentSessionId(data.content);
            }
          } catch (e) {
            // Warn but don't crash — incomplete JSON from a split chunk
            console.warn('SSE parse warning:', e.message, '| raw:', dataStr.substring(0, 80));
          }
        }
      }

      // If the stream ended with no answer produced (e.g. thinking mode hit max iterations)
      if (!fullText) {
        setMessages(prev => prev.map(msg =>
          msg.id === aiMessageId
            ? {
              ...msg,
              text: '⚠️ The research process completed but could not produce a final answer. This can happen with very complex questions in Thinking Mode. Try rephrasing your question or switch to ⚡ Fast mode.',
              isThinking: false
            }
            : msg
        ));
      }

      await loadSessions();
    } catch (error) {
      console.error('CRITICAL: Chat stream error:', error);
      setMessages(prev => prev.map(msg =>
        msg.id === aiMessageId
          ? { ...msg, text: `❌ Network Error: ${error.message}. Please check if the server is running or try a simpler query.`, isThinking: false }
          : msg
      ));
      // Reset connection status if it's a network failure
      setConnectionStatus('disconnected');
      setTimeout(checkConnection, 2000);
    } finally {
      clearTimeout(watchdog);
      isLoadingRef.current = false;
      setIsLoading(false);

      // Aggressive Reset: force unlock the UI
      setConnectionStatus('connected');

      // Ensure the AI message is no longer in "Thinking" state
      setMessages(prev => prev.map(msg =>
        msg.id === aiMessageId ? { ...msg, isThinking: false } : msg
      ));

      console.log("🏁 Chat flow completed, strictly unlocking input.");

      // Small delay to ensure the DOM has updated and element is enabled
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.disabled = false;
          inputRef.current.focus();
        }
      }, 300);
    }
  };

  const handleNewChat = async () => {
    try {
      await fetch(`${API_URL}/api/clear-history`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: currentSessionId })
      });

      const newSessionId = `research-session-${Date.now()}`;
      setCurrentSessionId(newSessionId);
      setMessages([
        {
          id: 1,
          text: "Welcome to the Research Assistant! 🌊 I'm here to help you with aquatic research, conservation strategies, hydrological analysis, and ecosystem management. Upload your research documents and ask me anything!",
          isUser: false
        }
      ]);

      await loadSessions();
    } catch (error) {
      console.error('Error clearing history:', error);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    try {
      const response = await fetch(`${API_URL}/api/session/${sessionId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        await loadSessions();
        if (sessionId === currentSessionId) {
          handleNewChat();
        }
      }
    } catch (error) {
      console.error('Error deleting session:', error);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getModelIcon = (modelKey) => {
    const model = availableModels.find(m => m.key === modelKey);
    if (model) {
      // Return different icons based on model type
      if (model.key === 'deepseek-v3') return '🧠'; // Brain icon for DeepSeek-V3
      if (model.key === 'gpt-oss-20b') return '🔬'; // Microscope for GPT-OSS-20B
      return model.type === 'deepseek' ? '🧠' : '🔬';
    }
    return '🌊';
  };

  const getCurrentModelName = () => {
    const model = availableModels.find(m => m.key === selectedModel);
    return model ? model.name : 'Research Assistant';
  };

  const showSources = (sources) => {
    setCurrentSources(sources);
    setShowSourcesPanel(true);
  };

  // Research color theme - updated to aquatic blue theme
  const researchColors = {
    primary: '#1a759f',      // Deep ocean blue
    secondary: '#168aad',    // Medium blue
    accent: '#52b788',       // Green accent for balance
    deepBlue: '#184e77',     // Darker blue
    lightBlue: '#76c893',    // Light green-blue
    lightBg: '#f0f9ff',      // Very light blue
    darkBg: '#0a2342',       // Dark navy blue
  };

  // Fixed Send Icon component
  const SendIcon = ({ disabled, isDark }) => {
    const fillColor = disabled
      ? (isDark ? '#9ca3af' : '#6b7280')
      : '#ffffff';

    return (
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill={fillColor}
        xmlns="http://www.w3.org/2000/svg"
      >
        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
      </svg>
    );
  };

  const handleLogout = () => {
    logoutUser();
    setUser(null);
  };

  if (!user) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        width: '100vw',
        backgroundColor: darkMode ? researchColors.darkBg : researchColors.lightBg
      }}>
        <div style={{
          backgroundColor: darkMode ? '#111827' : '#ffffff',
          padding: '40px',
          borderRadius: '16px',
          boxShadow: '0 10px 25px rgba(0,0,0,0.2)',
          textAlign: 'center',
          maxWidth: '400px',
          width: '90%',
          border: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`
        }}>
          <div style={{ fontSize: '48px', marginBottom: '20px' }}>🔬</div>
          <h1 style={{ fontSize: '24px', fontWeight: '700', marginBottom: '8px', color: darkMode ? '#ffffff' : '#111827' }}>WCPA Assistant</h1>
          <p style={{ color: darkMode ? '#9ca3af' : '#6b7280', marginBottom: '32px' }}>Welcome to the Wetland Conservation research portal. Please sign in to continue.</p>
          <Login onLogin={setUser} />
        </div>
      </div>
    );
  }

  return (
    <div className={`app-container ${darkMode ? 'dark-mode' : ''}`} style={{
      display: 'flex',
      height: '100vh',
      width: '100vw',
      overflow: 'hidden',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
      backgroundColor: 'var(--bg-color)',
      color: 'var(--text-color)'
    }}>
      <style>{`
        .pulse-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background-color: #1a759f;
          animation: pulse 1.5s infinite ease-in-out;
        }
        @keyframes pulse {
          0% { transform: scale(0.8); opacity: 0.5; }
          50% { transform: scale(1.2); opacity: 1; }
          100% { transform: scale(0.8); opacity: 0.5; }
        }
      `}</style>
      {/* Connection Status Banner */}
      {connectionStatus !== 'connected' && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          padding: '8px',
          backgroundColor: connectionStatus === 'checking' ? '#fbbf24' : '#ef4444',
          color: '#ffffff',
          textAlign: 'center',
          fontSize: '14px',
          fontWeight: '500',
          zIndex: 1000
        }}>
          {connectionStatus === 'checking' ? '🔄 Connecting to research backend...' : `❌ Backend disconnected. Make sure Flask is running on ${API_URL}`}
        </div>
      )}

      {/* Sources Panel */}
      {showSourcesPanel && (
        <div style={{
          position: 'fixed',
          right: 0,
          top: connectionStatus !== 'connected' ? '32px' : '0',
          bottom: 0,
          width: '400px',
          backgroundColor: darkMode ? '#111827' : '#ffffff',
          borderLeft: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`,
          zIndex: 999,
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{
            padding: '16px',
            borderBottom: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <h3 style={{ margin: 0, fontSize: '16px', fontWeight: '600' }}>📚 Research Sources</h3>
            <button
              onClick={() => setShowSourcesPanel(false)}
              style={{
                background: 'none',
                border: 'none',
                color: darkMode ? '#9ca3af' : '#6b7280',
                cursor: 'pointer',
                fontSize: '20px'
              }}
            >
              ×
            </button>
          </div>
          <div style={{ flex: 1, overflow: 'auto', padding: '16px' }}>
            {currentSources.map((source, idx) => (
              <div key={idx} style={{
                marginBottom: '16px',
                padding: '12px',
                backgroundColor: darkMode ? '#1f2937' : '#f9fafb',
                borderRadius: '8px',
                border: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontWeight: '600', fontSize: '14px', color: darkMode ? '#f3f4f6' : '#111827' }}>
                    #{source.rank} {source.filename}
                  </span>
                  <span style={{
                    padding: '2px 8px',
                    backgroundColor: darkMode ? '#374151' : '#e5e7eb',
                    borderRadius: '4px',
                    fontSize: '12px',
                    color: darkMode ? '#d1d5db' : '#6b7280'
                  }}>
                    Page {source.page} | Score: {source.relevance_score?.toFixed(3) || 'N/A'}
                  </span>
                </div>
                <div style={{
                  fontSize: '13px',
                  color: darkMode ? '#d1d5db' : '#4b5563',
                  lineHeight: '1.5',
                  fontStyle: 'italic'
                }}>
                  "{source.content}..."
                </div>
                <div style={{
                  marginTop: '8px',
                  fontSize: '11px',
                  color: darkMode ? '#9ca3af' : '#6b7280',
                  textTransform: 'uppercase'
                }}>
                  Type: {source.type}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Sidebar Toggle Button */}
      <div
        className="sidebar-toggle-btn"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        style={{
          position: 'absolute',
          top: '8px',
          left: sidebarOpen ? '290px' : '16px',
          zIndex: 50,
          background: 'transparent',
          color: darkMode ? 'white' : 'black',
          border: 'none',
          padding: '8px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          transition: 'left 0.3s ease',
          userSelect: 'none',
          marginTop: connectionStatus !== 'connected' ? '32px' : '0'
        }}
        title={sidebarOpen ? "Minimize Sidebar" : "Expand Sidebar"}
      >
        {sidebarOpen ? (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="11 17 6 12 11 7"></polyline>
            <polyline points="18 17 13 12 18 7"></polyline>
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="13 17 18 12 13 7"></polyline>
            <polyline points="6 17 11 12 6 7"></polyline>
          </svg>
        )}
      </div>

      {/* Sidebar */}
      <div className={`sidebar ${sidebarOpen ? '' : 'closed'}`} style={{
        display: sidebarOpen ? 'flex' : 'none',
        transition: 'all 0.3s ease',
        overflow: 'hidden',
        marginTop: connectionStatus !== 'connected' ? '32px' : '0'
      }}>

        <div className="sidebar-profile-card">
          <div className="sidebar-profile-avatar">
            <img src={user?.picture || 'https://cdn-icons-png.flaticon.com/512/149/149071.png'} alt="Profile" style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover' }} />
          </div>
          <div style={{ fontWeight: 'bold', fontSize: '15px', marginBottom: '2px' }}>
            {user?.name || 'Guest'}
          </div>
          <div style={{ fontSize: '11px', opacity: 0.9 }}>
            {user?.email || 'guest@local'}
          </div>
        </div>

        <button className="sidebar-btn btn-teal" onClick={handleLogout}>
          ↗️ Sign Out
        </button>

        <button className="sidebar-btn btn-green-dark" onClick={() => setDarkMode(!darkMode)}>
          {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
        </button>

        <div style={{ marginTop: '8px' }}>
          <div className="sidebar-section-title">CHAT MANAGEMENT</div>
          <div className="sidebar-row">
            <button className="sidebar-btn btn-green-light" style={{ flex: 1 }} onClick={handleNewChat}>
              ✨ New
            </button>
            <button className="sidebar-btn btn-green-dark" style={{ flex: 1 }} onClick={() => setMessages([])}>
              🗑️ Clear
            </button>
          </div>
        </div>

        <div style={{ width: '100%' }}>
          <button className="sidebar-btn btn-outline" style={{ justifyContent: 'flex-start', paddingLeft: '12px' }}>
            › 💬 Conversations
          </button>
          <div style={{ padding: '8px 0 0 16px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {sessions.slice(0, 5).map(session => (
              <div
                key={session.session_id}
                onClick={() => loadSession(session.session_id)}
                style={{
                  padding: '6px 8px',
                  fontSize: '13px',
                  cursor: 'pointer',
                  color: session.session_id === currentSessionId ? '#16a34a' : '#4b5563',
                  fontWeight: session.session_id === currentSessionId ? '600' : '400',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis'
                }}
              >
                {session.title}
              </div>
            ))}
          </div>
        </div>

        <div style={{ marginTop: '8px' }}>
          <div className="sidebar-section-title">EXPORT CHAT</div>
          <button className="sidebar-btn btn-green-dark" onClick={() => alert('PDF Download functionality goes here.')}>
            📄 Download PDF
          </button>
        </div>

        <div style={{ marginTop: 'auto', paddingTop: '16px' }}>
          <button className="sidebar-btn btn-outline" style={{ justifyContent: 'flex-start', paddingLeft: '12px' }}>
            › ℹ️ About Marsh Warden
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-chat-container" style={{
        marginTop: connectionStatus !== 'connected' ? '32px' : '0'
      }}>
        {/* Header */}
        <header className="header-banner">
          <div className="header-badge">
            🌍 Powered by IWMI Research
          </div>
          <h1>Marsh Warden</h1>
          <p>Wetland Information & Conservation Policy support Assistant - Sri Lanka</p>
          <div className="header-image-container">
            <img src="/wetland_banner.png" alt="Wetland Background" />
          </div>
        </header>

        {/* Messages Area */}
        <div className="chat-scroll-area" style={{
          backgroundColor: darkMode ? '#111827' : '#f8fafc'
        }}>
          {messages.filter(m => m.isUser).length >= 12 && (
            <div style={{
              width: '100%',
              maxWidth: '800px',
              backgroundColor: '#fffbeb',
              border: '1px solid #f59e0b',
              borderRadius: '8px',
              padding: '12px 16px',
              marginBottom: '16px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '8px',
              textAlign: 'center',
              color: '#92400e',
              fontWeight: '500',
              animation: 'fadeIn 0.3s ease-in-out'
            }}>
              <div>{messages.filter(m => m.isUser).length >= 15
                ? "⚠️ You have reached the limit of 15 questions for this session."
                : `💡 Note: You have asked ${messages.filter(m => m.isUser).length} questions. We recommend starting a new chat after 15 questions.`}</div>
              <button
                onClick={handleNewChat}
                style={{
                  padding: '6px 16px',
                  backgroundColor: '#92400e',
                  color: 'white',
                  border: 'none',
                  borderRadius: '20px',
                  cursor: 'pointer',
                  fontSize: '13px',
                  fontWeight: '600',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}
              >
                Start New Chat Click After Automatically Get The New Chat
              </button>
            </div>
          )}

          {messages.length === 0 && (
            <div className="welcome-card">
              <h3>🌿 Welcome Wetland Information & Conservation Policy support Assistant - Sri Lanka</h3>
              <p>I'm your AI-powered expert for wetland conservation and environmental policy analysis. Ask me questions about:</p>
              <ul className="welcome-list">
                <li>Wetland conservation strategies and restoration techniques</li>
                <li>Environmental policy frameworks and regulatory compliance</li>
                <li>Nature-based solutions and ecosystem services valuation</li>
                <li>Climate adaptation and mitigation through wetland management</li>
                <li>Sustainable development and biodiversity conservation policies</li>
              </ul>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              style={{
                width: '100%',
                maxWidth: '800px',
                display: 'flex',
                gap: '12px',
                alignItems: 'flex-start',
                justifyContent: message.isUser ? 'flex-end' : 'flex-start',
                marginTop: '16px'
              }}
            >
              {!message.isUser && (
                <div style={{
                  width: '36px',
                  height: '36px',
                  minWidth: '36px',
                  borderRadius: '50%',
                  backgroundColor: selectedModel === 'deepseek-v3' ? researchColors.primary : researchColors.deepBlue,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#ffffff',
                  fontWeight: '600',
                  fontSize: '14px',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  {getModelIcon(selectedModel)}
                </div>
              )}

              <div style={{
                flex: message.isUser ? '0 1 auto' : 1,
                minWidth: '100px',
                width: message.isUser ? 'auto' : '100%'
              }}>
                <div style={{
                  padding: '12px 16px',
                  borderRadius: '12px',
                  backgroundColor: message.isUser
                    ? researchColors.deepBlue
                    : (darkMode ? '#374151' : '#ffffff'),
                  color: message.isUser ? '#ffffff' : (darkMode ? '#e5e7eb' : '#111827'),
                  boxShadow: darkMode ? 'none' : '0 2px 8px rgba(0,0,0,0.08)',
                  border: message.isUser ? 'none' : `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`,
                  fontSize: '15px',
                  lineHeight: '1.5'
                }}>
                  {message.isUser ? message.text : (
                    message.isThinking ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div className="pulse-dot"></div>
                        <span style={{ fontStyle: 'italic', opacity: 0.8 }}>
                          {message.thinkingText || "Researching..."}
                        </span>
                      </div>
                    ) : (
                      <>
                        {message.mode && (
                          <div style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '4px',
                            marginBottom: '8px',
                            padding: '2px 8px',
                            borderRadius: '10px',
                            fontSize: '11px',
                            fontWeight: '600',
                            backgroundColor: message.mode === 'fast' ? 'rgba(26,117,159,0.15)' : 'rgba(24,78,119,0.15)',
                            color: message.mode === 'fast' ? '#1a759f' : '#184e77',
                            border: `1px solid ${message.mode === 'fast' ? 'rgba(26,117,159,0.3)' : 'rgba(24,78,119,0.3)'}`
                          }}>
                            {message.mode === 'fast' ? '⚡ Fast Mode' : '🧠 Thinking Mode'}
                          </div>
                        )}
                        {renderFormattedText(message.text, darkMode)}
                      </>
                    )
                  )}
                </div>

                {!message.isUser && message.sources && message.sources.length > 0 && (
                  <div style={{
                    marginTop: '8px',
                    padding: '12px',
                    borderRadius: '8px',
                    backgroundColor: darkMode ? '#1f2937' : '#f9fafb',
                    fontSize: '12px',
                    border: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`
                  }}>
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: '8px'
                    }}>
                      <div style={{ fontWeight: '600', color: darkMode ? '#9ca3af' : '#6b7280' }}>
                        📚 Research Sources ({message.sources.length})
                      </div>
                      <button
                        onClick={() => showSources(message.sources)}
                        style={{
                          padding: '4px 8px',
                          backgroundColor: 'transparent',
                          border: `1px solid ${darkMode ? '#4b5563' : '#d1d5db'}`,
                          borderRadius: '4px',
                          fontSize: '11px',
                          color: darkMode ? '#9ca3af' : '#6b7280',
                          cursor: 'pointer'
                        }}
                      >
                        View Research Details
                      </button>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      {message.sources.slice(0, 4).map((source, idx) => (
                        <div key={idx} style={{
                          padding: '4px 8px',
                          backgroundColor: darkMode ? '#374151' : '#e5e7eb',
                          borderRadius: '4px',
                          fontSize: '11px',
                          color: darkMode ? '#d1d5db' : '#4b5563'
                        }}>
                          {source.filename} (p{source.page})
                        </div>
                      ))}
                      {message.sources.length > 4 && (
                        <div style={{
                          padding: '4px 8px',
                          backgroundColor: darkMode ? '#374151' : '#e5e7eb',
                          borderRadius: '4px',
                          fontSize: '11px',
                          color: darkMode ? '#9ca3af' : '#6b7280',
                          fontStyle: 'italic'
                        }}>
                          +{message.sources.length - 4} more
                        </div>
                      )}
                    </div>
                    {message.retrievalStats && (
                      <div style={{
                        marginTop: '8px',
                        paddingTop: '8px',
                        borderTop: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`,
                        fontSize: '11px',
                        color: darkMode ? '#9ca3af' : '#6b7280'
                      }}>
                        Research search: {message.retrievalStats.retrieved_chunks} chunks → {message.retrievalStats.reranked_chunks} reranked → {message.retrievalStats.final_chunks} used
                      </div>
                    )}
                  </div>
                )}
              </div>

              {message.isUser && (
                <div style={{
                  width: '36px',
                  height: '36px',
                  minWidth: '36px',
                  borderRadius: '50%',
                  backgroundColor: researchColors.deepBlue,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: '600',
                  fontSize: '14px',
                  color: '#ffffff',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                }}>
                  👤
                </div>
              )}
            </div>
          ))}


          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="input-container-wrapper">
          <div className="input-footer-text">
            <span>🌿 Marsh Warden - Empowering Evidence-Based Decisions</span>
          </div>

          <div className="floating-input-pill">
            <div className="pill-icon-group">
              <button
                className={`mode-toggle-btn ${searchMode === 'fast' ? 'active' : ''}`}
                onClick={() => setSearchMode('fast')}
                title="Marsh Fast: Quick direct answers (applies to your NEXT question)"
                style={{
                  padding: '4px 10px',
                  borderRadius: '16px',
                  fontSize: '11px',
                  fontWeight: '600',
                  border: 'none',
                  backgroundColor: searchMode === 'fast' ? researchColors.primary : 'transparent',
                  color: searchMode === 'fast' ? 'white' : '#9ca3af',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  marginRight: '2px',
                  opacity: isLoading ? 0.7 : 1
                }}
              >
                ⚡ Fast
              </button>
              <button
                className={`mode-toggle-btn ${searchMode === 'thinking' ? 'active' : ''}`}
                onClick={() => setSearchMode('thinking')}
                title="Marsh Thinking: Deep research ReAct loop (applies to your NEXT question)"
                style={{
                  padding: '4px 10px',
                  borderRadius: '16px',
                  fontSize: '11px',
                  fontWeight: '600',
                  border: 'none',
                  backgroundColor: searchMode === 'thinking' ? researchColors.deepBlue : 'transparent',
                  color: searchMode === 'thinking' ? 'white' : '#9ca3af',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  opacity: isLoading ? 0.7 : 1
                }}
              >
                🧠 Thinking
              </button>
              {isLoading && (
                <span style={{
                  fontSize: '10px',
                  color: '#9ca3af',
                  fontStyle: 'italic',
                  alignSelf: 'center',
                  marginLeft: '4px'
                }}>
                  (next question)
                </span>
              )}
            </div>

            <textarea
              ref={inputRef}
              className="pill-textarea"
              placeholder="Start typing to talk with RAG Agent"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading || messages.filter(m => m.isUser).length >= 15}
              rows={1}
            />

            <button className="pill-icon-btn" style={{ marginLeft: '8px', color: '#9ca3af' }} title="Voice Input">
              <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 10v2a7 7 0 01-14 0v-2m7 9v-2m-4 0h8m-4-14a3 3 0 00-3 3v4a3 3 0 106 0V6a3 3 0 00-3-3z"></path></svg>
            </button>

            <button
              className="pill-send-btn"
              onClick={handleSendMessage}
              disabled={!inputText.trim() || isLoading || messages.filter(m => m.isUser).length >= 15}
              style={{
                color: (!inputText.trim() || isLoading || messages.filter(m => m.isUser).length >= 15) ? '#64748b' : 'white'
              }}
              title="Send Message"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="currentColor"
                style={{ width: '20px', height: '20px', flexShrink: 0 }}
              >
                <path d="M3.478 2.404a.75.75 0 00-.926.941l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.404z" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes bounce {
          0%, 60%, 100% { transform: translateY(0); }
          30% { transform: translateY(-10px); }
        }
      `}</style>
    </div >
  );
}

export default App;
