/**
 * AI Agent Gateway - Full Application Controller
 * ============================================================================
 * Vanilla ES6+ Asynchronous Single Page Application
 * Features:
 *   - Real-time Server-Sent Events (SSE) / ReadableStream token-by-token rendering
 *   - Zero-dependency custom Markdown & Code parser with copy tools
 *   - Multi-agent orchestration (Portfolio, E-commerce, Analytics, General)
 *   - Session persistence & conversation management
 *   - Health monitoring & live connection status
 *   - Toast notifications & chat export (JSON / Markdown)
 * ============================================================================
 */

(function () {
  'use strict';

  // --- 1. Agent Definitions & Prompt Starters ---
  const AGENT_CATALOG = {
    portfolio: {
      id: 'portfolio',
      name: 'Portfolio Agent',
      icon: '💼',
      role: 'Proyectos y CV',
      desc: 'Experiencia profesional, CV, casos de estudio y stack tecnológico',
      welcome: 'Bienvenido al Portfolio Agent',
      welcomeSubtitle: 'Puedo responder preguntas sobre proyectos, tecnologías utilizadas, casos de estudio y trayectoria profesional.',
      starters: [
        '¿Cuáles son los proyectos más destacados y sus arquitecturas?',
        '¿Qué stack tecnológico dominas para frontend y backend?',
        'Explica la arquitectura del microservicio de IA con FastAPI',
        '¿Cómo está diseñado el pipeline de streaming con Google GenAI?'
      ]
    },
    ecommerce: {
      id: 'ecommerce',
      name: 'E-commerce Agent',
      icon: '🛍️',
      role: 'Catálogo y Ventas',
      desc: 'Catálogo de productos, cotizaciones, soporte de pedidos y pagos',
      welcome: 'Bienvenido al E-commerce Agent',
      welcomeSubtitle: 'Asistente especializado en catálogo de productos, recomendaciones personalizadas, cotizaciones y pedidos.',
      starters: [
        '¿Qué soluciones o servicios de software tienes disponibles?',
        'Cotizar una integración de chatbot de IA para mi negocio',
        '¿Cuáles son los métodos de pago y garantías ofrecidas?',
        'Recomiéndame la mejor arquitectura para una tienda online'
      ]
    },
    analytics: {
      id: 'analytics',
      name: 'Analytics Agent',
      icon: '📊',
      role: 'Métricas e Informes',
      desc: 'Métricas de negocio, KPIs, rendimiento de modelos y análisis',
      welcome: 'Bienvenido al Analytics Agent',
      welcomeSubtitle: 'Especialista en análisis de métricas, rendimiento de inferencia LLM, consumo de tokens y KPIs de negocio.',
      starters: [
        'Genera un reporte de métricas de rendimiento y latencia',
        '¿Cómo optimizar el consumo de tokens y caché en Redis?',
        'Métricas de conversión y engagement de los agentes',
        'Explica el flujo de observabilidad y logging estructurado'
      ]
    },
    general: {
      id: 'general',
      name: 'General Assistant',
      icon: '⚡',
      role: 'Asistente Multi-Agente',
      desc: 'Orquestador general, consultas abiertas y asistencia integral',
      welcome: 'Bienvenido al General Assistant',
      welcomeSubtitle: 'Orquestador inteligente con acceso a herramientas y respuestas abiertas para cualquier consulta.',
      starters: [
        '¿Qué agentes están disponibles y qué capacidades tienen?',
        '¿Cómo funciona la memoria de conversación distribuida?',
        'Ayúdame a redactar la documentación técnica del gateway',
        '¿Cómo integrar el widget drop-in en una aplicación web?'
      ]
    }
  };

  // --- 2. Markdown & Code Parser Engine (Zero Dependencies) ---
  class MarkdownEngine {
    static escapeHtml(str) {
      if (typeof str !== 'string') return '';
      return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    static render(markdownText) {
      if (!markdownText) return '';

      let text = markdownText.replace(/\r\n/g, '\n');

      // 1. Isolate and preserve fenced code blocks
      const codeBlocks = [];
      text = text.replace(/```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
        codeBlocks.push({ lang: lang.trim() || 'plaintext', code });
        return placeholder;
      });

      // 2. Isolate inline code
      const inlineCodes = [];
      text = text.replace(/`([^`\n]+)`/g, (match, code) => {
        const placeholder = `__INLINE_CODE_${inlineCodes.length}__`;
        inlineCodes.push(this.escapeHtml(code));
        return placeholder;
      });

      // 3. Escape HTML in ordinary text to avoid XSS
      text = this.escapeHtml(text);

      // 4. Headers
      text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
      text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>');
      text = text.replace(/^# (.*$)/gim, '<h1>$1</h1>');

      // 5. Blockquotes
      text = text.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

      // 6. Bold, Italics, Strikethrough
      text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      text = text.replace(/__([^_]+)__/g, '<strong>$1</strong>');
      text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
      text = text.replace(/_([^_]+)_/g, '<em>$1</em>');
      text = text.replace(/~~([^~]+)~~/g, '<del>$1</del>');

      // 7. Links with security attributes
      text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

      // 8. Horizontal rules
      text = text.replace(/^---$/gim, '<hr>');

      // 9. Markdown Tables
      text = text.replace(/((?:\|[^\n]+\|\r?\n)+)/g, (tableMatch) => {
        const lines = tableMatch.trim().split('\n').filter(l => l.trim().length > 0);
        if (lines.length < 2) return tableMatch;

        let tableHtml = '<div class="table-wrapper"><table>';
        let isHead = true;

        lines.forEach((line, idx) => {
          if (line.includes('---')) return; // delimiter line
          const cells = line.split('|').map(c => c.trim()).filter((c, i, arr) => i > 0 && i < arr.length - 1);
          
          if (idx === 0) {
            tableHtml += '<thead><tr>';
            cells.forEach(cell => { tableHtml += `<th>${cell}</th>`; });
            tableHtml += '</tr></thead><tbody>';
          } else {
            tableHtml += '<tr>';
            cells.forEach(cell => { tableHtml += `<td>${cell}</td>`; });
            tableHtml += '</tr>';
          }
        });

        tableHtml += '</tbody></table></div>';
        return tableHtml;
      });

      // 10. Ordered and Unordered Lists
      text = text.replace(/((?:^(?:[\*\-\+]|\d+\.) .*(?:\n|$))+)/gim, (listMatch) => {
        const isOrdered = /^\d+\./.test(listMatch.trim());
        const tag = isOrdered ? 'ol' : 'ul';
        const items = listMatch.trim().split('\n').map(item => {
          const itemText = item.replace(/^(?:[\*\-\+]|\d+\.)\s+/, '');
          return `<li>${itemText}</li>`;
        }).join('');
        return `<${tag}>${items}</${tag}>`;
      });

      // 11. Paragraphs
      const paragraphs = text.split(/\n{2,}/);
      text = paragraphs.map(p => {
        p = p.trim();
        if (!p) return '';
        if (p.startsWith('<h') || p.startsWith('<blockquote') || p.startsWith('<div') || 
            p.startsWith('<ul') || p.startsWith('<ol') || p.startsWith('<hr') || p.startsWith('__CODE_BLOCK_')) {
          return p;
        }
        return `<p>${p.replace(/\n/g, '<br>')}</p>`;
      }).join('\n');

      // 12. Restore inline codes
      inlineCodes.forEach((code, idx) => {
        text = text.replace(`__INLINE_CODE_${idx}__`, `<code>${code}</code>`);
      });

      // 13. Restore code blocks with interactive Header & Copy Button
      codeBlocks.forEach((block, idx) => {
        const escapedCode = this.escapeHtml(block.code);
        const codeBlockHtml = `
          <div class="code-block-container" data-block-id="${idx}">
            <div class="code-header">
              <span class="code-language">${block.lang}</span>
              <button type="button" class="btn-copy-code" data-raw-code="${this.escapeHtml(block.code)}">
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                <span class="copy-label">Copiar código</span>
              </button>
            </div>
            <pre><code class="language-${block.lang}">${escapedCode}</code></pre>
          </div>
        `;
        text = text.replace(`__CODE_BLOCK_${idx}__`, codeBlockHtml);
      });

      return text;
    }
  }

  // --- 3. Main Application Class ---
  class ChatbotApp {
    constructor() {
      // Configuration & Persistent State
      this.apiBaseUrl = localStorage.getItem('ai_gateway_api_url') || 'http://localhost:8000';
      this.streamMode = localStorage.getItem('ai_gateway_stream_mode') || 'sse'; // 'sse' or 'sync'
      this.activeAgent = localStorage.getItem('ai_gateway_active_agent') || 'portfolio';
      this.sessionId = this._initSessionId();
      
      // Runtime State
      this.messages = [];
      this.isStreaming = false;
      this.isConnected = false;
      this.abortController = null;
      this._healthCheckTimer = null;

      // DOM Elements Cache
      this._cacheDom();

      // Initialization
      this._init();
    }

    // --- Session ID Management ---
    _initSessionId() {
      let sid = sessionStorage.getItem('ai_gateway_session_id') || localStorage.getItem('ai_gateway_session_id');
      if (!sid) {
        sid = 'sess_' + Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
        sessionStorage.setItem('ai_gateway_session_id', sid);
        localStorage.setItem('ai_gateway_session_id', sid);
      }
      return sid;
    }

    _generateNewSessionId() {
      const newSid = 'sess_' + Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
      this.sessionId = newSid;
      sessionStorage.setItem('ai_gateway_session_id', newSid);
      localStorage.setItem('ai_gateway_session_id', newSid);
      this._updateSessionUI();
      return newSid;
    }

    // --- Cache DOM References ---
    _cacheDom() {
      this.dom = {
        app: document.getElementById('app'),
        // Header
        agentSelectContainer: document.getElementById('agent-select-container'),
        agentTrigger: document.getElementById('agent-trigger'),
        activeAgentIcon: document.getElementById('active-agent-icon'),
        activeAgentName: document.getElementById('active-agent-name'),
        activeAgentDesc: document.getElementById('active-agent-desc'),
        agentOptions: document.getElementById('agent-options'),
        statusBadge: document.getElementById('connection-status-badge'),
        statusDot: document.getElementById('status-dot'),
        statusLabel: document.getElementById('status-label'),
        btnNewChat: document.getElementById('btn-new-chat'),
        btnExportChat: document.getElementById('btn-export-chat'),
        btnOpenSettings: document.getElementById('btn-open-settings'),
        
        // Chat Area
        chatScrollContainer: document.getElementById('chat-scroll-container'),
        welcomeScreen: document.getElementById('welcome-screen'),
        welcomeAvatar: document.getElementById('welcome-avatar'),
        welcomeTitle: document.getElementById('welcome-title'),
        welcomeSubtitle: document.getElementById('welcome-subtitle'),
        startersGrid: document.getElementById('starters-grid'),
        messagesContainer: document.getElementById('messages-container'),
        typingIndicator: document.getElementById('typing-indicator'),
        typingAvatar: document.getElementById('typing-avatar'),
        btnScrollBottom: document.getElementById('btn-scroll-bottom'),
        scrollAnchor: document.getElementById('scroll-anchor'),

        // Footer Input
        chatForm: document.getElementById('chat-form'),
        messageInput: document.getElementById('message-input'),
        charCounter: document.getElementById('char-counter'),
        btnSend: document.getElementById('btn-send'),
        btnAbortStream: document.getElementById('btn-abort-stream'),
        footerSessionId: document.getElementById('footer-session-id'),
        sessionDisplay: document.getElementById('session-display'),

        // Settings Modal
        settingsModal: document.getElementById('settings-modal'),
        btnCloseSettings: document.getElementById('btn-close-settings'),
        btnCancelSettings: document.getElementById('btn-cancel-settings'),
        btnSaveSettings: document.getElementById('btn-save-settings'),
        inputApiUrl: document.getElementById('input-api-url'),
        btnTestConnection: document.getElementById('btn-test-connection'),
        inputSessionId: document.getElementById('input-session-id'),
        btnCopySession: document.getElementById('btn-copy-session'),
        btnRegenSession: document.getElementById('btn-regen-session'),
        btnClearLocalStorage: document.getElementById('btn-clear-local-storage'),
        radioStreamModes: document.getElementsByName('stream_mode'),

        // Confirm Modal
        confirmModal: document.getElementById('confirm-modal'),
        btnCloseConfirm: document.getElementById('btn-close-confirm'),
        btnCancelConfirm: document.getElementById('btn-cancel-confirm'),
        btnConfirmClear: document.getElementById('btn-confirm-clear'),

        // Toast Container
        toastContainer: document.getElementById('toast-container')
      };
    }

    // --- App Lifecycle Initialization ---
    _init() {
      this._updateAgentUI();
      this._updateSessionUI();
      this._loadStoredMessages();
      this._bindEvents();
      this._checkHealth();

      // Check health every 30 seconds
      this._healthCheckTimer = setInterval(() => this._checkHealth(), 30000);
      
      // Also ping on window focus
      window.addEventListener('focus', () => this._checkHealth());
    }

    // --- Event Listeners ---
    _bindEvents() {
      const { dom } = this;

      // 1. Agent Select Dropdown Toggle
      dom.agentTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        const isOpen = dom.agentSelectContainer.classList.toggle('open');
        dom.agentTrigger.setAttribute('aria-expanded', isOpen);
      });

      // Close dropdown when clicking outside
      document.addEventListener('click', (e) => {
        if (!dom.agentSelectContainer.contains(e.target)) {
          dom.agentSelectContainer.classList.remove('open');
          dom.agentTrigger.setAttribute('aria-expanded', 'false');
        }
      });

      // Agent Option Click
      dom.agentOptions.addEventListener('click', (e) => {
        const option = e.target.closest('.option-item');
        if (option) {
          const selectedAgent = option.dataset.agent;
          this.setAgent(selectedAgent);
          dom.agentSelectContainer.classList.remove('open');
          dom.agentTrigger.setAttribute('aria-expanded', 'false');
        }
      });

      // 2. Input Handling & Auto-resize
      dom.messageInput.addEventListener('input', () => {
        this._autoResizeInput();
        const length = dom.messageInput.value.length;
        dom.charCounter.textContent = `${length}/4000`;
        dom.btnSend.disabled = !dom.messageInput.value.trim() || this.isStreaming;
      });

      dom.messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          if (!dom.btnSend.disabled) {
            this.handleSendMessage();
          }
        }
      });

      // 3. Form Submit
      dom.chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        this.handleSendMessage();
      });

      // 4. Abort Streaming Button
      dom.btnAbortStream.addEventListener('click', () => {
        if (this.abortController) {
          this.abortController.abort();
          this.showToast('Generación de respuesta cancelada', 'info');
        }
      });

      // 5. Scroll container & Floating Scroll-To-Bottom
      dom.chatScrollContainer.addEventListener('scroll', () => {
        const threshold = 150;
        const isNearBottom = dom.chatScrollContainer.scrollHeight - dom.chatScrollContainer.scrollTop - dom.chatScrollContainer.clientHeight < threshold;
        if (isNearBottom) {
          dom.btnScrollBottom.classList.add('hidden');
        } else {
          dom.btnScrollBottom.classList.remove('hidden');
        }
      });

      dom.btnScrollBottom.addEventListener('click', () => {
        this.scrollToBottom(true);
      });

      // 6. Prompt Starters / Suggestion Chips Click
      dom.startersGrid.addEventListener('click', (e) => {
        const chip = e.target.closest('.starter-chip');
        if (chip) {
          const prompt = chip.dataset.prompt;
          if (prompt) {
            dom.messageInput.value = prompt;
            this._autoResizeInput();
            dom.btnSend.disabled = false;
            this.handleSendMessage();
          }
        }
      });

      // 7. Message Body Delegated Actions (Copy code, Copy message, Retry)
      dom.messagesContainer.addEventListener('click', (e) => {
        // Copy code button
        const copyCodeBtn = e.target.closest('.btn-copy-code');
        if (copyCodeBtn) {
          const codeText = copyCodeBtn.dataset.rawCode;
          if (codeText) {
            navigator.clipboard.writeText(codeText).then(() => {
              const label = copyCodeBtn.querySelector('.copy-label');
              if (label) {
                const prev = label.textContent;
                label.textContent = '¡Copiado!';
                setTimeout(() => label.textContent = prev, 2000);
              }
              this.showToast('Código copiado al portapapeles', 'success');
            });
          }
          return;
        }

        // Copy message button
        const copyMsgBtn = e.target.closest('.btn-copy-message');
        if (copyMsgBtn) {
          const msgId = copyMsgBtn.dataset.msgId;
          const msg = this.messages.find(m => m.id === msgId);
          if (msg) {
            navigator.clipboard.writeText(msg.content).then(() => {
              this.showToast('Mensaje copiado al portapapeles', 'success');
            });
          }
          return;
        }

        // Retry button
        const retryBtn = e.target.closest('.btn-retry-message');
        if (retryBtn) {
          const lastUserMsg = [...this.messages].reverse().find(m => m.role === 'user');
          if (lastUserMsg) {
            dom.messageInput.value = lastUserMsg.content;
            this._autoResizeInput();
            this.handleSendMessage();
          }
          return;
        }
      });

      // 8. Header Action Modals
      dom.btnNewChat.addEventListener('click', () => {
        dom.confirmModal.classList.remove('hidden');
      });

      dom.btnCloseConfirm.addEventListener('click', () => dom.confirmModal.classList.add('hidden'));
      dom.btnCancelConfirm.addEventListener('click', () => dom.confirmModal.classList.add('hidden'));

      dom.btnConfirmClear.addEventListener('click', () => {
        this.clearConversation();
        dom.confirmModal.classList.add('hidden');
        this.showToast('Nueva sesión iniciada', 'success');
      });

      // Export conversation
      dom.btnExportChat.addEventListener('click', () => {
        this.exportConversation();
      });

      // Settings Modal
      dom.btnOpenSettings.addEventListener('click', () => {
        this._openSettingsModal();
      });

      dom.btnCloseSettings.addEventListener('click', () => dom.settingsModal.classList.add('hidden'));
      dom.btnCancelSettings.addEventListener('click', () => dom.settingsModal.classList.add('hidden'));
      dom.btnSaveSettings.addEventListener('click', () => this._saveSettings());

      dom.btnTestConnection.addEventListener('click', () => this._testCustomConnection());

      dom.btnCopySession.addEventListener('click', () => {
        navigator.clipboard.writeText(this.sessionId).then(() => {
          this.showToast('Session ID copiado', 'info');
        });
      });

      dom.btnRegenSession.addEventListener('click', () => {
        this._generateNewSessionId();
        dom.inputSessionId.value = this.sessionId;
        this.showToast('Nuevo Session ID generado', 'success');
      });

      dom.btnClearLocalStorage.addEventListener('click', () => {
        if (confirm('¿Estás seguro de que deseas borrar todo el almacenamiento local?')) {
          localStorage.clear();
          sessionStorage.clear();
          location.reload();
        }
      });

      // Click session display in footer to copy
      dom.footerSessionId.addEventListener('click', () => {
        navigator.clipboard.writeText(this.sessionId).then(() => {
          this.showToast(`Session ID copiado: ${this.sessionId}`, 'info');
        });
      });
    }

    // --- UI Helpers ---
    _autoResizeInput() {
      const input = this.dom.messageInput;
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 160) + 'px';
    }

    scrollToBottom(smooth = true) {
      const el = this.dom.chatScrollContainer;
      el.scrollTo({
        top: el.scrollHeight,
        behavior: smooth ? 'smooth' : 'auto'
      });
    }

    _updateSessionUI() {
      this.dom.sessionDisplay.textContent = this.sessionId.substring(0, 14) + '...';
      this.dom.sessionDisplay.title = this.sessionId;
      if (this.dom.inputSessionId) {
        this.dom.inputSessionId.value = this.sessionId;
      }
    }

    setAgent(agentId) {
      if (!AGENT_CATALOG[agentId]) return;
      this.activeAgent = agentId;
      localStorage.setItem('ai_gateway_active_agent', agentId);
      this._updateAgentUI();

      // If no conversation messages yet, update the welcome screen
      if (this.messages.length === 0) {
        this._renderWelcomeScreen();
      }

      this.showToast(`Agente activo: ${AGENT_CATALOG[agentId].name}`, 'info');
    }

    _updateAgentUI() {
      const agent = AGENT_CATALOG[this.activeAgent] || AGENT_CATALOG.portfolio;
      this.dom.activeAgentIcon.textContent = agent.icon;
      this.dom.activeAgentName.textContent = agent.name;
      this.dom.activeAgentDesc.textContent = agent.role;
      this.dom.typingAvatar.textContent = agent.icon;

      // Update options active state in dropdown
      this.dom.agentOptions.querySelectorAll('.option-item').forEach(opt => {
        const isMatch = opt.dataset.agent === this.activeAgent;
        opt.classList.toggle('active', isMatch);
        opt.setAttribute('aria-selected', isMatch ? 'true' : 'false');
      });

      this._renderWelcomeScreen();
    }

    _renderWelcomeScreen() {
      const agent = AGENT_CATALOG[this.activeAgent] || AGENT_CATALOG.portfolio;
      this.dom.welcomeAvatar.textContent = agent.icon;
      this.dom.welcomeTitle.textContent = agent.welcome;
      this.dom.welcomeSubtitle.textContent = agent.welcomeSubtitle;

      // Render Starters
      this.dom.startersGrid.innerHTML = agent.starters.map(starter => `
        <button type="button" class="starter-chip" data-prompt="${MarkdownEngine.escapeHtml(starter)}">
          <span class="starter-chip-icon">›</span>
          <span>${starter}</span>
        </button>
      `).join('');
    }

    // --- Message History & Storage Management ---
    _loadStoredMessages() {
      try {
        const stored = localStorage.getItem(`ai_chat_history_${this.sessionId}`);
        if (stored) {
          this.messages = JSON.parse(stored);
          this._renderMessageFeed();
        } else {
          this.dom.welcomeScreen.classList.remove('hidden');
        }
      } catch (err) {
        console.warn('Error loading chat history:', err);
        this.dom.welcomeScreen.classList.remove('hidden');
      }
    }

    _saveStoredMessages() {
      try {
        localStorage.setItem(`ai_chat_history_${this.sessionId}`, JSON.stringify(this.messages));
      } catch (err) {
        console.warn('Error persisting messages:', err);
      }
    }

    _renderMessageFeed() {
      this.dom.messagesContainer.innerHTML = '';
      if (this.messages.length === 0) {
        this.dom.welcomeScreen.classList.remove('hidden');
        return;
      }

      this.dom.welcomeScreen.classList.add('hidden');
      this.messages.forEach(msg => {
        this._renderMessageRow(msg);
      });
      this.scrollToBottom(false);
    }

    _renderMessageRow(msg) {
      const row = document.createElement('div');
      row.className = `message-row ${msg.role}`;
      row.id = `msg-${msg.id}`;

      if (msg.role === 'user') {
        row.innerHTML = `
          <div class="message-bubble">
            <p>${MarkdownEngine.escapeHtml(msg.content)}</p>
          </div>
          <div class="message-avatar user-avatar">Tú</div>
        `;
      } else if (msg.role === 'assistant') {
        const agent = AGENT_CATALOG[msg.agent_id || this.activeAgent] || AGENT_CATALOG.portfolio;
        const timeFormatted = new Date(msg.timestamp || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        row.innerHTML = `
          <div class="message-avatar">${agent.icon}</div>
          <div class="message-content-wrapper">
            <div class="message-meta-header">
              <span class="agent-badge-label">${agent.name}</span>
              <span class="message-timestamp">${timeFormatted}</span>
            </div>
            <div class="message-bubble markdown-body">
              ${MarkdownEngine.render(msg.content)}
            </div>
            <div class="message-toolbar">
              <button type="button" class="btn-toolbar-action btn-copy-message" data-msg-id="${msg.id}" title="Copiar respuesta">
                <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                <span>Copiar</span>
              </button>
              ${msg.metadata?.tokens_used ? `
                <span class="meta-stats">${msg.metadata.tokens_used} tokens • ${msg.metadata.latency_ms || ''}ms</span>
              ` : ''}
            </div>
          </div>
        `;
      } else {
        // System message
        row.innerHTML = `
          <div class="message-bubble">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="16" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <span>${MarkdownEngine.escapeHtml(msg.content)}</span>
          </div>
        `;
      }

      this.dom.messagesContainer.appendChild(row);
    }

    // --- Health Check Service ---
    async _checkHealth() {
      const { statusDot, statusLabel } = this.dom;
      statusDot.className = 'status-dot status-checking';
      statusLabel.textContent = 'Verificando...';

      try {
        const response = await fetch(`${this.apiBaseUrl}/health`, {
          method: 'GET',
          headers: { 'Accept': 'application/json' },
          mode: 'cors'
        });

        if (response.ok) {
          const data = await response.json();
          this.isConnected = true;
          statusDot.className = 'status-dot status-online';
          statusLabel.textContent = 'Online';
          this.dom.statusBadge.title = `Conectado a ${data.app_name || 'AI Gateway'} (v${data.version || '0.1.0'}) [${data.environment || 'dev'}]`;
        } else {
          throw new Error(`HTTP ${response.status}`);
        }
      } catch (err) {
        this.isConnected = false;
        statusDot.className = 'status-dot status-offline';
        statusLabel.textContent = 'Offline';
        this.dom.statusBadge.title = `Sin conexión con ${this.apiBaseUrl}. Asegúrate de iniciar FastAPI en localhost:8000`;
      }
    }

    async _testCustomConnection() {
      const url = this.dom.inputApiUrl.value.trim().replace(/\/$/, '');
      const btn = this.dom.btnTestConnection;
      btn.disabled = true;
      btn.textContent = 'Probando...';

      const startTime = performance.now();
      try {
        const res = await fetch(`${url}/health`, { method: 'GET', mode: 'cors' });
        const latency = Math.round(performance.now() - startTime);
        if (res.ok) {
          const data = await res.json();
          this.showToast(`Ping exitoso (${latency}ms): ${data.app_name || 'Gateway'} OK`, 'success');
        } else {
          throw new Error(`HTTP ${res.status}`);
        }
      } catch (err) {
        this.showToast(`Error al conectar con ${url}: ${err.message}`, 'error');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Test Ping';
      }
    }

    // --- Send & Streaming Pipeline ---
    async handleSendMessage() {
      const text = this.dom.messageInput.value.trim();
      if (!text || this.isStreaming) return;

      // 1. Reset input UI
      this.dom.messageInput.value = '';
      this.dom.charCounter.textContent = '0/4000';
      this._autoResizeInput();
      this.dom.btnSend.disabled = true;

      // 2. Hide welcome screen
      this.dom.welcomeScreen.classList.add('hidden');

      // 3. Append User Message
      const userMsg = {
        id: 'msg_user_' + Date.now(),
        role: 'user',
        content: text,
        timestamp: Date.now()
      };
      this.messages.push(userMsg);
      this._renderMessageRow(userMsg);
      this.scrollToBottom();
      this._saveStoredMessages();

      // 4. Prepare Placeholder Assistant Message
      const assistantMsg = {
        id: 'msg_ai_' + Date.now(),
        role: 'assistant',
        agent_id: this.activeAgent,
        content: '',
        timestamp: Date.now(),
        metadata: {}
      };
      this.messages.push(assistantMsg);

      // Create streaming row in DOM
      const row = document.createElement('div');
      row.className = 'message-row assistant';
      row.id = `msg-${assistantMsg.id}`;
      const agent = AGENT_CATALOG[this.activeAgent] || AGENT_CATALOG.portfolio;
      const timeFormatted = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      row.innerHTML = `
        <div class="message-avatar">${agent.icon}</div>
        <div class="message-content-wrapper">
          <div class="message-meta-header">
            <span class="agent-badge-label">${agent.name}</span>
            <span class="message-timestamp">${timeFormatted}</span>
          </div>
          <div class="message-bubble markdown-body">
            <span class="streaming-cursor"></span>
          </div>
          <div class="message-toolbar hidden" id="toolbar-${assistantMsg.id}">
            <button type="button" class="btn-toolbar-action btn-copy-message" data-msg-id="${assistantMsg.id}" title="Copiar respuesta">
              <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
              </svg>
              <span>Copiar</span>
            </button>
            <span class="meta-stats" id="stats-${assistantMsg.id}"></span>
          </div>
        </div>
      `;
      this.dom.messagesContainer.appendChild(row);
      this.scrollToBottom();

      const bubble = row.querySelector('.message-bubble');
      const toolbar = row.querySelector(`#toolbar-${assistantMsg.id}`);
      const stats = row.querySelector(`#stats-${assistantMsg.id}`);

      // 5. Update streaming state
      this.isStreaming = true;
      this.dom.btnAbortStream.classList.remove('hidden');
      this.dom.typingIndicator.classList.remove('hidden');
      this.abortController = new AbortController();

      const requestPayload = {
        agent_id: this.activeAgent,
        session_id: this.sessionId,
        message: text,
        stream: this.streamMode === 'sse'
      };

      const startTime = performance.now();
      let tokenCount = 0;

      try {
        const endpoint = this.streamMode === 'sse' 
          ? `${this.apiBaseUrl}/api/v1/chat/stream` 
          : `${this.apiBaseUrl}/api/v1/chat`;

        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream, application/json'
          },
          body: JSON.stringify(requestPayload),
          signal: this.abortController.signal,
          mode: 'cors'
        });

        // Hide typing indicator wave once first packet arrives
        this.dom.typingIndicator.classList.add('hidden');

        if (!response.ok) {
          const errDetail = await response.text();
          throw new Error(`HTTP ${response.status}: ${errDetail || response.statusText}`);
        }

        // 6. Handle Streaming vs Synchronous Response
        const isSseResponse = response.headers.get('content-type')?.includes('text/event-stream');

        if (isSseResponse && response.body) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder('utf-8');
          let accumulatedText = '';
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // save incomplete trailing line

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed || trimmed.startsWith(':')) continue; // Skip SSE comments/keepalives

              if (trimmed.startsWith('data:')) {
                const rawData = trimmed.slice(5).trim();
                if (rawData === '[DONE]') break;

                try {
                  const parsed = JSON.parse(rawData);
                  const chunk = parsed.token || parsed.message || parsed.content || '';
                  accumulatedText += chunk;
                  tokenCount++;
                } catch {
                  // Fallback for raw string token streams
                  accumulatedText += rawData;
                  tokenCount++;
                }
              } else {
                accumulatedText += line;
                tokenCount++;
              }

              assistantMsg.content = accumulatedText;
              bubble.innerHTML = MarkdownEngine.render(accumulatedText) + '<span class="streaming-cursor"></span>';
              this.scrollToBottom();
            }
          }

          // Final flush
          assistantMsg.content = accumulatedText || 'Respuesta completada.';
          bubble.innerHTML = MarkdownEngine.render(assistantMsg.content);

        } else {
          // Standard JSON Synchronous payload
          const data = await response.json();
          const replyText = data.message || 'Respuesta completada.';
          assistantMsg.content = replyText;
          assistantMsg.metadata = data.metadata || {};
          bubble.innerHTML = MarkdownEngine.render(replyText);
        }

        const totalLatency = Math.round(performance.now() - startTime);
        assistantMsg.metadata.latency_ms = totalLatency;
        if (tokenCount > 0) assistantMsg.metadata.tokens_used = tokenCount;

        if (toolbar) toolbar.classList.remove('hidden');
        if (stats) {
          stats.textContent = `${totalLatency}ms${tokenCount ? ` • ~${tokenCount} tokens` : ''}`;
        }

      } catch (err) {
        this.dom.typingIndicator.classList.add('hidden');
        
        if (err.name === 'AbortError') {
          assistantMsg.content = assistantMsg.content || '*Generación detenida por el usuario.*';
          bubble.innerHTML = MarkdownEngine.render(assistantMsg.content);
        } else {
          const errorMsg = `⚠️ **Error de conexión con el Gateway de IA**:\n\`${err.message}\`\n\nVerifica que el servicio esté corriendo en \`${this.apiBaseUrl}\` o revisa la configuración en el icono ⚙️ de ajustes.`;
          assistantMsg.content = errorMsg;
          bubble.innerHTML = MarkdownEngine.render(errorMsg);

          // Add a retry button in the bubble
          const retryContainer = document.createElement('div');
          retryContainer.style.marginTop = '10px';
          retryContainer.innerHTML = `
            <button type="button" class="btn-secondary btn-retry-message" style="font-size: 12px; padding: 4px 10px;">
              🔄 Reintentar solicitud
            </button>
          `;
          bubble.appendChild(retryContainer);
          this.showToast('Error al obtener respuesta del agente', 'error');
        }
      } finally {
        this.isStreaming = false;
        this.abortController = null;
        this.dom.btnAbortStream.classList.add('hidden');
        this.dom.btnSend.disabled = !this.dom.messageInput.value.trim();
        this._saveStoredMessages();
        this.scrollToBottom();
      }
    }

    // --- Conversation Actions ---
    clearConversation() {
      this.messages = [];
      this._generateNewSessionId();
      this._renderMessageFeed();
      this._saveStoredMessages();
    }

    exportConversation() {
      if (this.messages.length === 0) {
        this.showToast('No hay mensajes para exportar', 'info');
        return;
      }

      const agent = AGENT_CATALOG[this.activeAgent] || AGENT_CATALOG.portfolio;
      let mdContent = `# Transcripción de Chat - AI Agent Gateway\n\n`;
      mdContent += `- **Fecha:** ${new Date().toLocaleString()}\n`;
      mdContent += `- **Agente:** ${agent.name} (${this.activeAgent})\n`;
      mdContent += `- **Session ID:** \`${this.sessionId}\`\n\n---\n\n`;

      this.messages.forEach(msg => {
        const time = new Date(msg.timestamp).toLocaleTimeString();
        if (msg.role === 'user') {
          mdContent += `### 👤 Usuario (${time})\n\n${msg.content}\n\n`;
        } else if (msg.role === 'assistant') {
          mdContent += `### 🤖 ${agent.name} (${time})\n\n${msg.content}\n\n`;
        }
      });

      const blob = new Blob([mdContent], { type: 'text/markdown;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `chat_export_${this.activeAgent}_${Date.now()}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      this.showToast('Conversación exportada como Markdown (.md)', 'success');
    }

    // --- Settings Modal Actions ---
    _openSettingsModal() {
      const { dom } = this;
      dom.inputApiUrl.value = this.apiBaseUrl;
      dom.inputSessionId.value = this.sessionId;

      dom.radioStreamModes.forEach(radio => {
        radio.checked = radio.value === this.streamMode;
      });

      dom.settingsModal.classList.remove('hidden');
    }

    _saveSettings() {
      const { dom } = this;
      const newUrl = dom.inputApiUrl.value.trim().replace(/\/$/, '') || 'http://localhost:8000';
      this.apiBaseUrl = newUrl;
      localStorage.setItem('ai_gateway_api_url', newUrl);

      dom.radioStreamModes.forEach(radio => {
        if (radio.checked) {
          this.streamMode = radio.value;
          localStorage.setItem('ai_gateway_stream_mode', radio.value);
        }
      });

      dom.settingsModal.classList.add('hidden');
      this._checkHealth();
      this.showToast('Configuración guardada exitosamente', 'success');
    }

    // --- Toast Notification Manager ---
    showToast(message, type = 'info') {
      const toast = document.createElement('div');
      toast.className = `toast ${type}`;
      
      const icons = {
        success: '✓',
        error: '✕',
        info: 'ℹ'
      };

      toast.innerHTML = `
        <span style="font-weight: 700;">${icons[type] || '•'}</span>
        <span>${MarkdownEngine.escapeHtml(message)}</span>
      `;

      this.dom.toastContainer.appendChild(toast);

      setTimeout(() => {
        toast.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(8px)';
        setTimeout(() => toast.remove(), 250);
      }, 3200);
    }
  }

  // --- 4. Bootstrap on DOM Ready ---
  document.addEventListener('DOMContentLoaded', () => {
    window.App = new ChatbotApp();
  });

})();
