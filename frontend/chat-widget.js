/**
 * AI Agent Gateway - Universal Drop-in Chat Widget
 * ============================================================================
 * Architecture: Zero-Dependencies Vanilla JavaScript Web Component (Shadow DOM)
 * Embeddable into Django Monolith, external websites, or standalone portals.
 *
 * Usage:
 *   <script src="chat-widget.js" data-api-url="http://localhost:8000" data-agent="portfolio"></script>
 * Or:
 *   <ai-chat-widget api-url="http://localhost:8000" default-agent="portfolio"></ai-chat-widget>
 * Or JavaScript API:
 *   window.AiChatWidget.init({ apiUrl: 'http://localhost:8000', defaultAgent: 'portfolio' });
 * ============================================================================
 */

(function () {
  'use strict';

  // --- 1. Agent Metadata & Prompt Starters ---
  const AGENTS = {
    portfolio: {
      id: 'portfolio',
      name: 'Portfolio Agent',
      icon: '💼',
      role: 'Proyectos & CV',
      desc: 'Experiencia profesional, arquitectura, proyectos y stack técnico',
      starters: [
        '¿Cuáles son los proyectos más destacados?',
        '¿Qué stack tecnológico y herramientas dominas?',
        '¿Cómo está estructurado este microservicio de IA?',
        'Explica la experiencia en arquitecturas escalables'
      ]
    },
    ecommerce: {
      id: 'ecommerce',
      name: 'E-commerce Agent',
      icon: '🛍️',
      role: 'Catálogo & Ventas',
      desc: 'Búsqueda de productos, recomendaciones personalizadas y cotizaciones',
      starters: [
        '¿Qué productos o servicios recomendados tienes?',
        'Cotizar un desarrollo de software a medida',
        '¿Cuáles son los métodos de pago y tiempos de entrega?',
        'Recomiéndame una solución para comercio electrónico'
      ]
    },
    analytics: {
      id: 'analytics',
      name: 'Analytics Agent',
      icon: '📊',
      role: 'Métricas & KPIs',
      desc: 'Métricas de rendimiento, análisis de datos e informes ejecutivos',
      starters: [
        'Resumen de métricas de rendimiento del microservicio',
        '¿Cómo se optimiza la latencia en streaming de LLM?',
        'Generar informe sobre uso de tokens y tasa de aciertos en caché',
        'Explicar el pipeline de observabilidad y métricas'
      ]
    },
    general: {
      id: 'general',
      name: 'General Assistant',
      icon: '⚡',
      role: 'Asistente Multi-Agente',
      desc: 'Orquestación general, consultas abiertas y asistencia integral',
      starters: [
        '¿Qué agentes están disponibles y qué función cumple cada uno?',
        '¿Cómo funciona la memoria de conversación con Redis?',
        'Ayúdame a redactar un resumen técnico del gateway',
        '¿Cómo integrar este widget en un monolito Django?'
      ]
    }
  };

  // --- 2. Lightweight Zero-Dependency Markdown Parser ---
  class MarkdownRenderer {
    static escapeHtml(str) {
      return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    static render(text) {
      if (!text) return '';

      // Normalize newlines
      let content = text.replace(/\r\n/g, '\n');

      // 1. Extract and protect code blocks
      const codeBlocks = [];
      content = content.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
        const id = `__CODE_BLOCK_${codeBlocks.length}__`;
        codeBlocks.push({ lang: lang.trim() || 'plaintext', code });
        return id;
      });

      // 2. Protect inline code
      const inlineCodes = [];
      content = content.replace(/`([^`\n]+)`/g, (match, code) => {
        const id = `__INLINE_CODE_${inlineCodes.length}__`;
        inlineCodes.push(this.escapeHtml(code));
        return id;
      });

      // 3. Escape general HTML in the rest of content
      content = this.escapeHtml(content);

      // 4. Headers
      content = content.replace(/^### (.*$)/gim, '<h3>$1</h3>');
      content = content.replace(/^## (.*$)/gim, '<h2>$1</h2>');
      content = content.replace(/^# (.*$)/gim, '<h1>$1</h1>');

      // 5. Blockquotes
      content = content.replace(/^\> (.*$)/gim, '<blockquote>$1</blockquote>');

      // 6. Bold & Italics & Strikethrough
      content = content.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      content = content.replace(/__([^_]+)__/g, '<strong>$1</strong>');
      content = content.replace(/\*([^*]+)\*/g, '<em>$1</em>');
      content = content.replace(/_([^_]+)_/g, '<em>$1</em>');
      content = content.replace(/~~([^~]+)~~/g, '<del>$1</del>');

      // 7. Links
      content = content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

      // 8. Horizontal rules
      content = content.replace(/^---$/gim, '<hr>');

      // 9. Tables
      content = content.replace(/((?:\|[^\n]+\|\r?\n)+)/g, (match) => {
        const lines = match.trim().split('\n').filter(l => l.trim().length > 0);
        if (lines.length < 2) return match;
        
        let html = '<div class="table-wrapper"><table>';
        let isHeader = true;

        lines.forEach((line, idx) => {
          if (line.includes('---')) return; // Separator row
          const cells = line.split('|').map(c => c.trim()).filter((c, i, arr) => i > 0 && i < arr.length - 1);
          
          if (idx === 0) {
            html += '<thead><tr>';
            cells.forEach(c => html += `<th>${c}</th>`);
            html += '</tr></thead><tbody>';
          } else {
            html += '<tr>';
            cells.forEach(c => html += `<td>${c}</td>`);
            html += '</tr>';
          }
        });

        html += '</tbody></table></div>';
        return html;
      });

      // 10. Unordered & Ordered Lists
      content = content.replace(/((?:^(?:[\*\-\+]|\d+\.) .*(?:\n|$))+)/gim, (match) => {
        const isOrdered = /^\d+\./.test(match.trim());
        const tag = isOrdered ? 'ol' : 'ul';
        const items = match.trim().split('\n').map(item => {
          const itemText = item.replace(/^(?:[\*\-\+]|\d+\.)\s+/, '');
          return `<li>${itemText}</li>`;
        }).join('');
        return `<${tag}>${items}</${tag}>`;
      });

      // 11. Paragraphs (lines separated by double newlines)
      const paragraphs = content.split(/\n{2,}/);
      content = paragraphs.map(p => {
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
        content = content.replace(`__INLINE_CODE_${idx}__`, `<code>${code}</code>`);
      });

      // 13. Restore code blocks with Header & Copy button
      codeBlocks.forEach((block, idx) => {
        const escapedCode = this.escapeHtml(block.code);
        const codeBlockHtml = `
          <div class="code-block-container" data-code-index="${idx}">
            <div class="code-header">
              <span class="code-language">${block.lang}</span>
              <button type="button" class="btn-copy-code" data-code="${this.escapeHtml(block.code)}">
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
                <span class="copy-text">Copiar</span>
              </button>
            </div>
            <pre><code class="language-${block.lang}">${escapedCode}</code></pre>
          </div>
        `;
        content = content.replace(`__CODE_BLOCK_${idx}__`, codeBlockHtml);
      });

      return content;
    }
  }

  // --- 3. Scoped Styles (Injected into Shadow DOM) ---
  const WIDGET_STYLES = `
    *, *::before, *::after {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    :host {
      --fab-size: 56px;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --accent-glow: rgba(59, 130, 246, 0.35);
      --accent-subtle: rgba(59, 130, 246, 0.12);
      --bg-surface: #0f172a;
      --bg-panel: #111827;
      --bg-card: #1e293b;
      --bg-card-hover: #27354f;
      --bg-bubble-user: #1d4ed8;
      --bg-bubble-ai: #172033;
      --border-subtle: #1e293b;
      --border-medium: #334155;
      --border-focus: #3b82f6;
      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --font-sans: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', Consolas, Monaco, monospace;
      --radius-sm: 6px;
      --radius-md: 10px;
      --radius-lg: 16px;
      --radius-full: 9999px;
      --shadow-fab: 0 8px 24px rgba(0, 0, 0, 0.4), 0 2px 6px rgba(59, 130, 246, 0.3);
      --shadow-window: 0 16px 40px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.08);
      --status-online: #10b981;
      --status-checking: #f59e0b;
      --status-offline: #ef4444;
      font-family: var(--font-sans);
      font-size: 14px;
      line-height: 1.5;
      color: var(--text-primary);
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 999999;
      pointer-events: none;
    }

    .widget-root {
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      pointer-events: auto;
    }

    /* Floating Action Button (FAB Launcher) */
    .widget-fab {
      width: var(--fab-size);
      height: var(--fab-size);
      border-radius: 50%;
      background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      box-shadow: var(--shadow-fab);
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
      position: relative;
      user-select: none;
    }

    .widget-fab:hover {
      transform: scale(1.06) translateY(-2px);
      box-shadow: 0 12px 28px rgba(0, 0, 0, 0.5), 0 4px 12px var(--accent-glow);
    }

    .widget-fab:active {
      transform: scale(0.96);
    }

    .fab-icon-chat, .fab-icon-close {
      position: absolute;
      transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .fab-icon-chat {
      opacity: 1;
      transform: scale(1) rotate(0deg);
    }

    .fab-icon-close {
      opacity: 0;
      transform: scale(0.6) rotate(-90deg);
    }

    .widget-fab.open .fab-icon-chat {
      opacity: 0;
      transform: scale(0.6) rotate(90deg);
    }

    .widget-fab.open .fab-icon-close {
      opacity: 1;
      transform: scale(1) rotate(0deg);
    }

    /* Unread Indicator Pulse */
    .fab-pulse-badge {
      position: absolute;
      top: -2px;
      right: -2px;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background-color: var(--status-online);
      border: 2px solid var(--bg-surface);
      box-shadow: 0 0 8px rgba(16, 185, 129, 0.8);
    }

    /* Chat Floating Window */
    .widget-window {
      position: absolute;
      bottom: calc(var(--fab-size) + 16px);
      right: 0;
      width: 420px;
      height: 640px;
      max-width: calc(100vw - 32px);
      max-height: calc(100vh - 100px);
      background-color: var(--bg-panel);
      border: 1px solid var(--border-medium);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-window);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      opacity: 0;
      transform: scale(0.92) translateY(20px);
      pointer-events: none;
      transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      transform-origin: bottom right;
    }

    .widget-window.visible {
      opacity: 1;
      transform: scale(1) translateY(0);
      pointer-events: auto;
    }

    /* Window Header */
    .window-header {
      background-color: var(--bg-surface);
      border-bottom: 1px solid var(--border-subtle);
      padding: 12px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-shrink: 0;
    }

    .header-agent-info {
      display: flex;
      align-items: center;
      gap: 10px;
      cursor: pointer;
      padding: 4px 8px;
      border-radius: var(--radius-md);
      transition: background-color 0.15s ease;
      position: relative;
    }

    .header-agent-info:hover {
      background-color: var(--bg-card);
    }

    .agent-avatar-badge {
      width: 32px;
      height: 32px;
      border-radius: 50%;
      background-color: var(--accent-subtle);
      border: 1px solid rgba(59, 130, 246, 0.3);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      flex-shrink: 0;
    }

    .agent-title-box {
      display: flex;
      flex-direction: column;
      text-align: left;
    }

    .agent-title-row {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .agent-title {
      font-size: 13.5px;
      font-weight: 600;
      color: var(--text-primary);
    }

    .agent-chevron {
      color: var(--text-muted);
      transition: transform 0.15s ease;
    }

    .header-agent-info.dropdown-open .agent-chevron {
      transform: rotate(180deg);
    }

    .agent-subrole {
      font-size: 11px;
      color: var(--text-secondary);
    }

    .header-controls {
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .status-indicator-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: 4px;
      transition: background-color 0.2s ease;
    }

    .status-indicator-dot.online {
      background-color: var(--status-online);
      box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);
    }

    .status-indicator-dot.checking {
      background-color: var(--status-checking);
      animation: statusPulse 1.2s infinite ease-in-out;
    }

    .status-indicator-dot.offline {
      background-color: var(--status-offline);
      box-shadow: 0 0 6px rgba(239, 68, 68, 0.6);
    }

    @keyframes statusPulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.3; transform: scale(0.8); }
    }

    .btn-header-action {
      background: transparent;
      border: none;
      color: var(--text-secondary);
      width: 28px;
      height: 28px;
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.15s ease;
    }

    .btn-header-action:hover {
      background-color: var(--bg-card);
      color: var(--text-primary);
    }

    /* Agent Dropdown Menu inside Window */
    .agent-dropdown-menu {
      position: absolute;
      top: 56px;
      left: 12px;
      right: 12px;
      background-color: var(--bg-surface);
      border: 1px solid var(--border-medium);
      border-radius: var(--radius-md);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
      padding: 6px;
      z-index: 100;
      display: none;
      animation: menuFadeIn 0.15s ease-out;
    }

    .agent-dropdown-menu.open {
      display: block;
    }

    @keyframes menuFadeIn {
      from { opacity: 0; transform: translateY(-6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .agent-menu-item {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      border-radius: var(--radius-sm);
      cursor: pointer;
      transition: background-color 0.15s ease;
    }

    .agent-menu-item:hover {
      background-color: var(--bg-card);
    }

    .agent-menu-item.active {
      background-color: var(--accent-subtle);
    }

    .agent-menu-icon {
      font-size: 18px;
    }

    .agent-menu-content {
      display: flex;
      flex-direction: column;
    }

    .agent-menu-name {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-primary);
    }

    .agent-menu-desc {
      font-size: 11px;
      color: var(--text-secondary);
    }

    /* Conversation Body Container */
    .window-body {
      flex: 1;
      overflow-y: auto;
      overflow-x: hidden;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      scroll-behavior: smooth;
    }

    .window-body::-webkit-scrollbar {
      width: 5px;
    }
    .window-body::-webkit-scrollbar-track {
      background: transparent;
    }
    .window-body::-webkit-scrollbar-thumb {
      background: var(--border-medium);
      border-radius: 4px;
    }

    /* Welcome Hero inside Widget */
    .widget-welcome {
      text-align: center;
      padding: 16px 8px;
      animation: fadeIn 0.3s ease-out;
    }

    .welcome-icon-box {
      width: 50px;
      height: 50px;
      border-radius: 50%;
      background-color: var(--accent-subtle);
      border: 1px solid rgba(59, 130, 246, 0.3);
      font-size: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 12px;
    }

    .welcome-heading {
      font-size: 16px;
      font-weight: 700;
      color: var(--text-primary);
      margin-bottom: 6px;
    }

    .welcome-text {
      font-size: 12.5px;
      color: var(--text-secondary);
      line-height: 1.5;
      margin-bottom: 16px;
    }

    .widget-starters-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .widget-starter-btn {
      background-color: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: 8px 12px;
      color: var(--text-secondary);
      font-size: 12px;
      text-align: left;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .widget-starter-btn:hover {
      background-color: var(--bg-card-hover);
      border-color: var(--border-focus);
      color: var(--text-primary);
      transform: translateX(2px);
    }

    .starter-bullet {
      color: var(--accent);
    }

    /* Message Rows */
    .msg-row {
      display: flex;
      gap: 8px;
      width: 100%;
      animation: msgIn 0.2s ease-out;
    }

    @keyframes msgIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .msg-row.user {
      justify-content: flex-end;
    }

    .msg-row.user .msg-bubble {
      background-color: var(--bg-bubble-user);
      color: #ffffff;
      border-radius: var(--radius-md) var(--radius-md) 2px var(--radius-md);
      max-width: 82%;
      padding: 10px 14px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
      font-size: 13.5px;
      line-height: 1.45;
    }

    .msg-row.assistant {
      justify-content: flex-start;
    }

    .msg-avatar {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      background-color: var(--bg-card);
      border: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 14px;
      flex-shrink: 0;
      margin-top: 2px;
    }

    .msg-content-wrap {
      display: flex;
      flex-direction: column;
      max-width: calc(100% - 36px);
    }

    .msg-header-meta {
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 3px;
      font-size: 11px;
      color: var(--text-muted);
    }

    .msg-header-meta .agent-tag {
      font-weight: 600;
      color: var(--text-primary);
    }

    .msg-row.assistant .msg-bubble {
      background-color: var(--bg-bubble-ai);
      border: 1px solid var(--border-subtle);
      border-radius: 2px var(--radius-md) var(--radius-md) var(--radius-md);
      padding: 12px 14px;
      color: var(--text-primary);
      font-size: 13px;
      line-height: 1.5;
      word-break: break-word;
    }

    .msg-row.system {
      justify-content: center;
    }

    .msg-row.system .msg-bubble {
      background-color: var(--bg-card);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-full);
      padding: 4px 12px;
      font-size: 11px;
      color: var(--text-secondary);
    }

    /* Markdown inside Widget */
    .msg-bubble p { margin-bottom: 8px; }
    .msg-bubble p:last-child { margin-bottom: 0; }
    .msg-bubble h1, .msg-bubble h2, .msg-bubble h3 {
      font-weight: 600;
      margin: 10px 0 6px;
      line-height: 1.3;
    }
    .msg-bubble h1 { font-size: 16px; }
    .msg-bubble h2 { font-size: 14.5px; }
    .msg-bubble h3 { font-size: 13px; }
    .msg-bubble ul, .msg-bubble ol { margin: 6px 0 8px 18px; }
    .msg-bubble li { margin-bottom: 3px; }
    .msg-bubble blockquote {
      border-left: 3px solid var(--accent);
      background-color: var(--accent-subtle);
      padding: 6px 10px;
      margin: 8px 0;
      font-style: italic;
      color: var(--text-secondary);
    }
    .msg-bubble a {
      color: #60a5fa;
      text-decoration: underline;
    }
    .msg-bubble code:not(pre code) {
      font-family: var(--font-mono);
      font-size: 11.5px;
      background-color: #080c14;
      color: #38bdf8;
      padding: 2px 4px;
      border-radius: 4px;
    }

    /* Code block container */
    .code-block-container {
      margin: 10px 0;
      border-radius: var(--radius-sm);
      border: 1px solid var(--border-medium);
      background-color: #080c14;
      overflow: hidden;
    }

    .code-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background-color: #0d121d;
      padding: 4px 8px;
      border-bottom: 1px solid var(--border-subtle);
    }

    .code-language {
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-secondary);
      text-transform: lowercase;
    }

    .btn-copy-code {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-size: 10px;
      font-family: var(--font-sans);
      cursor: pointer;
      padding: 2px 5px;
      border-radius: 3px;
    }

    .btn-copy-code:hover {
      background-color: var(--bg-card);
      color: var(--text-primary);
    }

    .code-block-container pre {
      margin: 0;
      padding: 8px 10px;
      overflow-x: auto;
    }

    .code-block-container code {
      font-family: var(--font-mono);
      font-size: 11.5px;
      color: #f1f5f9;
      line-height: 1.45;
      white-space: pre;
    }

    .table-wrapper {
      overflow-x: auto;
      margin: 8px 0;
    }

    .msg-bubble table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }

    .msg-bubble th, .msg-bubble td {
      border: 1px solid var(--border-medium);
      padding: 6px 8px;
      text-align: left;
    }

    .msg-bubble th {
      background-color: var(--bg-card);
    }

    /* Streaming Cursor */
    .cursor-caret {
      display: inline-block;
      width: 5px;
      height: 12px;
      background-color: var(--accent);
      margin-left: 2px;
      vertical-align: middle;
      animation: caretBlink 0.8s infinite;
    }

    @keyframes caretBlink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0; }
    }

    /* Typing Dots */
    .typing-wave {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 8px 12px;
      background-color: var(--bg-bubble-ai);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      width: fit-content;
    }

    .typing-wave-dot {
      width: 5px;
      height: 5px;
      background-color: var(--accent);
      border-radius: 50%;
      animation: typingWave 1.2s infinite ease-in-out;
    }

    .typing-wave-dot:nth-child(1) { animation-delay: 0s; }
    .typing-wave-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-wave-dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes typingWave {
      0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
      40% { transform: translateY(-4px); opacity: 1; }
    }

    /* Input Footer Area */
    .window-footer {
      background-color: var(--bg-surface);
      border-top: 1px solid var(--border-subtle);
      padding: 10px 14px 12px;
      flex-shrink: 0;
    }

    .input-box {
      display: flex;
      align-items: flex-end;
      gap: 8px;
      background-color: var(--bg-panel);
      border: 1px solid var(--border-medium);
      border-radius: var(--radius-md);
      padding: 6px 8px 6px 12px;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .input-box:focus-within {
      border-color: var(--border-focus);
      box-shadow: 0 0 0 2px var(--accent-subtle);
    }

    .widget-textarea {
      flex: 1;
      min-height: 24px;
      max-height: 120px;
      background: transparent;
      border: none;
      outline: none;
      color: var(--text-primary);
      font-family: var(--font-sans);
      font-size: 13.5px;
      line-height: 1.4;
      resize: none;
    }

    .widget-textarea::placeholder {
      color: var(--text-muted);
    }

    .btn-send-widget {
      width: 32px;
      height: 32px;
      border-radius: var(--radius-sm);
      background-color: var(--accent);
      border: none;
      color: #ffffff;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.15s ease;
      flex-shrink: 0;
    }

    .btn-send-widget:hover:not(:disabled) {
      background-color: var(--accent-hover);
    }

    .btn-send-widget:disabled {
      background-color: var(--border-medium);
      color: var(--text-muted);
      cursor: not-allowed;
      opacity: 0.5;
    }

    .btn-abort-widget {
      background-color: rgba(239, 68, 68, 0.15);
      border: 1px solid var(--status-offline);
      color: var(--status-offline);
      border-radius: var(--radius-sm);
      padding: 4px 8px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
    }

    .footer-credits {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 10px;
      color: var(--text-muted);
      margin-top: 6px;
      padding: 0 2px;
    }

    .session-indicator {
      font-family: var(--font-mono);
      cursor: pointer;
    }

    .session-indicator:hover {
      color: var(--text-secondary);
    }

    /* Mobile Responsive Adaptation */
    @media (max-width: 480px) {
      :host {
        bottom: 16px;
        right: 16px;
      }
      .widget-window {
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        max-width: 100vw;
        max-height: 100vh;
        border-radius: 0;
        border: none;
      }
    }
  `;

  // --- 4. Custom Element: <ai-chat-widget> ---
  class AiChatWidgetElement extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });

      // Internal State
      this.apiUrl = this.getAttribute('api-url') || 'http://localhost:8000';
      this.activeAgent = this.getAttribute('default-agent') || 'portfolio';
      this.isOpen = false;
      this.isStreaming = false;
      this.isConnected = false;
      this.messages = [];
      this.abortController = null;
      this.sessionId = this._getOrCreateSessionId();

      // Health ping timer
      this._pingTimer = null;
    }

    static get observedAttributes() {
      return ['api-url', 'default-agent'];
    }

    attributeChangedCallback(name, oldValue, newValue) {
      if (oldValue === newValue) return;
      if (name === 'api-url') this.apiUrl = newValue;
      if (name === 'default-agent' && AGENTS[newValue]) {
        this.activeAgent = newValue;
        this._updateAgentDisplay();
      }
    }

    connectedCallback() {
      this._render();
      this._bindEvents();
      this._checkHealth();
      this._loadSessionMessages();

      // Recurring health check every 30s
      this._pingTimer = setInterval(() => this._checkHealth(), 30000);
    }

    disconnectedCallback() {
      if (this._pingTimer) clearInterval(this._pingTimer);
    }

    // --- Session Management (sessionStorage) ---
    _getOrCreateSessionId() {
      const STORAGE_KEY = 'ai_gateway_session_id';
      let id = sessionStorage.getItem(STORAGE_KEY);
      if (!id) {
        id = 'sess_' + Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
        sessionStorage.setItem(STORAGE_KEY, id);
      }
      return id;
    }

    _loadSessionMessages() {
      try {
        const key = `ai_chat_history_${this.sessionId}`;
        const saved = sessionStorage.getItem(key);
        if (saved) {
          this.messages = JSON.parse(saved);
          this._renderMessagesFeed();
        }
      } catch (e) {
        console.warn('Could not load session history:', e);
      }
    }

    _saveSessionMessages() {
      try {
        const key = `ai_chat_history_${this.sessionId}`;
        sessionStorage.setItem(key, JSON.stringify(this.messages));
      } catch (e) {
        console.warn('Could not save session history:', e);
      }
    }

    // --- Health Check ---
    async _checkHealth() {
      const dot = this.shadowRoot.querySelector('.status-indicator-dot');
      if (dot) {
        dot.className = 'status-indicator-dot checking';
        dot.title = 'Comprobando conexión con el Gateway...';
      }

      try {
        const res = await fetch(`${this.apiUrl}/health`, { method: 'GET', mode: 'cors' });
        if (res.ok) {
          const data = await res.json();
          this.isConnected = true;
          if (dot) {
            dot.className = 'status-indicator-dot online';
            dot.title = `Conectado: ${data.app_name || 'AI Gateway'} (${data.version || 'v0.1.0'})`;
          }
        } else {
          throw new Error(`HTTP ${res.status}`);
        }
      } catch (err) {
        this.isConnected = false;
        if (dot) {
          dot.className = 'status-indicator-dot offline';
          dot.title = 'Sin conexión con el Gateway de IA (' + this.apiUrl + ')';
        }
      }
    }

    // --- DOM Structure Render ---
    _render() {
      const agent = AGENTS[this.activeAgent] || AGENTS.portfolio;

      this.shadowRoot.innerHTML = `
        <style>${WIDGET_STYLES}</style>
        
        <div class="widget-root">
          <!-- Floating Window -->
          <div class="widget-window" id="widget-window">
            
            <!-- Window Header -->
            <div class="window-header">
              <div class="header-agent-info" id="agent-selector-btn" title="Cambiar agente de IA">
                <div class="agent-avatar-badge" id="header-avatar">${agent.icon}</div>
                <div class="agent-title-box">
                  <div class="agent-title-row">
                    <span class="agent-title" id="header-agent-name">${agent.name}</span>
                    <svg class="agent-chevron" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="m6 9 6 6 6-6"/>
                    </svg>
                  </div>
                  <span class="agent-subrole" id="header-agent-role">${agent.role}</span>
                </div>
              </div>

              <!-- Agent Selector Dropdown inside window -->
              <div class="agent-dropdown-menu" id="agent-dropdown">
                ${Object.values(AGENTS).map(a => `
                  <div class="agent-menu-item ${a.id === this.activeAgent ? 'active' : ''}" data-agent-id="${a.id}">
                    <span class="agent-menu-icon">${a.icon}</span>
                    <div class="agent-menu-content">
                      <span class="agent-menu-name">${a.name}</span>
                      <span class="agent-menu-desc">${a.desc}</span>
                    </div>
                  </div>
                `).join('')}
              </div>

              <!-- Header Controls -->
              <div class="header-controls">
                <span class="status-indicator-dot checking" title="Verificando conexión"></span>
                <button type="button" class="btn-header-action" id="btn-clear-chat" title="Limpiar y Nueva Sesión">
                  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                  </svg>
                </button>
                <button type="button" class="btn-header-action" id="btn-minimize-window" title="Minimizar">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M18 6L6 18M6 6l12 12"/>
                  </svg>
                </button>
              </div>
            </div>

            <!-- Conversation Feed -->
            <div class="window-body" id="window-body">
              ${this._getWelcomeHtml(agent)}
            </div>

            <!-- Typing indicator wave -->
            <div class="typing-wave" id="typing-indicator" style="display: none; margin: 0 16px 8px;">
              <span class="typing-wave-dot"></span>
              <span class="typing-wave-dot"></span>
              <span class="typing-wave-dot"></span>
            </div>

            <!-- Input Footer -->
            <div class="window-footer">
              <div class="input-box">
                <textarea
                  class="widget-textarea"
                  id="widget-input"
                  placeholder="Escribe un mensaje... (Enter para enviar)"
                  rows="1"
                  maxlength="3000"
                ></textarea>
                
                <button type="button" class="btn-abort-widget" id="btn-abort" style="display: none;">
                  <svg viewBox="0 0 24 24" width="12" height="12" fill="currentColor">
                    <rect x="6" y="6" width="12" height="12" rx="2"/>
                  </svg>
                  <span>Stop</span>
                </button>

                <button type="button" class="btn-send-widget" id="btn-send" disabled>
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="m22 2-7 20-4-9-9-4Z"/>
                    <path d="M22 2 11 13"/>
                  </svg>
                </button>
              </div>

              <div class="footer-credits">
                <span>AI Gateway • Streaming SSE</span>
                <span class="session-indicator" id="footer-sess-btn" title="Session ID en sessionStorage">${this.sessionId.substring(0, 14)}...</span>
              </div>
            </div>

          </div>

          <!-- Floating Trigger Button (FAB) -->
          <div class="widget-fab" id="widget-fab" role="button" aria-label="Abrir asistente de IA">
            <span class="fab-pulse-badge" id="fab-pulse"></span>
            <svg class="fab-icon-chat" viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <svg class="fab-icon-close" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 6L6 18M6 6l12 12"/>
            </svg>
          </div>
        </div>
      `;
    }

    _getWelcomeHtml(agent) {
      return `
        <div class="widget-welcome" id="widget-welcome">
          <div class="welcome-icon-box">${agent.icon}</div>
          <h3 class="welcome-heading">${agent.name}</h3>
          <p class="welcome-text">${agent.desc}</p>
          <div class="widget-starters-list">
            ${agent.starters.map(st => `
              <button type="button" class="widget-starter-btn" data-starter="${MarkdownRenderer.escapeHtml(st)}">
                <span class="starter-bullet">›</span>
                <span>${st}</span>
              </button>
            `).join('')}
          </div>
        </div>
      `;
    }

    // --- Event Binding ---
    _bindEvents() {
      const root = this.shadowRoot;
      const fab = root.querySelector('#widget-fab');
      const windowEl = root.querySelector('#widget-window');
      const minimizeBtn = root.querySelector('#btn-minimize-window');
      const agentBtn = root.querySelector('#agent-selector-btn');
      const agentDropdown = root.querySelector('#agent-dropdown');
      const clearBtn = root.querySelector('#btn-clear-chat');
      const input = root.querySelector('#widget-input');
      const sendBtn = root.querySelector('#btn-send');
      const abortBtn = root.querySelector('#btn-abort');
      const body = root.querySelector('#window-body');

      // FAB Toggle
      fab.addEventListener('click', () => {
        this.isOpen = !this.isOpen;
        fab.classList.toggle('open', this.isOpen);
        windowEl.classList.toggle('visible', this.isOpen);
        if (this.isOpen) {
          root.querySelector('#fab-pulse').style.display = 'none';
          setTimeout(() => input.focus(), 150);
          this._scrollToBottom();
        }
      });

      // Minimize
      minimizeBtn.addEventListener('click', () => {
        this.isOpen = false;
        fab.classList.remove('open');
        windowEl.classList.remove('visible');
      });

      // Agent Selector dropdown
      agentBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        agentDropdown.classList.toggle('open');
        agentBtn.classList.toggle('dropdown-open');
      });

      // Click outside agent dropdown
      root.addEventListener('click', (e) => {
        if (!agentBtn.contains(e.target) && !agentDropdown.contains(e.target)) {
          agentDropdown.classList.remove('open');
          agentBtn.classList.remove('dropdown-open');
        }
      });

      // Select Agent
      agentDropdown.querySelectorAll('.agent-menu-item').forEach(item => {
        item.addEventListener('click', () => {
          const newAgentId = item.dataset.agentId;
          this.setAgent(newAgentId);
          agentDropdown.classList.remove('open');
          agentBtn.classList.remove('dropdown-open');
        });
      });

      // Clear chat
      clearBtn.addEventListener('click', () => {
        if (confirm('¿Deseas reiniciar la conversación y generar una nueva sesión?')) {
          this.clearSession();
        }
      });

      // Starter chip clicks
      body.addEventListener('click', (e) => {
        const starterBtn = e.target.closest('.widget-starter-btn');
        if (starterBtn) {
          const text = starterBtn.dataset.starter;
          if (text) {
            input.value = text;
            this._handleSend();
          }
        }

        // Copy code buttons inside message body
        const copyBtn = e.target.closest('.btn-copy-code');
        if (copyBtn) {
          const code = copyBtn.dataset.code;
          if (code) {
            navigator.clipboard.writeText(code).then(() => {
              const label = copyBtn.querySelector('.copy-text');
              if (label) {
                const old = label.textContent;
                label.textContent = '¡Copiado!';
                setTimeout(() => label.textContent = old, 2000);
              }
            });
          }
        }
      });

      // Auto-resize textarea
      input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
        sendBtn.disabled = !input.value.trim() || this.isStreaming;
      });

      // Enter key to send (Shift+Enter for newline)
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          if (!sendBtn.disabled) {
            this._handleSend();
          }
        }
      });

      // Send button
      sendBtn.addEventListener('click', () => this._handleSend());

      // Abort button
      abortBtn.addEventListener('click', () => {
        if (this.abortController) {
          this.abortController.abort();
          this.abortController = null;
        }
      });
    }

    _updateAgentDisplay() {
      const agent = AGENTS[this.activeAgent] || AGENTS.portfolio;
      const root = this.shadowRoot;
      root.querySelector('#header-avatar').textContent = agent.icon;
      root.querySelector('#header-agent-name').textContent = agent.name;
      root.querySelector('#header-agent-role').textContent = agent.role;

      // Update active in dropdown
      root.querySelectorAll('.agent-menu-item').forEach(el => {
        el.classList.toggle('active', el.dataset.agentId === this.activeAgent);
      });

      // If no messages yet, update welcome hero
      if (this.messages.length === 0) {
        root.querySelector('#window-body').innerHTML = this._getWelcomeHtml(agent);
      }
    }

    _scrollToBottom() {
      const body = this.shadowRoot.querySelector('#window-body');
      if (body) {
        body.scrollTop = body.scrollHeight;
      }
    }

    // --- Message Rendering Feed ---
    _renderMessagesFeed() {
      const body = this.shadowRoot.querySelector('#window-body');
      if (this.messages.length === 0) {
        body.innerHTML = this._getWelcomeHtml(AGENTS[this.activeAgent] || AGENTS.portfolio);
        return;
      }

      body.innerHTML = '';
      this.messages.forEach(msg => {
        this._appendMessageToDOM(msg, false);
      });
      this._scrollToBottom();
    }

    _appendMessageToDOM(msg, shouldScroll = true) {
      const body = this.shadowRoot.querySelector('#window-body');
      const welcome = body.querySelector('#widget-welcome');
      if (welcome) welcome.remove();

      const row = document.createElement('div');
      row.className = `msg-row ${msg.role}`;
      row.id = `msg-${msg.id}`;

      if (msg.role === 'user') {
        row.innerHTML = `
          <div class="msg-bubble">${MarkdownRenderer.escapeHtml(msg.content)}</div>
        `;
      } else if (msg.role === 'assistant') {
        const agent = AGENTS[msg.agent_id || this.activeAgent] || AGENTS.portfolio;
        row.innerHTML = `
          <div class="msg-avatar">${agent.icon}</div>
          <div class="msg-content-wrap">
            <div class="msg-header-meta">
              <span class="agent-tag">${agent.name}</span>
              <span>${new Date(msg.timestamp || Date.now()).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
            </div>
            <div class="msg-bubble markdown-body">${MarkdownRenderer.render(msg.content)}</div>
          </div>
        `;
      } else {
        row.innerHTML = `
          <div class="msg-bubble">${MarkdownRenderer.escapeHtml(msg.content)}</div>
        `;
      }

      body.appendChild(row);
      if (shouldScroll) this._scrollToBottom();
    }

    // --- Send & SSE Streaming Pipeline ---
    async _handleSend() {
      const input = this.shadowRoot.querySelector('#widget-input');
      const text = input.value.trim();
      if (!text || this.isStreaming) return;

      input.value = '';
      input.style.height = 'auto';
      this.shadowRoot.querySelector('#btn-send').disabled = true;

      // 1. Add User Message
      const userMsg = {
        id: 'msg_' + Date.now(),
        role: 'user',
        content: text,
        timestamp: Date.now()
      };
      this.messages.push(userMsg);
      this._appendMessageToDOM(userMsg);
      this._saveSessionMessages();

      // 2. Prepare Assistant Bubble
      const assistantMsg = {
        id: 'msg_ai_' + Date.now(),
        role: 'assistant',
        agent_id: this.activeAgent,
        content: '',
        timestamp: Date.now()
      };
      this.messages.push(assistantMsg);

      // Create streaming placeholder DOM element
      const body = this.shadowRoot.querySelector('#window-body');
      const row = document.createElement('div');
      row.className = 'msg-row assistant';
      row.id = `msg-${assistantMsg.id}`;
      const agent = AGENTS[this.activeAgent] || AGENTS.portfolio;
      
      row.innerHTML = `
        <div class="msg-avatar">${agent.icon}</div>
        <div class="msg-content-wrap">
          <div class="msg-header-meta">
            <span class="agent-tag">${agent.name}</span>
            <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
          </div>
          <div class="msg-bubble markdown-body"><span class="cursor-caret"></span></div>
        </div>
      `;
      body.appendChild(row);
      this._scrollToBottom();

      const bubble = row.querySelector('.msg-bubble');
      const typingIndicator = this.shadowRoot.querySelector('#typing-indicator');
      const abortBtn = this.shadowRoot.querySelector('#btn-abort');

      this.isStreaming = true;
      typingIndicator.style.display = 'inline-flex';
      abortBtn.style.display = 'flex';
      this.abortController = new AbortController();

      const payload = {
        agent_id: this.activeAgent,
        session_id: this.sessionId,
        message: text,
        stream: true
      };

      try {
        const response = await fetch(`${this.apiUrl}/api/v1/chat/stream`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream, application/json'
          },
          body: JSON.stringify(payload),
          signal: this.abortController.signal
        });

        typingIndicator.style.display = 'none';

        if (!response.ok) {
          throw new Error(`Error en servidor: HTTP ${response.status}`);
        }

        // Check if response is stream
        if (response.body && response.headers.get('content-type')?.includes('text/event-stream')) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder('utf-8');
          let accumulatedText = '';
          let buffer = '';

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep last incomplete line

            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed || trimmed.startsWith(':')) continue; // comments or keepalives

              if (trimmed.startsWith('data:')) {
                const dataStr = trimmed.slice(5).trim();
                if (dataStr === '[DONE]') break;

                try {
                  const parsed = JSON.parse(dataStr);
                  const chunk = parsed.token || parsed.message || parsed.content || '';
                  accumulatedText += chunk;
                } catch {
                  // Plain string data chunk
                  accumulatedText += dataStr;
                }
              } else {
                accumulatedText += line;
              }

              assistantMsg.content = accumulatedText;
              bubble.innerHTML = MarkdownRenderer.render(accumulatedText) + '<span class="cursor-caret"></span>';
              this._scrollToBottom();
            }
          }

          // Clean final render without cursor
          bubble.innerHTML = MarkdownRenderer.render(accumulatedText || 'Respuesta completada.');
          assistantMsg.content = accumulatedText;
        } else {
          // Standard JSON fallback
          const data = await response.json();
          const replyText = data.message || 'Respuesta recibida.';
          assistantMsg.content = replyText;
          bubble.innerHTML = MarkdownRenderer.render(replyText);
        }

      } catch (err) {
        typingIndicator.style.display = 'none';
        if (err.name === 'AbortError') {
          bubble.innerHTML = MarkdownRenderer.render(assistantMsg.content || '*Generación detenida por el usuario.*');
        } else {
          const errMsg = `⚠️ **Error de conexión**: ${err.message}. Asegúrate de que el backend de FastAPI esté corriendo en \`${this.apiUrl}\`.`;
          assistantMsg.content = errMsg;
          bubble.innerHTML = MarkdownRenderer.render(errMsg);
        }
      } finally {
        this.isStreaming = false;
        this.abortController = null;
        abortBtn.style.display = 'none';
        this.shadowRoot.querySelector('#btn-send').disabled = false;
        this._saveSessionMessages();
        this._scrollToBottom();
      }
    }

    // --- Public Methods ---
    open() {
      if (!this.isOpen) {
        this.isOpen = true;
        this.shadowRoot.querySelector('#widget-fab').classList.add('open');
        this.shadowRoot.querySelector('#widget-window').classList.add('visible');
        this.shadowRoot.querySelector('#fab-pulse').style.display = 'none';
        this._scrollToBottom();
      }
    }

    close() {
      if (this.isOpen) {
        this.isOpen = false;
        this.shadowRoot.querySelector('#widget-fab').classList.remove('open');
        this.shadowRoot.querySelector('#widget-window').classList.remove('visible');
      }
    }

    setAgent(agentId) {
      if (AGENTS[agentId]) {
        this.activeAgent = agentId;
        this._updateAgentDisplay();
      }
    }

    clearSession() {
      this.sessionId = 'sess_' + Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
      sessionStorage.setItem('ai_gateway_session_id', this.sessionId);
      this.messages = [];
      this._saveSessionMessages();
      this._renderMessagesFeed();
      this.shadowRoot.querySelector('#footer-sess-btn').textContent = this.sessionId.substring(0, 14) + '...';
    }

    sendMessage(text, agentId = null) {
      if (agentId) this.setAgent(agentId);
      this.open();
      const input = this.shadowRoot.querySelector('#widget-input');
      input.value = text;
      this._handleSend();
    }
  }

  // --- 5. Register Custom Element & Global Drop-in Auto-Init ---
  if (!customElements.get('ai-chat-widget')) {
    customElements.define('ai-chat-widget', AiChatWidgetElement);
  }

  // Global Helper API
  window.AiChatWidget = {
    _instance: null,
    
    init(options = {}) {
      if (this._instance) return this._instance;
      
      let widgetEl = document.querySelector('ai-chat-widget');
      if (!widgetEl) {
        widgetEl = document.createElement('ai-chat-widget');
        if (options.apiUrl) widgetEl.setAttribute('api-url', options.apiUrl);
        if (options.defaultAgent) widgetEl.setAttribute('default-agent', options.defaultAgent);
        document.body.appendChild(widgetEl);
      }
      this._instance = widgetEl;
      return widgetEl;
    },

    open() { this._instance?.open(); },
    close() { this._instance?.close(); },
    setAgent(id) { this._instance?.setAgent(id); },
    sendMessage(text, agentId) { this._instance?.sendMessage(text, agentId); },
    clearSession() { this._instance?.clearSession(); }
  };

  // Auto-instantiate if loaded via <script data-api-url="...">
  document.addEventListener('DOMContentLoaded', () => {
    const currentScript = document.currentScript || document.querySelector('script[src*="chat-widget.js"]');
    const autoInit = currentScript?.dataset.autoInit !== 'false';
    
    if (autoInit && !document.querySelector('ai-chat-widget')) {
      const apiUrl = currentScript?.dataset.apiUrl || 'http://localhost:8000';
      const defaultAgent = currentScript?.dataset.agent || 'portfolio';
      window.AiChatWidget.init({ apiUrl, defaultAgent });
    }
  });

})();
