# 🎨 Frontend & Drop-in Chat Widget | AI Agent Gateway

Interfaz de usuario moderna, fluida y responsiva para el Chatbot de IA y Gateway Multi-Agente construida con **HTML5 semántico**, **Vanilla CSS** (Design System con Tokens) y **Vanilla JavaScript (ES6+ asíncrono)** bajo la regla estricta de **Zero-Dependencies**.

---

## 📦 Componentes Incluidos

| Archivo | Rol / Descripción |
| :--- | :--- |
| [`index.html`](./index.html) | Aplicación web completa de chat independiente (Full-Screen Web App). |
| [`styles.css`](./styles.css) | Hoja de estilos con variables CSS, tema oscuro slate/zinc, animaciones y diseño responsivo. |
| [`app.js`](./app.js) | Controlador ES6+ asíncrono para la aplicación completa (SSE stream, historial, markdown, configuración). |
| [`chat-widget.js`](./chat-widget.js) | **Widget universal "Drop-in"** encapsulado con **Shadow DOM** (aislamiento 100% de CSS/JS). |
| [`demo.html`](./demo.html) | Página de prueba para validar la integración del widget en un portal anfitrión / monolito. |

---

## 🚀 1. Integración Drop-in (Monolito Django o Sitios Externos)

Para integrar el chatbot en cualquier template de Django (`templates/base.html`) o sitio web estático/externo:

### Opción A: Inclusión vía Script Tag (Automático)
Inserta la siguiente etiqueta antes del cierre de `</body>`:

```html
<!-- Carga del Drop-in Widget -->
<script 
  src="/static/chat-widget.js" 
  data-api-url="http://localhost:8000"
  data-agent="portfolio"
></script>
```

### Opción B: Declaración como Web Component
```html
<script src="/static/chat-widget.js" data-auto-init="false"></script>
<ai-chat-widget api-url="http://localhost:8000" default-agent="portfolio"></ai-chat-widget>
```

### Opción C: Control Programático con JavaScript API
```javascript
// Inicializar
window.AiChatWidget.init({
  apiUrl: 'http://localhost:8000',
  defaultAgent: 'portfolio' // 'portfolio' | 'ecommerce' | 'analytics' | 'general'
});

// Controlar apertura y envío de mensajes desde la app anfitriona
window.AiChatWidget.open();
window.AiChatWidget.setAgent('ecommerce');
window.AiChatWidget.sendMessage('Hola, ¿qué servicios tienen disponibles?');
window.AiChatWidget.clearSession();
```

---

## ⚡ 2. Características y Funcionalidades

### 🔹 Consumo de Streaming SSE en Tiempo Real
- Conexión a `POST /api/v1/chat/stream` usando `fetch()` con `ReadableStream` (`reader.read()`).
- Decodificación por fragmentos UTF-8 y lectura de eventos `data: {"token": "..."}` o streams de texto directo.
- Cursor interactivo y actualización incremental del DOM sin retrasos perceptibles ni parpadeos.
- Botón de **Detener Generación (AbortController)** durante el streaming.

### 🔹 Selector y Orquestación Multi-Agente
- 💼 **Portfolio Agent (`portfolio`)**: Experiencia profesional, proyectos, stack y CV.
- 🛍️ **E-commerce Agent (`ecommerce`)**: Catálogo de productos, cotizaciones y pedidos.
- 📊 **Analytics Agent (`analytics`)**: Métricas de rendimiento, latencia y KPIs de negocio.
- ⚡ **General Assistant (`general`)**: Consultas abiertas y orquestador general.

### 🔹 Persistencia de Sesión Multi-Turno
- Generación y almacenamiento automático de `session_id` en `sessionStorage` (widget) y `localStorage` (app completa).
- Soporte para mantener el hilo conversacional sincronizado con la memoria en Redis del microservicio.
- Botón para reiniciar conversación y generar un nuevo `session_id`.

### 🔹 Renderizador Markdown & Bloques de Código Integrado (Zero-Dependencies)
- Soporte nativo para:
  - Títulos (`#`, `##`, `###`), Negrita (`**`), Cursiva (`*`), Tachado (`~~`).
  - Listas ordenadas y desordenadas.
  - Citas en bloque (`> quote`).
  - Tablas (`| col | col |`).
  - Enlaces seguros (`target="_blank" rel="noopener noreferrer"`).
  - Bloques de código con cabecera de lenguaje y botón interactivo **"Copiar código"** con feedback inmediato.

### 🔹 Monitor de Salud y Estado de Conexión
- Ping recurrente al endpoint `GET /health`.
- Badge visual en tiempo real:
  - 🟢 **Online**: Microservicio FastAPI conectado y operativo.
  - 🟡 **Verificando**: Comprobando conectividad o reconectando.
  - 🔴 **Offline**: Sin conexión (muestra instrucciones de resolución).

---

## 🛠️ 3. Cómo Servir y Probar Localmente

### Opción 1: Servidor HTTP Local (Python)
```bash
# Desde la carpeta frontend/
cd frontend
python -m http.server 3000
```
Abrir en el navegador:
- App Completa: [http://localhost:3000/index.html](http://localhost:3000/index.html)
- Demo Drop-in Widget: [http://localhost:3000/demo.html](http://localhost:3000/demo.html)

### Opción 2: Montado en FastAPI como Archivos Estáticos
En `app/main.py` de FastAPI:
```python
from fastapi.staticfiles import StaticFiles

# Montar directorio frontend
application.mount("/chat", StaticFiles(directory="frontend", html=True), name="frontend")
```
Acceder directamente en: [http://localhost:8000/chat](http://localhost:8000/chat)

---

## 🧪 4. Guía de Verificación para QA Engineer

1. **Prueba de Carga & Estilos:**
   - Abrir `index.html` y `demo.html` en navegadores modernos (Chrome, Firefox, Safari, Edge).
   - Verificar diseño responsivo redimensionando a dimensiones móviles (< 480px) y desktop.
2. **Prueba de Shadow DOM:**
   - En `demo.html`, verificar que los estilos CSS globales de la página anfitriona no afecten el interior del widget y viceversa.
3. **Prueba de Conexión y Salud:**
   - Iniciar FastAPI (`uvicorn app.main:app --reload --port 8000`).
   - El badge debe cambiar a **Online** (verde). Si se apaga FastAPI, debe pasar a **Offline** (rojo).
4. **Prueba de Streaming SSE:**
   - Enviar un mensaje. Observar la animación de typing y cómo el texto se renderiza token a token con el cursor activo.
   - Presionar "Detener" durante la respuesta para verificar que el `AbortController` cancela el stream de forma limpia.
5. **Prueba de Markdown & Código:**
   - Solicitar un ejemplo de código en Python. Verificar el renderizado del bloque de código y probar el botón "Copiar código".
6. **Prueba de Multi-Turno & Sesión:**
   - Enviar varios mensajes consecutivos y verificar que el `session_id` se conserva entre peticiones.
   - Presionar "Nuevo Chat" y verificar que se genera un nuevo `session_id`.
