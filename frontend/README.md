# 🎨 Frontend & Drop-in Chat Widget | AI Agent Gateway

Interfaz de usuario moderna, fluida y responsiva para el Chatbot de IA y Gateway Multi-Agente construida con **HTML5 semántico**, **Vanilla CSS** (Design System con Tokens) y **Vanilla JavaScript (ES6+ asíncrono)** bajo la regla estricta de **Zero-Dependencies**.

---

## 📦 Componentes Incluidos

| Archivo | Rol / Descripción |
| :--- | :--- |
| [`index.html`](./index.html) | Aplicación web completa de chat independiente (Full-Screen Web App) con Tabs en Navbar y panel de métricas en vivo. |
| [`styles.css`](./styles.css) | Hoja de estilos con variables CSS, tema oscuro slate/zinc, soporte para tablas KPI, animaciones y diseño responsivo. |
| [`app.js`](./app.js) | Controlador ES6+ asíncrono para la aplicación completa (SSE stream, live performance metrics, markdown table copy, sesión). |
| [`chat-widget.js`](./chat-widget.js) | **Widget universal "Drop-in"** encapsulado con **Shadow DOM** (aislamiento 100% de CSS/JS). |
| [`demo.html`](./demo.html) | Página de prueba para validar la integración del widget en un portal anfitrión / monolito. |

---

## 📊 Novedades Ticket FE-02: Analytics Corporativo & Panel de Métricas

1. **Pestaña de Navegación "📊 Analytics Corporativo" en el Navbar**:
   - Pestaña de navegación en el Navbar (`💼 Portfolio` | `🛍️ E-Commerce Chat` | `📊 Analytics Corporativo` | `⚡ General`).
   - Diferenciación visual temática con badge de **KPIs**.
2. **Barra de Métricas de Rendimiento en Tiempo Real**:
   - ⚡ **Latencia:** Mide el tiempo de respuesta inicial / TTFT (*Time to First Token*) en milisegundos.
   - 🔤 **Tokens:** Contador acumulado en tiempo real.
   - 🚀 **Velocidad:** Cálculo de tokens por segundo (`tok/s`).
   - 🧠 **Modelo:** Indicador del modelo activo (`gemini-2.0-flash`).
   - 📡 **Canal SSE:** Indicador pulsante de estado de conexión Server-Sent Events.
3. **Barra de Accesos Rápidos de Analytics (`analytics-quick-strip`)**:
   - 📈 **Resumen de Ventas:** Resumen ejecutivo de ventas y facturación trimestral.
   - 👥 **Tráfico y Usuarios:** Métricas de visitas, DAU, MAU y duración de sesión.
   - 🎯 **Embudo de Conversión:** Análisis de etapas y tasas de abandono.
   - 📊 **KPIs Mensuales:** Comparativa de MRR, CAC, LTV y Churn.
   - 📉 **Latencia & Rendimiento:** Métricas p50/p95/p99 y tasa de aciertos en Redis.
   - 💰 **ROI & Inversión:** Cálculo de retorno por canal de marketing.
4. **Renderizado de Tablas Markdown Avanzado con Scroll Suave & Copia**:
   - Contenedor con scroll horizontal táctil suave (`overflow-x: auto`) y cabeceras fijas.
   - Detección y renderizado automático de badges de KPI (▲ Verde para incrementos/éxitos, ▼ Rojo para caídas/alertas, ● Azul para estabilidad).
   - Botón interactivo **"📋 Copiar Tabla"** para exportar instantáneamente el contenido de la tabla.
5. **Control de Conversación Rápido**:
   - Botón `[ ↺ Reiniciar Chat ]` en la barra de métricas y cabecera para comenzar nuevas sesiones limpias.

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
  defaultAgent: 'analytics' // 'portfolio' | 'ecommerce' | 'analytics' | 'general'
});

// Controlar apertura y envío de mensajes desde la app anfitriona
window.AiChatWidget.open();
window.AiChatWidget.setAgent('analytics');
window.AiChatWidget.sendMessage('Genera una tabla comparativa de KPIs mensuales.');
window.AiChatWidget.clearSession();
```

---

## ⚡ 2. Características y Funcionalidades

### 🔹 Consumo de Streaming SSE en Tiempo Real
- Conexión a `POST /api/v1/chat/stream` usando `fetch()` con `ReadableStream` (`reader.read()`).
- Decodificación por fragmentos UTF-8 y lectura de eventos `data: {"token": "..."}` o streams de texto directo.
- Cursor interactivo y actualización incremental del DOM sin retrasos perceptibles ni parpadeos.
- Botón de **Detener Generación (`AbortController`)** durante el streaming.

### 🔹 Selector y Orquestación Multi-Agente
- 💼 **Portfolio Agent (`portfolio`)**: Experiencia profesional, proyectos, stack y CV.
- 🛍️ **E-commerce Agent (`ecommerce`)**: Catálogo de productos, cotizaciones y pedidos.
- 📊 **Analytics Agent (`analytics`)**: Métricas corporativas, rendimiento, latencia y KPIs de negocio.
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
  - Tablas con scroll horizontal suave y badges de KPIs.
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
- App Completa con Tabs y Métricas: [http://localhost:3000/index.html](http://localhost:3000/index.html)
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

1. **Prueba de Navegación & Tabs del Navbar:**
   - Hacer clic en cada pestaña del Navbar (`Portfolio`, `E-Commerce Chat`, `Analytics Corporativo`, `General`).
   - Verificar que la pestaña activa cambie de estilo visual y actualice el avatar y las sugerencias contextuales.
2. **Prueba de Barra de Métricas en Vivo:**
   - Enviar una consulta. Observar cómo la barra superior calcula en tiempo real la **latencia inicial (ms)**, los **tokens acumulados** y la **velocidad (tok/s)**.
3. **Prueba de Consultas Rápidas de Analytics:**
   - Al seleccionar la pestaña "📊 Analytics Corporativo", hacer clic en cualquiera de los chips de negocio (ej. "📈 Resumen de Ventas" o "📊 KPIs Mensuales").
   - Verificar que se envíe la consulta y se renderice la respuesta estructurada en tabla.
4. **Prueba de Tablas Markdown con Scroll & Botón Copiar:**
   - Verificar el scroll horizontal suave en resoluciones estrechas.
   - Hacer clic en el botón "📋 Copiar Tabla" y validar que el contenido Markdown se copie correctamente al portapapeles.
5. **Prueba de Reinicio Rápido de Conversación:**
   - Presionar el botón `[ ↺ Reiniciar Chat ]` en la barra de métricas y verificar que se genera un nuevo `session_id` y se limpia el historial.
