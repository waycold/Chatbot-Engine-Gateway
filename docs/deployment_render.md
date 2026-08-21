# 🚀 Guía Integral de Despliegue en Render: Chatbot-Engine-Gateway

Esta guía detalla el procedimiento completo para desplegar el microservicio **Chatbot-Engine-Gateway** en la plataforma en la nube **Render**, asegurando alta disponibilidad, seguridad, rendimiento en streaming Server-Sent Events (SSE) y prevención de hibernación por inactividad.

---

## 📑 Tabla de Contenidos
1. [Arquitectura de Infraestructura](#-arquitectura-de-infraestructura)
2. [Requisitos Previos](#-requisitos-previos)
3. [Estrategia de Despliegue con Blueprint (`render.yaml`)](#-estrategia-de-despliegue-con-blueprint-renderyaml)
4. [Configuración de Variables de Entorno y Secretos](#-configuración-de-variables-de-entorno-y-secretos)
5. [Configuración de Redis para Memoria de Agentes](#-configuración-de-redis-para-memoria-de-agentes)
6. [Mecanismo Anti-Inactividad (Keep-Alive)](#-mecanismo-anti-inactividad-keep-alive)
7. [Optimización de Uvicorn para Streaming SSE](#-optimización-de-uvicorn-para-streaming-sse)
8. [Verificación y Health Checks](#-verificación-y-health-checks)
9. [Resolución de Problemas Frecuentes (Troubleshooting)](#-resolución-de-problemas-frecuentes-troubleshooting)

---

## 🏛️ Arquitectura de Infraestructura

```
                            ┌────────────────────────────────────────┐
                            │            CLIENTES / WEB              │
                            └──────────────────┬─────────────────────┘
                                               │ HTTPS / SSE Stream
                                               ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ RENDER CLUSTER (Oregon / Global Edge Proxy)                                       │
│                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────────┐   │
│   │ Web Service: ai-agent-gateway (Docker Multi-Stage Container)              │   │
│   │                                                                           │   │
│   │   • Non-Root User (appuser:1000)                                          │   │
│   │   • FastAPI + Uvicorn (timeout-keep-alive: 65s)                           │   │
│   │   • Endpoints: /api/v1/chat/stream, /health, /ping                        │   │
│   └───────────────┬───────────────────────────┬───────────────────────┬───────┘   │
└───────────────────┼───────────────────────────┼───────────────────────┼───────────┘
                    │                           │                       │
                    ▼                           ▼                       ▼
    ┌────────────────────────┐      ┌─────────────────────┐  ┌─────────────────────┐
    │ Google Gemini Studio   │      │ Redis Session Store │  │ Monolito Django     │
    │ (gemini-3.7-flash API) │      │ (Upstash / Render)  │  │ (Backend de Negocio)│
    └────────────────────────┘      └─────────────────────┘  └─────────────────────┘
```

---

## 📋 Requisitos Previos

1. **Cuenta en Render**: [https://render.com](https://render.com) (Plan Free, Individual o Team).
2. **Repositorio en GitHub**: Acceso de lectura/escritura al repositorio `Chatbot-Engine-Gateway`.
3. **Google AI Studio API Key**: Clave válida para el modelo `gemini-3.7-flash` generada en [Google AI Studio](https://aistudio.google.com/).
4. **Backend Django URL y Secret**: URL accesible del backend transaccional y el `INTERNAL_API_SECRET` compartido.
5. **Instancia de Redis**: URI de conexión (`redis://...` o `rediss://...` con SSL).

---

## 🛠️ Estrategia de Despliegue con Blueprint (`render.yaml`)

El repositorio incluye el archivo Infrastructure-as-Code (IaC) `render.yaml` listo para aprovisionar automáticamente el microservicio.

### Pasos de Despliegue:

1. Inicia sesión en el [Render Dashboard](https://dashboard.render.com/).
2. Haz clic en **New +** en la esquina superior derecha y selecciona **Blueprint**.
3. Conecta tu cuenta de GitHub y selecciona el repositorio `Chatbot-Engine-Gateway`.
4. Render detectará automáticamente el archivo `render.yaml` y configurará:
   - **Service Name**: `ai-agent-gateway`
   - **Runtime**: `Docker` (usando el `Dockerfile` multi-stage optimizado)
   - **Health Check Path**: `/health`
   - **Branch**: `main`
   - **Auto-Deploy**: Activado en cada `git push` a `main`.
5. Haz clic en **Apply Blueprint**.

---

## 🔐 Configuración de Variables de Entorno y Secretos

En el Dashboard de Render, navega a tu servicio **ai-agent-gateway** > **Environment** y completa los valores requeridos:

| Variable | Tipo | Requerido | Descripción / Ejemplo |
| :--- | :--- | :---: | :--- |
| `GEMINI_API_KEY` | Secret | **Sí** | Clave de API de Google Gemini (ej: `AIzaSy...`). |
| `INTERNAL_API_SECRET` | Secret | **Sí** | Secreto compartido generado para validar requests internos. |
| `DJANGO_BACKEND_URL` | String | **Sí** | URL base del backend Django (ej: `https://mi-backend.onrender.com`). |
| `REDIS_URL` | Secret | **Sí** | URI de conexión a Redis (ej: `rediss://default:pwd@host.upstash.io:6379`). |
| `ENVIRONMENT` | String | **Sí** | `production` |
| `DEBUG` | Boolean | **Sí** | `false` |
| `BACKEND_CORS_ORIGINS` | String | **Sí** | Orígenes permitidos separados por comas (ej: `https://mi-frontend.vercel.app,https://mi-backend.onrender.com`). |
| `DEFAULT_MODEL` | String | No | Modelo LLM por defecto (`gemini-3.7-flash`). |
| `PYTHONUNBUFFERED` | String | No | `1` (fuerza salida inmediata de logs sin buffer). |

---

## 🗄️ Configuración de Redis para Memoria de Agentes

El microservicio utiliza Redis para almacenar el historial de conversación y contexto de sesión de los agentes.

### Opción A: Upstash Redis Serverless (Recomendado para Free/Serverless Tier)
1. Crea una base de datos gratuita en [Upstash Console](https://console.upstash.com/).
2. Copia la cadena de conexión con TLS (`rediss://default:...@...upstash.io:6379`).
3. Pégala en la variable `REDIS_URL` en Render.

### Opción B: Redis en Render
1. En Render Dashboard, haz clic en **New +** > **Redis**.
2. Asigna un nombre (ej: `ai-gateway-redis`).
3. Una vez creado, copia la **Internal Redis URL** y asígnala a `REDIS_URL`.

---

## ⏱️ Mecanismo Anti-Inactividad (Keep-Alive)

En los planes **Render Free / Eco**, los servicios web se suspenden automáticamente tras **15 minutos de inactividad**. La reactivación ("cold start") puede demorar entre 30 y 50 segundos.

Para garantizar respuesta inmediata (0 cold-start), se incluye una estrategia Keep-Alive automatizada:

### Método 1: GitHub Actions Cron (Incluido en el Repositorio)

El archivo `.github/workflows/keep_alive.yml` ejecuta el script ligero `scripts/keep_alive.py` cada **10 minutos**:

1. En tu repositorio GitHub, ve a **Settings** > **Secrets and variables** > **Actions**.
2. En la pestaña **Variables** (o **Secrets**), crea:
   - **Name**: `GATEWAY_URL`
   - **Value**: `https://ai-agent-gateway.onrender.com` (la URL pública generada por Render).
3. La acción se ejecutará automáticamente cada 10 minutos y registrará la latencia en el Step Summary de GitHub Actions.

### Método 2: Pings Manuales o Cron Externo
Puedes configurar un monitor HTTP gratuito en [cron-job.org](https://cron-job.org) o [UptimeRobot](https://uptimerobot.com):
- **URL**: `https://ai-agent-gateway.onrender.com/health`
- **Método**: `GET`
- **Frecuencia**: Cada 10 minutos.
- **Respuesta esperada**: HTTP `200 OK` con payload `{"status":"ok"}`.

> [!NOTE]
> El endpoint `/health` y `/ping` no consumen tokens de Gemini ni ejecutan consultas pesadas en Redis, lo que permite pings constantes a costo cero.

---

## 🌊 Optimización de Uvicorn para Streaming SSE

Para evitar que el proxy inverso de Render o Cloudflare interrumpa las conexiones de Server-Sent Events durante respuestas largas de LLM:

1. **Keep-Alive Timeout**: Configurado en `65s` (superior al timeout de 60s de los balanceadores).
2. **Proxy Headers**: Activado `--proxy-headers` y `--forwarded-allow-ips='*'` para preservar IPs originales y protocolos HTTPS.
3. **No-Buffering**: Los endpoints de streaming devuelven cabeceras estándar SSE:
   - `Content-Type: text/event-stream`
   - `Cache-Control: no-cache, no-transform`
   - `X-Accel-Buffering: no`

---

## 🔍 Verificación y Health Checks

Una vez desplegado el servicio, puedes verificar su operatividad:

### 1. Ping Ultraligero
```bash
curl -i https://ai-agent-gateway.onrender.com/ping
```
**Respuesta:**
```json
{"ping": "pong"}
```

### 2. Health Check de Servicio
```bash
curl -i https://ai-agent-gateway.onrender.com/health
```
**Respuesta:**
```json
{
  "status": "ok",
  "app_name": "AI Agent Gateway",
  "environment": "production",
  "version": "0.1.0"
}
```

### 3. Diagnóstico Profundo de Dependencias
```bash
curl -i https://ai-agent-gateway.onrender.com/health/details
```
**Respuesta:**
```json
{
  "status": "ok",
  "app_name": "AI Agent Gateway",
  "environment": "production",
  "version": "0.1.0",
  "redis_healthy": true,
  "django_healthy": true
}
```

### 4. Prueba de Streaming SSE
```bash
curl -N -X POST "https://ai-agent-gateway.onrender.com/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -H "X-Internal-Secret: TU_SECRETO_AQUI" \
  -d '{
    "agent_id": "general_support",
    "session_id": "test-render-session-1",
    "message": "Hola, confirma que el servicio en Render está operativo."
  }'
```

---

## 🔧 Resolución de Problemas Frecuentes (Troubleshooting)

### 1. El servicio no inicia y falla el Health Check inicial
- **Causa**: Render espera que el servicio responda en `0.0.0.0:$PORT`.
- **Solución**: El contenedor utiliza `scripts/render_entrypoint.sh` que resuelve dinámicamente la variable `$PORT` asignada por Render.

### 2. Error 401/403 en peticiones desde Django
- **Causa**: La variable `INTERNAL_API_SECRET` no coincide entre el backend Django y el Gateway.
- **Solución**: Verifica que el valor configurado en el Dashboard de Render coincida exactamente con el de Django.

### 3. Desconexiones en Streaming SSE
- **Causa**: Buffering de proxy o timeout de inactividad de conexión HTTP.
- **Solución**: Asegúrate de que `TIMEOUT_KEEP_ALIVE=65` esté activo y que el cliente consuma los chunks inmediatamente.

### 4. Error de conexión con Redis (`ConnectionRefusedError`)
- **Causa**: Si usas Upstash o Redis con TLS en la nube, la URL debe comenzar con `rediss://` (con doble `s`).
