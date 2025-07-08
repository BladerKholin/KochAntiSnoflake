# Sistema Distribuido Koch - Docker Compose

Sistema distribuido para ejecutar algoritmos de generación de curvas de Koch (Classic y JAX) usando Docker y comunicación HTTP.

## 🏗️ Arquitectura

- **Orquestador**: Gestiona cola de trabajos y distribución a esclavos
- **3 Esclavos**: Ejecutan algoritmos y generan imágenes
- **Algoritmos**: `ClassicKoch` (recursivo) y `JAXKoch` (optimizado)
- **Dashboard**: Web en tiempo real con gráficos interactivos
- **Elasticidad**: Detección automática de esclavos activos/inactivos

## 🚀 Inicio Rápido

```bash
# Automatizado
python inicio_rapido.py

# Manual
docker-compose up -d
```

Dashboard: [http://localhost:8000](http://localhost:8000)

## 📋 Comandos Principales

### Administración
```bash
# Estado del sistema
python admin.py estado

# Agregar trabajo
python admin.py trabajo ClassicKoch 5 --size 3.0 --dpi 300

# Comparar algoritmos
python admin.py comparar --iteraciones 3 4 5 6

# Stress test
python admin.py stress --cantidad 15

# Monitoreo continuo
python admin.py monitorear --intervalo 3
```

### Pruebas Automatizadas
```bash
# Ejecutar 26 casos de prueba
python test.py

# Con intervalo personalizado
python test.py --intervalo 0.5

# Verificar conectividad
python test.py --orquestador http://localhost:8000
```

## 🧪 Sistema de Pruebas

- **`test.py`**: Script de pruebas automatizadas
- **`test_cases.json`**: 26 casos predefinidos (ClassicKoch y JAXKoch, 1-11 iteraciones)
- Validación automática de conectividad y resultados

## 📊 Dashboard

**URL**: [http://localhost:8000](http://localhost:8000)

Características:
- Auto-actualización cada 3 segundos (AJAX)
- Gráficos interactivos (Chart.js)
- Estado de esclavos en tiempo real
- Cola de trabajos y estadísticas
- Responsive design

## 🐳 Docker

```bash
# Gestión básica
docker-compose up -d        # Iniciar
docker-compose logs -f      # Ver logs
docker-compose down         # Parar
docker-compose ps           # Estado

# Diagnóstico
docker stats                # Recursos
```

## 🛠️ API REST

- `GET /` - Dashboard web
- `GET /status` - Estado del sistema
- `POST /trabajo` - Agregar trabajo
- `GET /resultado/<id>` - Obtener resultado

Ejemplo trabajo:
```json
{
  "algoritmo": "ClassicKoch",
  "parametros": {
    "iteraciones": 5,
    "size": 3.0,
    "dpi": 300
  }
}
```

## 📁 Estructura

```
Tarea Distribuido/
├── docker-compose.yml      # Orquestación
├── admin.py               # Administración
├── test.py                # Pruebas automatizadas
├── test_cases.json        # Casos de prueba
├── inicio_rapido.py       # Inicio automatizado
├── orquestador/
│   ├── orquestador.py
│   ├── templates/dashboard.html
│   └── static/
├── esclavo/
│   ├── esclavo.py
│   ├── ClassicKoch.py
│   └── JaxKoch.py
├── config/pruebas.json
└── resultados/
```

## � Solución de Problemas

```bash
# Verificar estado
python admin.py estado
python test.py --orquestador http://localhost:8000
docker-compose ps

# Ver logs
docker-compose logs -f esclavo1

# Prueba básica
python admin.py trabajo ClassicKoch 1
```

---

**Sistema distribuido elástico con Docker, dashboard interactivo y suite de pruebas automatizadas**