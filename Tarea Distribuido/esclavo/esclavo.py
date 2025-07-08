from flask import Flask, request, jsonify
import time
import os
import uuid
from datetime import datetime
import requests
import threading
from ClassicKoch import KochSnowflake
from JaxKoch import JAXKochSnowflake

app = Flask(__name__)

class Esclavo:
    def __init__(self, node_id):
        self.node_id = node_id
        self.status = 'activo'
        self.trabajos_ejecutados = 0
        self.tiempo_total = 0
        self.orquestador_url = os.getenv('ORQUESTADOR_URL')
        
        # Inicializar generadores
        self.classic_koch = KochSnowflake()
        try:
            self.jax_koch = JAXKochSnowflake()
            self.jax_disponible = True
        except Exception as e:
            print(f"❌ JAX no disponible: {e}")
            self.jax_koch = None
            self.jax_disponible = False
        
        print(f"🤖 Esclavo {self.node_id} inicializado")
        print(f"   Classic Koch: ✅ Disponible")
        print(f"   JAX Koch: {'✅ Disponible' if self.jax_disponible else '❌ No disponible'}")
    
    def ejecutar_trabajo(self, trabajo):
        """Ejecuta un trabajo específico y retorna el resultado"""
        inicio = time.time()
        
        try:
            algoritmo = trabajo.get('algoritmo')
            parametros = trabajo.get('parametros', {})
            
            iteraciones = parametros.get('iteraciones', 4)
            size = parametros.get('size', 3.0)
            dpi = parametros.get('dpi', 300)
            
            # Generar nombre de archivo único
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.node_id}_{algoritmo}_{timestamp}_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join('/app/resultados', filename)
            
            print(f"🔄 Ejecutando {algoritmo} con {iteraciones} iteraciones...")
            
            # Ejecutar según el algoritmo
            if algoritmo == 'ClassicKoch':
                resultado = self.classic_koch.save_snowflake(
                    filename=filepath,
                    iterations=iteraciones,
                    size=size,
                    dpi=dpi,
                    figsize=(8, 8),
                    line_width=1.5,
                    color='darkblue'
                )
            elif algoritmo == 'JAXKoch':
                if not self.jax_disponible:
                    raise Exception("JAX no está disponible en este esclavo")
                resultado = self.jax_koch.save_snowflake(
                    filename=filepath,
                    iterations=iteraciones,
                    size=size,
                    dpi=dpi,
                    figsize=(8, 8),
                    line_width=1.5,
                    color='darkred'
                )
            else:
                raise Exception(f"Algoritmo no soportado: {algoritmo}")
            
            tiempo_ejecucion = time.time() - inicio
            
            # Actualizar estadísticas
            self.trabajos_ejecutados += 1
            self.tiempo_total += tiempo_ejecucion
            
            return {
                'exito': True,
                'algoritmo': algoritmo,
                'parametros': parametros,
                'resultado': resultado,
                'tiempo_ejecucion': tiempo_ejecucion,
                'archivo_generado': filename,
                'esclavo_id': self.node_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            tiempo_ejecucion = time.time() - inicio
            print(f"❌ Error ejecutando trabajo: {e}")
            
            return {
                'exito': False,
                'error': str(e),
                'tiempo_ejecucion': tiempo_ejecucion,
                'esclavo_id': self.node_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def obtener_estadisticas(self):
        """Retorna las estadísticas del esclavo"""
        return {
            'node_id': self.node_id,
            'status': self.status,
            'trabajos_ejecutados': self.trabajos_ejecutados,
            'tiempo_total': self.tiempo_total,
            'tiempo_promedio': self.tiempo_total / max(1, self.trabajos_ejecutados),
            'algoritmos_disponibles': {
                'ClassicKoch': True,
                'JAXKoch': self.jax_disponible
            }
        }

# Crear instancia del esclavo
node_id = os.getenv('NODE_ID', 'esclavo_desconocido')
esclavo = Esclavo(node_id)

@app.route('/status')
def status():
    """Endpoint para verificar el estado del esclavo"""
    return jsonify(esclavo.obtener_estadisticas())

@app.route('/ejecutar', methods=['POST'])
def ejecutar_trabajo():
    """Endpoint para ejecutar un trabajo"""
    try:
        trabajo = request.json
        if not trabajo:
            return jsonify({'error': 'No se recibió trabajo'}), 400
        
        print(f"📥 Trabajo recibido: {trabajo.get('id', 'sin-id')}")
        resultado = esclavo.ejecutar_trabajo(trabajo)
        
        if resultado['exito']:
            print(f"✅ Trabajo completado en {resultado['tiempo_ejecucion']:.2f}s")
            return jsonify(resultado)
        else:
            print(f"❌ Trabajo falló: {resultado['error']}")
            return jsonify(resultado), 500
            
    except Exception as e:
        print(f"❌ Error procesando trabajo: {e}")
        return jsonify({
            'exito': False,
            'error': str(e),
            'esclavo_id': esclavo.node_id,
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Endpoint para cerrar el esclavo de manera elegante"""
    def shutdown_server():
        time.sleep(1)
        # Notificar al orquestador que se va a cerrar
        if esclavo.orquestador_url:
            try:
                requests.post(f"{esclavo.orquestador_url}/esclavo_desconectado", 
                            json={'esclavo_id': esclavo.node_id})
            except:
                pass
        
        os._exit(0)
    
    threading.Thread(target=shutdown_server).start()
    return jsonify({'message': f'Esclavo {esclavo.node_id} cerrándose...'})

@app.route('/')
def info():
    """Información básica del esclavo"""
    stats = esclavo.obtener_estadisticas()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Esclavo {esclavo.node_id}</title>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="5">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .card {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .status-activo {{ color: green; font-weight: bold; }}
            .available {{ color: green; }}
            .unavailable {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Esclavo {esclavo.node_id}</h1>
            
            <div class="card">
                <h2>📊 Estadísticas</h2>
                <p><strong>Estado:</strong> <span class="status-{stats['status']}">{stats['status']}</span></p>
                <p><strong>Trabajos ejecutados:</strong> {stats['trabajos_ejecutados']}</p>
                <p><strong>Tiempo total:</strong> {stats['tiempo_total']:.2f}s</p>
                <p><strong>Tiempo promedio:</strong> {stats['tiempo_promedio']:.2f}s</p>
            </div>
            
            <div class="card">
                <h2>⚙️ Algoritmos Disponibles</h2>
                <p><strong>Classic Koch:</strong> 
                   <span class="{'available' if stats['algoritmos_disponibles']['ClassicKoch'] else 'unavailable'}">
                   {'✅ Disponible' if stats['algoritmos_disponibles']['ClassicKoch'] else '❌ No disponible'}
                   </span>
                </p>
                <p><strong>JAX Koch:</strong> 
                   <span class="{'available' if stats['algoritmos_disponibles']['JAXKoch'] else 'unavailable'}">
                   {'✅ Disponible' if stats['algoritmos_disponibles']['JAXKoch'] else '❌ No disponible'}
                   </span>
                </p>
            </div>
            
            <div class="card">
                <h2>🔗 API Endpoints</h2>
                <ul>
                    <li><code>GET /status</code> - Estado del esclavo</li>
                    <li><code>POST /ejecutar</code> - Ejecutar trabajo</li>
                    <li><code>POST /shutdown</code> - Cerrar esclavo</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

if __name__ == '__main__':
    print(f"🚀 Iniciando esclavo {esclavo.node_id}...")
    app.run(host='0.0.0.0', port=5000, debug=False)
