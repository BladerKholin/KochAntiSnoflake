import flask
from flask import Flask, request, jsonify, render_template
import requests
import json
import time
import threading
from datetime import datetime
import os
import uuid
from queue import Queue, Empty

app = Flask(__name__)

class Orquestador:
    def __init__(self):
        self.esclavos = {}  # {id: {url, status, last_ping, trabajos_ejecutados}}
        self.cola_trabajos = Queue()
        self.trabajos_en_proceso = {}  # {trabajo_id: {esclavo_id, inicio, ...}}
        self.resultados = {}  # {trabajo_id: resultado}
        self.estadisticas = {
            'trabajos_completados': 0,
            'trabajos_fallidos': 0,
            'tiempo_total_procesamiento': 0
        }
        
        # Inicializar esclavos desde variables de entorno
        esclavos_urls = os.getenv('ESCLAVOS', '').split(',')
        for i, url in enumerate(esclavos_urls):
            if url.strip():
                esclavo_id = f"esclavo{i+1}"
                self.esclavos[esclavo_id] = {
                    'url': url.strip(),
                    'status': 'desconocido',
                    'last_ping': None,
                    'trabajos_ejecutados': 0,
                    'tiempo_total': 0
                }
        
        # Iniciar hilos de monitoreo
        self.iniciar_monitoreo()
    
    def iniciar_monitoreo(self):
        """Inicia los hilos de monitoreo de esclavos y procesamiento de trabajos"""
        threading.Thread(target=self.monitorear_esclavos, daemon=True).start()
        threading.Thread(target=self.procesar_trabajos, daemon=True).start()
    
    def monitorear_esclavos(self):
        """Monitorea constantemente el estado de los esclavos"""
        while True:
            for esclavo_id, info in self.esclavos.items():
                try:
                    response = requests.get(f"{info['url']}/status", timeout=5)
                    if response.status_code == 200:
                        info['status'] = 'activo'
                        info['last_ping'] = datetime.now()
                    else:
                        info['status'] = 'error'
                except Exception as e:
                    info['status'] = 'inactivo'
                    print(f"Error conectando con {esclavo_id}: {e}")
            
            time.sleep(10)  # Revisar cada 10 segundos
    
    def procesar_trabajos(self):
        """Procesa los trabajos en la cola asignándolos a esclavos disponibles"""
        while True:
            try:
                # Buscar esclavos activos que NO estén ejecutando trabajos
                esclavos_disponibles = []
                for eid, info in self.esclavos.items():
                    if info['status'] == 'activo':
                        # Verificar si el esclavo NO está ejecutando un trabajo
                        esclavo_ocupado = any(
                            trabajo_info['esclavo_id'] == eid 
                            for trabajo_info in self.trabajos_en_proceso.values()
                        )
                        if not esclavo_ocupado:
                            esclavos_disponibles.append((eid, info))
                
                # Asignar trabajos a TODOS los esclavos disponibles
                trabajos_asignados = 0
                for esclavo_id, esclavo_info in esclavos_disponibles:
                    if not self.cola_trabajos.empty():
                        try:
                            trabajo = self.cola_trabajos.get_nowait()
                            # Asignar trabajo en un hilo separado para no bloquear
                            threading.Thread(
                                target=self.asignar_trabajo,
                                args=(trabajo, esclavo_id, esclavo_info),
                                daemon=True
                            ).start()
                            trabajos_asignados += 1
                            print(f"📤 Trabajo {trabajo['id'][:8]} asignado a {esclavo_id}")
                        except Exception as e:
                            print(f"Error asignando trabajo a {esclavo_id}: {e}")
                            break
                
                if trabajos_asignados > 0:
                    print(f"🚀 {trabajos_asignados} trabajos asignados en paralelo")
                
                time.sleep(2)  # Revisar cada 2 segundos para mejor responsividad
                
            except Exception as e:
                print(f"Error en el procesador de trabajos: {e}")
                time.sleep(5)
    
    def asignar_trabajo(self, trabajo, esclavo_id, esclavo_info):
        """Asigna un trabajo específico a un esclavo"""
        trabajo_id = trabajo['id']
        
        try:
            # Verificar que el esclavo sigue disponible
            if esclavo_id not in self.esclavos or self.esclavos[esclavo_id]['status'] != 'activo':
                print(f"⚠️ Esclavo {esclavo_id} ya no está disponible, reintentando trabajo...")
                self.cola_trabajos.put(trabajo)  # Devolver trabajo a la cola
                return
            
            # Verificar que el esclavo no esté ya procesando un trabajo
            esclavo_ocupado = any(
                trabajo_info['esclavo_id'] == esclavo_id 
                for trabajo_info in self.trabajos_en_proceso.values()
            )
            if esclavo_ocupado:
                print(f"⚠️ Esclavo {esclavo_id} ya está ocupado, reintentando trabajo...")
                self.cola_trabajos.put(trabajo)  # Devolver trabajo a la cola
                return
            
            # Registrar trabajo en proceso ANTES de enviarlo
            self.trabajos_en_proceso[trabajo_id] = {
                'esclavo_id': esclavo_id,
                'inicio': time.time(),
                'trabajo': trabajo
            }
            
            print(f"🔄 Enviando trabajo {trabajo_id[:8]} a {esclavo_id}...")
            
            # Enviar trabajo al esclavo
            response = requests.post(
                f"{esclavo_info['url']}/ejecutar",
                json=trabajo,
                timeout=300  # 5 minutos timeout
            )
            
            if response.status_code == 200:
                resultado = response.json()
                
                # Procesar resultado exitoso
                tiempo_ejecucion = time.time() - self.trabajos_en_proceso[trabajo_id]['inicio']
                
                self.resultados[trabajo_id] = {
                    'resultado': resultado,
                    'esclavo_id': esclavo_id,
                    'tiempo_ejecucion': tiempo_ejecucion,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Actualizar estadísticas
                self.esclavos[esclavo_id]['trabajos_ejecutados'] += 1
                self.esclavos[esclavo_id]['tiempo_total'] += tiempo_ejecucion
                self.estadisticas['trabajos_completados'] += 1
                self.estadisticas['tiempo_total_procesamiento'] += tiempo_ejecucion
                
                print(f"✅ Trabajo {trabajo_id[:8]} completado por {esclavo_id} en {tiempo_ejecucion:.2f}s")
            
            else:
                self.manejar_error_trabajo(trabajo_id, f"Error HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            self.manejar_error_trabajo(trabajo_id, str(e))
        
        finally:
            # Remover de trabajos en proceso
            if trabajo_id in self.trabajos_en_proceso:
                del self.trabajos_en_proceso[trabajo_id]
    
    def manejar_error_trabajo(self, trabajo_id, error):
        """Maneja errores en la ejecución de trabajos"""
        self.estadisticas['trabajos_fallidos'] += 1
        self.resultados[trabajo_id] = {
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        print(f"Error en trabajo {trabajo_id}: {error}")

# Instancia global del orquestador
orquestador = Orquestador()

@app.route('/')
def dashboard():
    """Dashboard web para monitorear el sistema"""
    return render_template('dashboard.html', 
                         esclavos=orquestador.esclavos,
                         stats=orquestador.estadisticas,
                         cola_size=orquestador.cola_trabajos.qsize(),
                         trabajos_en_proceso=orquestador.trabajos_en_proceso,
                         resultados=dict(list(orquestador.resultados.items())[-10:]),  # Últimos 10
                         now=time.time())

@app.route('/trabajo', methods=['POST'])
def agregar_trabajo():
    """API para agregar trabajos"""
    try:
        # Detectar si es JSON o form data
        if request.is_json:
            # Desde API JSON
            trabajo = request.json
            trabajo['id'] = str(uuid.uuid4())
        else:
            # Desde formulario web
            algoritmo = request.form.get('algoritmo')
            iteraciones = int(request.form.get('iteraciones', 4))
            size = float(request.form.get('size', 3.0))
            dpi = int(request.form.get('dpi', 300))
            
            trabajo = {
                'id': str(uuid.uuid4()),
                'algoritmo': algoritmo,
                'parametros': {
                    'iteraciones': iteraciones,
                    'size': size,
                    'dpi': dpi
                }
            }
        
        orquestador.cola_trabajos.put(trabajo)
        
        # Siempre devolver JSON para compatibilidad con AJAX
        return jsonify({
            'success': True,
            'message': 'Trabajo agregado exitosamente',
            'id': trabajo['id'],
            'trabajo': trabajo
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/resultado/<trabajo_id>')
def obtener_resultado(trabajo_id):
    """Obtener resultado de un trabajo específico"""
    if trabajo_id in orquestador.resultados:
        return jsonify(orquestador.resultados[trabajo_id])
    elif trabajo_id in orquestador.trabajos_en_proceso:
        return jsonify({'status': 'en_proceso'})
    else:
        return jsonify({'error': 'Trabajo no encontrado'}), 404

@app.route('/status')
def status():
    """Estado general del orquestador"""
    return jsonify({
        'esclavos': orquestador.esclavos,
        'estadisticas': orquestador.estadisticas,
        'cola_trabajos': orquestador.cola_trabajos.qsize(),
        'trabajos_en_proceso': len(orquestador.trabajos_en_proceso)
    })

@app.route('/cargar_trabajos', methods=['POST'])
def cargar_trabajos_json():
    """Cargar trabajos desde un archivo JSON"""
    try:
        trabajos = request.json.get('trabajos', [])
        for trabajo_data in trabajos:
            trabajo = {
                'id': str(uuid.uuid4()),
                'algoritmo': trabajo_data.get('algoritmo'),
                'parametros': trabajo_data.get('parametros')
            }
            orquestador.cola_trabajos.put(trabajo)
        
        return jsonify({'message': f'{len(trabajos)} trabajos agregados'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/trabajos_en_proceso')
def obtener_trabajos_en_proceso():
    """Endpoint para obtener solo los trabajos en proceso"""
    trabajos_con_tiempo = {}
    tiempo_actual = time.time()
    
    for trabajo_id, info in orquestador.trabajos_en_proceso.items():
        trabajos_con_tiempo[trabajo_id] = {
            'esclavo_id': info['esclavo_id'],
            'trabajo': info['trabajo'],
            'tiempo_ejecutando': tiempo_actual - info['inicio']
        }
    
    return jsonify(trabajos_con_tiempo)

@app.route('/ultimos_resultados')
def obtener_ultimos_resultados():
    """Endpoint para obtener los últimos resultados"""
    # Obtener últimos 10 resultados
    ultimos_resultados = dict(list(orquestador.resultados.items())[-10:])
    return jsonify(ultimos_resultados)

if __name__ == '__main__':
    print("🎯 Iniciando Orquestador Koch...")
    print(f"Esclavos configurados: {list(orquestador.esclavos.keys())}")
    app.run(host='0.0.0.0', port=8000, debug=True)
