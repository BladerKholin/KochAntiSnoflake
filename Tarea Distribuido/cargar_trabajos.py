#!/usr/bin/env python3
"""
Script para cargar trabajos automáticamente al orquestador
"""

import requests
import json
import time
import sys
import argparse

def cargar_trabajos_desde_archivo(orquestador_url, archivo_json):
    """Carga trabajos desde un archivo JSON al orquestador"""
    try:
        with open(archivo_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        trabajos = data.get('trabajos', [])
        
        if not trabajos:
            print("❌ No se encontraron trabajos en el archivo")
            return False
        
        print(f"📦 Cargando {len(trabajos)} trabajos al orquestador...")
        
        # Enviar trabajos al orquestador
        response = requests.post(
            f"{orquestador_url}/cargar_trabajos",
            json={'trabajos': trabajos},
            timeout=30
        )
        
        if response.status_code == 200:
            resultado = response.json()
            print(f"✅ {resultado['message']}")
            return True
        else:
            print(f"❌ Error del servidor: {response.status_code} - {response.text}")
            return False
            
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {archivo_json}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error al leer JSON: {e}")
        return False
    except requests.RequestException as e:
        print(f"❌ Error de conexión: {e}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def verificar_estado_orquestador(orquestador_url):
    """Verifica que el orquestador esté disponible"""
    try:
        response = requests.get(f"{orquestador_url}/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Orquestador disponible")
            print(f"   Esclavos activos: {sum(1 for e in status['esclavos'].values() if e['status'] == 'activo')}")
            print(f"   Trabajos en cola: {status['cola_trabajos']}")
            print(f"   Trabajos en proceso: {status['trabajos_en_proceso']}")
            return True
        else:
            print(f"❌ Orquestador no responde correctamente: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ No se puede conectar al orquestador: {e}")
        return False

def monitorear_progreso(orquestador_url, intervalo=5, timeout=300):
    """Monitorea el progreso de los trabajos"""
    print(f"👀 Monitoreando progreso (cada {intervalo}s, timeout {timeout}s)...")
    
    inicio = time.time()
    trabajos_completados_prev = 0
    
    while time.time() - inicio < timeout:
        try:
            response = requests.get(f"{orquestador_url}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                
                trabajos_completados = status['estadisticas']['trabajos_completados']
                trabajos_fallidos = status['estadisticas']['trabajos_fallidos']
                cola_trabajos = status['cola_trabajos']
                trabajos_en_proceso = status['trabajos_en_proceso']
                
                # Mostrar progreso si hay cambios
                if trabajos_completados != trabajos_completados_prev:
                    print(f"📊 Completados: {trabajos_completados}, Fallidos: {trabajos_fallidos}, "
                          f"En cola: {cola_trabajos}, En proceso: {trabajos_en_proceso}")
                    trabajos_completados_prev = trabajos_completados
                
                # Si no hay trabajos pendientes, terminar
                if cola_trabajos == 0 and trabajos_en_proceso == 0:
                    print("✅ Todos los trabajos han sido procesados")
                    return True
                    
            time.sleep(intervalo)
            
        except requests.RequestException:
            print("⚠️ Error conectando al orquestador durante monitoreo")
            time.sleep(intervalo)
    
    print("⏰ Timeout alcanzado durante el monitoreo")
    return False

def main():
    parser = argparse.ArgumentParser(description='Cargador de trabajos para el sistema Koch distribuido')
    parser.add_argument('--orquestador', default='http://localhost:8000', 
                       help='URL del orquestador (default: http://localhost:8000)')
    parser.add_argument('--archivo', default='config/pruebas.json',
                       help='Archivo JSON con los trabajos (default: config/pruebas.json)')
    parser.add_argument('--monitorear', action='store_true',
                       help='Monitorear el progreso después de cargar')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout para monitoreo en segundos (default: 300)')
    
    args = parser.parse_args()
    
    print("🎯 Cargador de Trabajos - Sistema Koch Distribuido")
    print("=" * 50)
    
    # Verificar estado del orquestador
    if not verificar_estado_orquestador(args.orquestador):
        print("❌ No se puede continuar sin conexión al orquestador")
        sys.exit(1)
    
    # Cargar trabajos
    if not cargar_trabajos_desde_archivo(args.orquestador, args.archivo):
        print("❌ Error cargando trabajos")
        sys.exit(1)
    
    # Monitorear si se solicita
    if args.monitorear:
        print()
        monitorear_progreso(args.orquestador, timeout=args.timeout)
    
    print("\n🎉 Proceso completado")

if __name__ == '__main__':
    main()
