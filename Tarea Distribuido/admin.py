#!/usr/bin/env python3
"""
Script de administración para el sistema Koch distribuido
"""

import requests
import json
import time
import argparse
import sys

class AdminSistemaKoch:
    def __init__(self, orquestador_url):
        self.orquestador_url = orquestador_url.rstrip('/')
    
    def estado_sistema(self):
        """Muestra el estado completo del sistema"""
        try:
            response = requests.get(f"{self.orquestador_url}/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                
                print("🎯 Estado del Sistema Koch Distribuido")
                print("=" * 50)
                
                # Estadísticas generales
                stats = status['estadisticas']
                print(f"📊 Estadísticas Generales:")
                print(f"   Trabajos completados: {stats['trabajos_completados']}")
                print(f"   Trabajos fallidos: {stats['trabajos_fallidos']}")
                print(f"   Tiempo total procesamiento: {stats['tiempo_total_procesamiento']:.2f}s")
                print(f"   Trabajos en cola: {status['cola_trabajos']}")
                print(f"   Trabajos en proceso: {status['trabajos_en_proceso']}")
                
                # Estado de esclavos
                print(f"\n🤖 Estado de Esclavos:")
                esclavos = status['esclavos']
                for esclavo_id, info in esclavos.items():
                    estado_emoji = "✅" if info['status'] == 'activo' else "❌"
                    print(f"   {estado_emoji} {esclavo_id}: {info['status']}")
                    print(f"      URL: {info['url']}")
                    print(f"      Trabajos ejecutados: {info['trabajos_ejecutados']}")
                    print(f"      Tiempo total: {info['tiempo_total']:.2f}s")
                    if info['last_ping']:
                        print(f"      Último ping: {info['last_ping']}")
                
                return True
            else:
                print(f"❌ Error obteniendo estado: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Error de conexión: {e}")
            return False
    
    def agregar_trabajo_individual(self, algoritmo, iteraciones, size=3.0, dpi=300):
        """Agrega un trabajo individual al sistema"""
        trabajo = {
            "algoritmo": algoritmo,
            "parametros": {
                "iteraciones": iteraciones,
                "size": size,
                "dpi": dpi
            }
        }
        
        try:
            response = requests.post(
                f"{self.orquestador_url}/trabajo",
                json=trabajo,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                resultado = response.json()
                print(f"✅ Trabajo agregado: {resultado['id']}")
                return True
            else:
                print(f"❌ Error agregando trabajo: {response.status_code} - {response.text}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Error de conexión: {e}")
            return False
    
    def generar_trabajos_comparacion(self, iteraciones_list=None):
        """Genera trabajos para comparar Classic vs JAX Koch"""
        if iteraciones_list is None:
            iteraciones_list = [3, 4, 5, 6, 7, 8]
        
        print(f"🔄 Generando trabajos de comparación para iteraciones: {iteraciones_list}")
        print("💡 Estos trabajos se ejecutarán en PARALELO en diferentes esclavos")
        
        trabajos_agregados = 0
        for iteraciones in iteraciones_list:
            # Classic Koch
            if self.agregar_trabajo_individual("ClassicKoch", iteraciones):
                trabajos_agregados += 1
            
            # JAX Koch
            if self.agregar_trabajo_individual("JAXKoch", iteraciones):
                trabajos_agregados += 1
        
        print(f"✅ {trabajos_agregados} trabajos agregados para comparación paralela")
        print("👀 Observa el dashboard para ver la ejecución en paralelo")
        return trabajos_agregados > 0
    
    def generar_trabajos_stress_test(self, cantidad=10):
        """Genera múltiples trabajos para probar el paralelismo"""
        print(f"🚀 Generando {cantidad} trabajos para stress test de paralelismo...")
        
        import random
        algoritmos = ["ClassicKoch", "JAXKoch"]
        trabajos_agregados = 0
        
        for i in range(cantidad):
            algoritmo = random.choice(algoritmos)
            iteraciones = random.randint(3, 6)
            size = round(random.uniform(2.0, 4.0), 1)
            dpi = random.choice([150, 200, 300])
            
            if self.agregar_trabajo_individual(algoritmo, iteraciones, size, dpi):
                trabajos_agregados += 1
                print(f"   📦 Trabajo {i+1}: {algoritmo} iter={iteraciones}")
        
        print(f"✅ {trabajos_agregados} trabajos agregados para stress test")
        return trabajos_agregados > 0
    
    def obtener_resultado(self, trabajo_id):
        """Obtiene el resultado de un trabajo específico"""
        try:
            response = requests.get(f"{self.orquestador_url}/resultado/{trabajo_id}", timeout=10)
            if response.status_code == 200:
                resultado = response.json()
                
                if 'error' in resultado:
                    print(f"❌ Trabajo {trabajo_id}: {resultado['error']}")
                elif 'status' in resultado and resultado['status'] == 'en_proceso':
                    print(f"🔄 Trabajo {trabajo_id}: En proceso")
                else:
                    print(f"✅ Trabajo {trabajo_id}: Completado")
                    print(f"   Esclavo: {resultado.get('esclavo_id', 'N/A')}")
                    print(f"   Tiempo: {resultado.get('tiempo_ejecucion', 0):.2f}s")
                    print(f"   Timestamp: {resultado.get('timestamp', 'N/A')}")
                
                return resultado
            else:
                print(f"❌ Error obteniendo resultado: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            print(f"❌ Error de conexión: {e}")
            return None
    
    def monitorear_continuo(self, intervalo=5):
        """Monitorea el sistema continuamente"""
        print(f"👀 Iniciando monitoreo continuo (cada {intervalo}s)")
        print("Presiona Ctrl+C para detener...")
        
        try:
            while True:
                print("\n" + "="*30 + f" {time.strftime('%H:%M:%S')} " + "="*30)
                self.estado_sistema()
                time.sleep(intervalo)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoreo detenido por el usuario")

def main():
    parser = argparse.ArgumentParser(description='Administrador del sistema Koch distribuido')
    parser.add_argument('--orquestador', default='http://localhost:8000',
                       help='URL del orquestador (default: http://localhost:8000)')
    
    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponibles')
    
    # Comando: estado
    subparsers.add_parser('estado', help='Mostrar estado del sistema')
    
    # Comando: trabajo
    trabajo_parser = subparsers.add_parser('trabajo', help='Agregar trabajo individual')
    trabajo_parser.add_argument('algoritmo', choices=['ClassicKoch', 'JAXKoch'],
                               help='Algoritmo a ejecutar')
    trabajo_parser.add_argument('iteraciones', type=int, help='Número de iteraciones')
    trabajo_parser.add_argument('--size', type=float, default=3.0, help='Tamaño (default: 3.0)')
    trabajo_parser.add_argument('--dpi', type=int, default=300, help='DPI (default: 300)')
    
    # Comando: comparar
    comparar_parser = subparsers.add_parser('comparar', help='Generar trabajos de comparación paralela')
    comparar_parser.add_argument('--iteraciones', nargs='+', type=int,
                                default=[3, 4, 5, 6, 7, 8],
                                help='Lista de iteraciones a comparar')
    
    # Comando: stress
    stress_parser = subparsers.add_parser('stress', help='Stress test de paralelismo')
    stress_parser.add_argument('--cantidad', type=int, default=10,
                              help='Cantidad de trabajos a generar (default: 10)')
    
    # Comando: resultado
    resultado_parser = subparsers.add_parser('resultado', help='Obtener resultado de trabajo')
    resultado_parser.add_argument('trabajo_id', help='ID del trabajo')
    
    # Comando: monitorear
    monitorear_parser = subparsers.add_parser('monitorear', help='Monitoreo continuo')
    monitorear_parser.add_argument('--intervalo', type=int, default=5,
                                  help='Intervalo en segundos (default: 5)')
    
    args = parser.parse_args()
    
    if not args.comando:
        parser.print_help()
        sys.exit(1)
    
    admin = AdminSistemaKoch(args.orquestador)
    
    # Ejecutar comando
    if args.comando == 'estado':
        admin.estado_sistema()
    
    elif args.comando == 'trabajo':
        admin.agregar_trabajo_individual(args.algoritmo, args.iteraciones, 
                                       args.size, args.dpi)
    
    elif args.comando == 'comparar':
        admin.generar_trabajos_comparacion(args.iteraciones)
    
    elif args.comando == 'stress':
        admin.generar_trabajos_stress_test(args.cantidad)
    
    elif args.comando == 'resultado':
        admin.obtener_resultado(args.trabajo_id)
    
    elif args.comando == 'monitorear':
        admin.monitorear_continuo(args.intervalo)

if __name__ == '__main__':
    main()
