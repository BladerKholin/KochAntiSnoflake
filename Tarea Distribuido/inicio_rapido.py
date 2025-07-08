#!/usr/bin/env python3
"""
Script de inicio rápido para el sistema Koch distribuido
"""

import subprocess
import time
import sys
import requests
import os

def ejecutar_comando(comando, mostrar_output=True):
    """Ejecuta un comando y retorna el resultado"""
    try:
        if mostrar_output:
            print(f"▶️ Ejecutando: {comando}")
        
        result = subprocess.run(comando, shell=True, capture_output=True, text=True)
        
        if mostrar_output and result.stdout:
            print(result.stdout)
        
        if result.returncode != 0:
            if mostrar_output and result.stderr:
                print(f"❌ Error: {result.stderr}")
            return False
        
        return True
    except Exception as e:
        if mostrar_output:
            print(f"❌ Excepción: {e}")
        return False

def verificar_docker():
    """Verifica que Docker esté disponible"""
    print("🐳 Verificando Docker...")
    if not ejecutar_comando("docker --version", False):
        print("❌ Docker no está disponible. Instale Docker primero.")
        return False
    
    if not ejecutar_comando("docker-compose --version", False):
        print("❌ Docker Compose no está disponible. Instale Docker Compose primero.")
        return False
    
    print("✅ Docker y Docker Compose están disponibles")
    return True

def construir_sistema():
    """Construye las imágenes Docker"""
    print("\n🔨 Construyendo imágenes Docker...")
    return ejecutar_comando("docker-compose build --no-cache")

def levantar_sistema():
    """Levanta todos los servicios"""
    print("\n🚀 Levantando servicios...")
    return ejecutar_comando("docker-compose up -d")

def esperar_servicios():
    """Espera a que los servicios estén listos"""
    print("\n⏳ Esperando a que los servicios estén listos...")
    
    servicios = [
        ("Orquestador", "http://localhost:8000/status"),
        ("Esclavo 1", "http://localhost:5001/status"),
        ("Esclavo 2", "http://localhost:5002/status"),
        ("Esclavo 3", "http://localhost:5003/status")
    ]
    
    max_intentos = 30
    intentos = 0
    
    while intentos < max_intentos:
        servicios_listos = 0
        
        for nombre, url in servicios:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    servicios_listos += 1
                    print(f"✅ {nombre} listo")
                else:
                    print(f"⏳ {nombre} iniciando...")
            except requests.RequestException:
                print(f"⏳ {nombre} no disponible aún...")
        
        if servicios_listos == len(servicios):
            print("🎉 Todos los servicios están listos!")
            return True
        
        intentos += 1
        time.sleep(2)
    
    print("❌ Timeout esperando servicios")
    return False

def cargar_trabajos_demo():
    """Carga trabajos de demostración"""
    print("\n📦 Cargando trabajos de demostración...")
    
    if os.path.exists("cargar_trabajos.py"):
        return ejecutar_comando("python cargar_trabajos.py --archivo config/pruebas.json")
    else:
        print("⚠️ Script cargar_trabajos.py no encontrado, saltando...")
        return True

def mostrar_informacion():
    """Muestra información del sistema"""
    print("\n" + "="*60)
    print("🎯 SISTEMA KOCH DISTRIBUIDO - INICIADO CORRECTAMENTE")
    print("="*60)
    print("🌐 Dashboard Principal: http://localhost:8000")
    print("🤖 Esclavos disponibles:")
    print("   • Esclavo 1: http://localhost:5001")
    print("   • Esclavo 2: http://localhost:5002") 
    print("   • Esclavo 3: http://localhost:5003")
    print("\n📋 Comandos útiles:")
    print("   • Ver estado: python admin.py estado")
    print("   • Monitorear: python admin.py monitorear")
    print("   • Agregar trabajo: python admin.py trabajo ClassicKoch 5")
    print("   • Comparar algoritmos: python admin.py comparar")
    print("\n🐳 Gestión Docker:")
    print("   • Ver logs: docker-compose logs -f")
    print("   • Parar sistema: docker-compose down")
    print("   • Estado contenedores: docker-compose ps")
    print("="*60)

def main():
    print("🎯 Inicio Rápido - Sistema Koch Distribuido")
    print("="*50)
    
    # Cambiar al directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Directorio de trabajo: {script_dir}")
    
    # Verificaciones previas
    if not verificar_docker():
        sys.exit(1)
    
    # Construcción y despliegue
    if not construir_sistema():
        print("❌ Error construyendo el sistema")
        sys.exit(1)
    
    if not levantar_sistema():
        print("❌ Error levantando el sistema")
        sys.exit(1)
    
    # Esperar a que esté listo
    if not esperar_servicios():
        print("❌ Los servicios no iniciaron correctamente")
        print("💡 Intenta revisar los logs con: docker-compose logs")
        sys.exit(1)
    
    # Cargar trabajos demo (opcional)
    cargar_trabajos_demo()
    
    # Mostrar información final
    mostrar_informacion()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)
