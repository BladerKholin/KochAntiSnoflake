#!/usr/bin/env python3
"""
Script simplificado para enviar todos los casos de prueba al orquestador
"""

import requests
import json
import time
import argparse
import sys
import os

def cargar_casos_prueba(archivo_casos):
    """Carga los casos de prueba desde el archivo JSON"""
    if not os.path.exists(archivo_casos):
        print(f"❌ No se encontró el archivo: {archivo_casos}")
        return []
    
    try:
        with open(archivo_casos, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('casos_prueba', [])
    except Exception as e:
        print(f"❌ Error cargando casos: {e}")
        return []

def enviar_trabajo(orquestador_url, caso, numero, total):
    """Envía un trabajo al orquestador"""
    trabajo = {
        "algoritmo": caso["algoritmo"],
        "parametros": caso["parametros"]
    }
    
    print(f"[{numero}/{total}] 📦 {caso['nombre']}")
    
    try:
        response = requests.post(
            f"{orquestador_url}/trabajo",
            json=trabajo,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            resultado = response.json()
            trabajo_id = resultado.get("id")
            print(f"         ✅ Enviado con ID: {trabajo_id}")
            return True
        else:
            print(f"         ❌ Error HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"         ❌ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Envía todos los casos de prueba al orquestador')
    parser.add_argument('--orquestador', default='http://localhost:8000',
                       help='URL del orquestador (default: http://localhost:8000)')
    parser.add_argument('--casos', default='test_cases.json',
                       help='Archivo de casos de prueba (default: test_cases.json)')
    parser.add_argument('--intervalo', type=float, default=0.1,
                       help='Intervalo entre envíos en segundos (default: 0.1)')
    
    args = parser.parse_args()
    
    # Verificar conectividad
    try:
        response = requests.get(f"{args.orquestador}/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ Orquestador no disponible en {args.orquestador}")
            sys.exit(1)
        print(f"✅ Orquestador disponible en {args.orquestador}")
    except Exception as e:
        print(f"❌ No se puede conectar al orquestador: {e}")
        sys.exit(1)
    
    # Cargar casos de prueba
    casos = cargar_casos_prueba(args.casos)
    if not casos:
        print("❌ No se pudieron cargar los casos de prueba")
        sys.exit(1)
    
    print(f"✅ Cargados {len(casos)} casos de prueba")
    print(f"🚀 Enviando todos los trabajos con intervalo de {args.intervalo}s")
    print("=" * 60)
    
    # Enviar todos los trabajos
    exitosos = 0
    fallidos = 0
    
    for i, caso in enumerate(casos, 1):
        if enviar_trabajo(args.orquestador, caso, i, len(casos)):
            exitosos += 1
        else:
            fallidos += 1
        
        # Esperar antes del siguiente (excepto el último)
        if i < len(casos):
            time.sleep(args.intervalo)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    print(f"Total enviados: {exitosos}")
    print(f"Fallidos: {fallidos}")
    print(f"Tasa de éxito: {(exitosos/len(casos)*100):.1f}%")
    print(f"\n✅ Todos los trabajos enviados! Revisa el dashboard del orquestador.")

if __name__ == '__main__':
    main()
