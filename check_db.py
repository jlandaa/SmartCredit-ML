import pandas as pd
from sqlalchemy import create_engine

def verificar_base_datos():
    print("🔍 Iniciando validación de la base de datos...\n")
    
    # 1. Conectamos al mismo motor SQLite que usa tu app
    engine = create_engine("sqlite:///historial_evaluaciones.db")
    
    try:
        # 2. Leemos la tabla completa usando Pandas
        df_historial = pd.read_sql_table('historial_predicciones', con=engine)
        
        # 3. Mostramos métricas clave
        print("✅ CONEXIÓN EXITOSA Y TABLA ENCONTRADA")
        print("-" * 40)
        print(f"Total de evaluaciones registradas: {len(df_historial)}")
        print("-" * 40)
        
        print("\n📊 ÚLTIMOS 3 REGISTROS GUARDADOS:")
        # Mostramos las últimas 3 filas para verificar los datos más recientes
        print(df_historial.tail(3).to_string(index=False))
        
        print("\n🛠️ ESTRUCTURA DE COLUMNAS (Ideal para mapear en LookML):")
        # Esto te servirá exactamente para saber qué tipo de dato usar en Looker (number, string, time)
        for col, dtype in df_historial.dtypes.items():
            print(f"- {col}: {dtype}")
            
    except ValueError:
        print("⚠️ AVISO: La tabla 'historial_predicciones' no existe.")
        print("👉 Esto suele ocurrir si aún no has realizado ninguna evaluación exitosa en tu app de Streamlit.")
        print("Ve a tu navegador, aprueba o rechaza un crédito y vuelve a correr este script.")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    verificar_base_datos()
