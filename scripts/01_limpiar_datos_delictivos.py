import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')
 
 
# Configuracion
 
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
# Bounding box de Lima Metropolitana + Callao
LIMA_BBOX = {
    "lat_min": -12.52,  # Sur: Pucusana
    "lat_max": -11.57,  # Norte: Ancón / Ventanilla
    "lon_min": -77.22,  # Oeste: Callao / La Punta
    "lon_max": -76.62   # Este: Chosica
}
 
# FILTROS CLAVE
 
# Combinaciones departamento + provincia que aceptamos
# Lima Metropolitana + Callao = área metropolitana funcional
FILTRO_UBICACION = [
    ("LIMA", "LIMA"),                    # Lima provincia de Lima
    ("LIMA METROPOLITANA", "LIMA"),       # Variante de nombre
    ("CALLAO", "CALLAO"),                # Provincia Constitucional del Callao
]
 
# Subtipos de delito que nos interesan (delitos patrimoniales)
SUBTIPOS_INTERES = ["ROBO", "HURTO"]
 
# Estados de coordenadas válidos
ESTADOS_VALIDOS = [1, 2]
 
# Columnas que necesitamos en el output final
COLUMNAS_OUTPUT = [
    'id_dgc_03', 'lat_hecho', 'long_hecho',
    'departamento_hecho', 'provincia_hecho',
    'distrito_hecho', 'ubigeo_hecho_delito',
    'direccion_hecho', 'fecha_hora_hecho',
    'turno_hecho', 'año_hecho', 'mes_hecho', 'dia_hecho',
    'subtipo_hecho', 'modalidad_hecho',
    'estado', 'fuente'
]
 
 
# Funciones
 
def buscar_archivos(directorio: Path) -> list:
    """Busca todos los .csv y .xlsx recursivamente."""
    archivos = []
    for ext in ["*.csv", "*.xlsx", "*.xls"]:
        archivos.extend(directorio.rglob(ext))
    return sorted(archivos)
 
 
def cargar_archivo(archivo: Path) -> pd.DataFrame:
    """Carga un archivo individual (CSV o Excel)."""
    try:
        if archivo.suffix == ".xlsx" or archivo.suffix == ".xls":
            df = pd.read_excel(archivo, engine="openpyxl")
        else:
            # Intentar con diferentes separadores y encodings
            for sep in [",", "\t", ";"]:
                for encoding in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        df = pd.read_csv(archivo, encoding=encoding, sep=sep)
                        # Verificar que parseó bien (más de 5 columnas)
                        if len(df.columns) > 5:
                            return df
                    except (UnicodeDecodeError, pd.errors.ParserError):
                        continue
            # Último intento
            df = pd.read_csv(archivo, encoding="latin-1")
        return df
    except Exception as e:
        print(f"    Error leyendo {archivo.name}: {e}")
        return pd.DataFrame()
 
 
def cargar_todos(directorio: Path) -> pd.DataFrame:
    """Carga y une todos los archivos del directorio."""
    archivos = buscar_archivos(directorio)
    
    if not archivos:
        print(f"  No se encontraron archivos en {directorio}")
        print(f"    Coloca tus CSVs descargados del MININTER ahí")
        return pd.DataFrame()
    
    print(f"  Archivos encontrados: {len(archivos)}")
    
    frames = []
    total_registros = 0
    
    for archivo in archivos:
        df = cargar_archivo(archivo)
        if not df.empty:
            # Normalizar nombres de columnas
            df.columns = df.columns.str.strip().str.lower()
            total_registros += len(df)
            
            # Mostrar info del archivo
            ruta_rel = archivo.relative_to(directorio)
            print(f"    OK {ruta_rel}: {len(df):,} registros")
            frames.append(df)
        else:
            print(f"    ERROR {archivo.name}: no se pudo leer")
    
    if not frames:
        return pd.DataFrame()
    
    print(f"\n  Total registros cargados: {total_registros:,}")
    
    # Unir todo (outer join para no perder datos si hay columnas diferentes)
    df_union = pd.concat(frames, ignore_index=True, join='outer')
    
    # Verificar duplicados por concatenación
    if "id_dgc_03" in df_union.columns:
        antes = len(df_union)
        df_union = df_union.drop_duplicates(subset=["id_dgc_03"])
        dups = antes - len(df_union)
        if dups > 0:
            print(f"  Duplicados entre archivos eliminados: {dups:,}")
    
    print(f"  Registros únicos totales: {len(df_union):,}")
    return df_union
 
 
def filtrar_lima(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra solo registros de Lima Metropolitana + Callao."""
    print(f"\n{'-'*55}")
    print(f"  FILTRO 1: Solo Lima Metropolitana + Callao")
    print(f"{'-'*55}")
    
    antes = len(df)
    
    # Normalizar texto
    if "departamento_hecho" in df.columns:
        df["departamento_hecho"] = df["departamento_hecho"].astype(str).str.strip().str.upper()
    if "provincia_hecho" in df.columns:
        df["provincia_hecho"] = df["provincia_hecho"].astype(str).str.strip().str.upper()
    
    # Mostrar distribución de departamentos
    if "departamento_hecho" in df.columns:
        dept_counts = df["departamento_hecho"].value_counts().head(10)
        dept_validos = set(d for d, p in FILTRO_UBICACION)
        print(f"  Top departamentos en los datos:")
        for dept, count in dept_counts.items():
            marca = "*" if dept in dept_validos else " "
            print(f"    {marca} {dept}: {count:,}")
    
    # Filtrar por combinaciones válidas de departamento + provincia
    if "departamento_hecho" in df.columns and "provincia_hecho" in df.columns:
        mask = pd.Series(False, index=df.index)
        for dept, prov in FILTRO_UBICACION:
            mask |= (df["departamento_hecho"] == dept) & (df["provincia_hecho"] == prov)
        df = df[mask].copy()
    
    print(f"  {antes:,} -> {len(df):,} (descartados: {antes - len(df):,})")
    
    # Mostrar desglose Lima vs Callao
    if "departamento_hecho" in df.columns and len(df) > 0:
        desglose = df["departamento_hecho"].value_counts()
        print(f"  Desglose:")
        for dept, count in desglose.items():
            print(f"    {dept}: {count:,}")
    
    return df
 
 
def filtrar_delitos_patrimoniales(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra solo ROBO y HURTO."""
    print(f"\n{'-'*55}")
    print(f"  FILTRO 2: Solo delitos patrimoniales (ROBO + HURTO)")
    print(f"{'-'*55}")
    
    antes = len(df)
    
    if "subtipo_hecho" in df.columns:
        df["subtipo_hecho"] = df["subtipo_hecho"].astype(str).str.strip().str.upper()
        
        # Mostrar distribución de subtipos en Lima
        subtipo_counts = df["subtipo_hecho"].value_counts().head(15)
        print(f"  Subtipos de delito en Lima (top 15):")
        for subtipo, count in subtipo_counts.items():
            marca = "*" if subtipo in SUBTIPOS_INTERES else " "
            print(f"    {marca} {subtipo}: {count:,}")
        
        df = df[df["subtipo_hecho"].isin(SUBTIPOS_INTERES)].copy()
    
    print(f"  {antes:,} -> {len(df):,} (descartados: {antes - len(df):,})")
    
    # Mostrar modalidades resultantes
    if "modalidad_hecho" in df.columns and len(df) > 0:
        print(f"\n  Modalidades incluidas:")
        for modal, count in df["modalidad_hecho"].value_counts().head(10).items():
            print(f"    {modal}: {count:,}")
    
    return df
 
 
def filtrar_coordenadas(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra por estado de coordenadas y valida geográficamente."""
    print(f"\n{'-'*55}")
    print(f"  FILTRO 3: Solo coordenadas válidas")
    print(f"{'-'*55}")
    
    antes = len(df)
    
    # Filtrar por estado
    if "estado" in df.columns:
        df["estado"] = pd.to_numeric(df["estado"], errors="coerce")
        
        estado_counts = df["estado"].value_counts().sort_index()
        print(f"  Estados de coordenadas:")
        for estado_val, count in estado_counts.items():
            pct = count / len(df) * 100
            if int(estado_val) == 1:
                etiq = "COORDENADA OK"
            elif int(estado_val) == 2:
                etiq = "VERIFICAR UBIGEO"
            elif int(estado_val) == 3:
                etiq = "GEO FORZADA -> comisaria (NO SIRVE)"
            else:
                etiq = "DESCONOCIDO"
            marca = "OK" if int(estado_val) in ESTADOS_VALIDOS else "NO"
            print(f"    {marca} Estado {int(estado_val)}: {count:>6,} ({pct:>5.1f}%) - {etiq}")
        
        df = df[df["estado"].isin(ESTADOS_VALIDOS)].copy()
        print(f"  Después de filtro estado: {len(df):,}")
    
    if len(df) == 0:
        return df
    
    # Validar coordenadas numericas
    for col in ["lat_hecho", "long_hecho"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["lat_hecho", "long_hecho"])
    
    # Filtrar dentro del bbox de Lima
    antes_bbox = len(df)
    df = df[
        (df["lat_hecho"] >= LIMA_BBOX["lat_min"]) &
        (df["lat_hecho"] <= LIMA_BBOX["lat_max"]) &
        (df["long_hecho"] >= LIMA_BBOX["lon_min"]) &
        (df["long_hecho"] <= LIMA_BBOX["lon_max"])
    ].copy()
    
    fuera = antes_bbox - len(df)
    if fuera > 0:
        print(f"  Fuera del bounding box de Lima: {fuera}")
    
    print(f"  {antes:,} -> {len(df):,} (descartados total: {antes - len(df):,})")
    return df
 
 
def generar_reporte(df: pd.DataFrame):
    """Genera reporte final con datos para CONIDA."""
    
    print(f"\n{'='*60}")
    print(f"  REPORTE FINAL")
    print(f"{'='*60}")
    
    print(f"\n  Total registros usables: {len(df):,}")
    
    # Por subtipo
    print(f"\n  Por tipo de delito:")
    for tipo, count in df["subtipo_hecho"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {tipo}: {count:,} ({pct:.1f}%)")
    
    # Por distrito
    if "distrito_hecho" in df.columns:
        print(f"\n  Top 15 distritos:")
        for i, (dist, count) in enumerate(df["distrito_hecho"].value_counts().head(15).items(), 1):
            print(f"    {i:>2}. {dist}: {count:,}")
        print(f"    ... ({df['distrito_hecho'].nunique()} distritos en total)")
    
    # Por mes
    if "mes_hecho" in df.columns:
        print(f"\n  Por mes:")
        meses_nombre = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                       7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
        for mes in sorted(df["mes_hecho"].dropna().unique()):
            count = len(df[df["mes_hecho"] == mes])
            nombre = meses_nombre.get(int(mes), f"?{mes}")
            print(f"    {nombre}: {count:,}")
    
    # Por turno
    if "turno_hecho" in df.columns:
        print(f"\n  Por turno:")
        for turno, count in df["turno_hecho"].value_counts().items():
            print(f"    {turno}: {count:,}")
    
    # Por modalidad
    if "modalidad_hecho" in df.columns:
        print(f"\n  Top 10 modalidades:")
        for modal, count in df["modalidad_hecho"].value_counts().head(10).items():
            print(f"    {modal}: {count:,}")
    
    # Coordenadas
    print(f"\n  Rango de coordenadas:")
    print(f"    Lat:  {df['lat_hecho'].min():.6f} a {df['lat_hecho'].max():.6f}")
    print(f"    Long: {df['long_hecho'].min():.6f} a {df['long_hecho'].max():.6f}")
    
    # DATOS PARA CONIDA
    print(f"\n{'='*60}")
    print(f"  DATOS PARA SOLICITUD DE IMÁGENES A CONIDA")
    print(f"{'='*60}")
    print(f"  Satélite solicitado: PerúSAT-1")
    print(f"  Resolución: Pancromática (0.7m) + Multiespectral (2.8m)")
    print(f"  Formato: GeoTIFF ortorectificado, EPSG:4326")
    print(f"")
    print(f"  Área de interés: Lima Metropolitana + Callao")
    print(f"    Esquina NW: ({df['lat_hecho'].max():.4f}, {df['long_hecho'].min():.4f})")
    print(f"    Esquina SE: ({df['lat_hecho'].min():.4f}, {df['long_hecho'].max():.4f})")
    
    if "año_hecho" in df.columns and "mes_hecho" in df.columns:
        año = int(df["año_hecho"].mode().iloc[0])
        min_mes = int(df["mes_hecho"].min())
        max_mes = int(df["mes_hecho"].max())
        meses_full = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",
                     6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",
                     10:"Octubre",11:"Noviembre",12:"Diciembre"}
        print(f"  Periodo datos delictivos: {meses_full[min_mes]} - {meses_full[max_mes]} {año}")
        print(f"  Periodo recomendado imágenes: misma ventana temporal")
        print(f"    (Preferir cielo despejado: Nov-Abr en Lima)")
    
    print(f"  Total puntos delictivos: {len(df):,}")
    print(f"  Distritos con datos: {df['distrito_hecho'].nunique()}")
 
 
# EJECUCION PRINCIPAL
if __name__ == "__main__":
    
    print("=======================================================")
    print("  LIMPIEZA DE DATOS DELICTIVOS - MININTER")
    print("  Formato: DELITOS TOTAL (todos los tipos mezclados)")
    print("  Output: Solo ROBO + HURTO en Lima + Callao con coords")
    print("=======================================================")
    
    # Paso 1: Cargar todos los archivos
    print(f"\n{'='*55}")
    print(f"  CARGANDO ARCHIVOS")
    print(f"{'='*55}")
    df = cargar_todos(RAW_DIR)
    
    if df.empty:
        print("\nNo se cargaron datos. Verifica tus archivos.")
        sys.exit(1)
    
    # Mostrar columnas disponibles
    print(f"\n  Columnas detectadas ({len(df.columns)}):")
    for col in sorted(df.columns):
        print(f"    {col}")
    
    # Paso 2: Filtrar Lima
    df = filtrar_lima(df)
    if df.empty:
        print("\nNo quedaron registros de Lima Metropolitana.")
        sys.exit(1)
    
    # Paso 3: Filtrar ROBO + HURTO
    df = filtrar_delitos_patrimoniales(df)
    if df.empty:
        print("\nNo quedaron registros de ROBO/HURTO.")
        sys.exit(1)
    
    # Paso 4: Filtrar coordenadas
    df = filtrar_coordenadas(df)
    if df.empty:
        print("\nNo quedaron registros con coordenadas validas.")
        print("  Esto puede pasar si todos tus datos son estado 3.")
        print("  Verifica la fuente de descarga del MININTER.")
        sys.exit(1)
    
    # Paso 5: Seleccionar columnas y guardar
    print(f"\n{'='*55}")
    print(f"  GUARDANDO RESULTADOS")
    print(f"{'='*55}")
    
    # Seleccionar solo columnas necesarias (las que existan)
    cols_disponibles = [c for c in COLUMNAS_OUTPUT if c in df.columns]
    df_final = df[cols_disponibles].copy()
    
    # CSV
    csv_path = OUTPUT_DIR / "delitos_lima_limpio.csv"
    df_final.to_csv(csv_path, index=False, encoding="utf-8")
    size_kb = csv_path.stat().st_size / 1024
    print(f"  OK CSV: {csv_path}")
    print(f"    {len(df_final):,} registros, {size_kb:.0f} KB")
    
    # GeoJSON para QGIS
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        
        geometry = [Point(lon, lat) for lon, lat in 
                   zip(df_final["long_hecho"], df_final["lat_hecho"])]
        gdf = gpd.GeoDataFrame(df_final, geometry=geometry, crs="EPSG:4326")
        
        geojson_path = OUTPUT_DIR / "delitos_lima_limpio.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")
        size_kb_geo = geojson_path.stat().st_size / 1024
        print(f"  OK GeoJSON: {geojson_path}")
        print(f"    {size_kb_geo:.0f} KB - abre directamente en QGIS")
    except ImportError:
        print("  Aviso: GeoPandas no instalado (pip install geopandas)")
        print("    CSV guardado correctamente, GeoJSON es opcional")
    except Exception as e:
        print(f"  Error en GeoJSON: {e}")
    
    # Reporte
    generar_reporte(df_final)
    
    # Proximos pasos
    print(f"\n{'='*60}")
    print("  QUE SIGUE")
    print(f"{'='*60}")
    print(f"  1. Abre delitos_lima_limpio.geojson en QGIS")
    print(f"     - Verifica visualmente que los puntos esten bien")
    print(f"  2. Copia los 'DATOS PARA CONIDA' de arriba")
    print(f"     - Envialos a tu asesor para solicitar PeruSAT-1")
    print(f"  3. Siguiente script: 02_crear_grilla_tiles.py")
    print(f"{'='*60}")
 