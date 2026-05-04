#!/usr/bin/env python3
"""
descargar_limites_lima.py
Descarga limites administrativos de Lima Metropolitana y Callao desde
OpenStreetMap (Nominatim) y los guarda como GeoJSON con MultiPolygon valido.

Genera:
    data/raw/limites/lima_metropolitana.geojson         (union de todos)
    data/raw/limites/distritos_individual.geojson       (cada distrito por separado)

Uso:
    python scripts/descargar_limites_lima.py
"""

import sys
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

try:
    import geopandas as gpd
    from shapely.geometry import shape, MultiPolygon, Polygon
    from shapely.ops import unary_union
except ImportError:
    print("Error: geopandas o shapely no instalados.")
    print("  conda activate tesis")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "limites"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "lima_metropolitana.geojson"
DISTRITOS_PATH = OUTPUT_DIR / "distritos_individual.geojson"


DISTRITOS_LIMA = [
    "Ancón", "Ate", "Barranco", "Breña", "Carabayllo", "Lima",
    "Chaclacayo", "Chorrillos", "Cieneguilla", "Comas", "El Agustino",
    "Independencia", "Jesús María", "La Molina", "La Victoria", "Lince",
    "Los Olivos", "Lurigancho", "Lurín", "Magdalena del Mar", "Miraflores",
    "Pachacámac", "Pucusana", "Pueblo Libre", "Puente Piedra", "Punta Hermosa",
    "Punta Negra", "Rímac", "San Bartolo", "San Borja", "San Isidro",
    "San Juan de Lurigancho", "San Juan de Miraflores", "San Luis",
    "San Martín de Porres", "San Miguel", "Santa Anita", "Santa María del Mar",
    "Santa Rosa", "Santiago de Surco", "Surquillo", "Villa El Salvador",
    "Villa María del Triunfo"
]

DISTRITOS_CALLAO = [
    "Callao", "Bellavista", "Carmen de la Legua Reynoso", "La Perla",
    "La Punta", "Mi Perú", "Ventanilla"
]


def descargar_via_nominatim():
    """Descarga distritos uno por uno via Nominatim."""
    print("Usando Nominatim...")
    geometrias = []
    nombres = []

    todos = [(d, "Lima") for d in DISTRITOS_LIMA] + \
            [(d, "Callao") for d in DISTRITOS_CALLAO]

    print(f"  Descargando {len(todos)} distritos...")

    for i, (distrito, region) in enumerate(todos, 1):
        query = f"{distrito}, {region}, Peru"
        url = ("https://nominatim.openstreetmap.org/search?q="
                + urllib.parse.quote(query)
                + "&format=json&polygon_geojson=1&limit=1")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tesis-pucp/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read())

            if data and "geojson" in data[0]:
                geom = shape(data[0]["geojson"])
                if geom.geom_type in ("Polygon", "MultiPolygon"):
                    geometrias.append(geom)
                    nombres.append(distrito)
                    print(f"  [{i}/{len(todos)}] OK: {distrito}")
                else:
                    print(f"  [{i}/{len(todos)}] OMITIDO ({geom.geom_type}): {distrito}")
            else:
                print(f"  [{i}/{len(todos)}] FALLO: {distrito}")
        except Exception as e:
            print(f"  [{i}/{len(todos)}] ERROR: {distrito} - {e}")

        time.sleep(1.1)  # rate limit

    return geometrias, nombres


def normalizar_a_multipolygon(geom):
    """Convierte cualquier geometria a MultiPolygon, filtrando lo no valido."""
    if geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        return MultiPolygon([geom])

    if geom.geom_type == "MultiPolygon":
        return geom

    if geom.geom_type == "GeometryCollection":
        polygons = []
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                polygons.append(g)
            elif g.geom_type == "MultiPolygon":
                polygons.extend(list(g.geoms))
        if polygons:
            return MultiPolygon(polygons)

    return None


def guardar_resultado(geometrias, nombres):
    """Une geometrias y guarda como GeoJSON con MultiPolygon valido."""
    if not geometrias:
        print("\nNo hay geometrias para guardar.")
        return False

    print(f"\nUniendo {len(geometrias)} geometrias...")

    # Aplicar buffer minimo para limpiar bordes y unir poligonos adyacentes
    geometrias_buffered = [g.buffer(0.0001).buffer(-0.0001) for g in geometrias]

    union = unary_union(geometrias_buffered)
    print(f"  Tipo de geometria union: {union.geom_type}")

    multipolygon = normalizar_a_multipolygon(union)
    if multipolygon is None:
        print("  ERROR: No se pudo construir un MultiPolygon valido.")
        return False

    print(f"  Numero de poligonos finales: {len(multipolygon.geoms)}")

    gdf = gpd.GeoDataFrame(
        {"name": ["Lima Metropolitana y Callao"]},
        geometry=[multipolygon],
        crs="EPSG:4326"
    )

    if not gdf.geometry.iloc[0].is_valid:
        print("  Advertencia: geometria no valida, reparando...")
        gdf.geometry = gdf.geometry.buffer(0)

    gdf.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"\nGuardado en: {OUTPUT_PATH}")
    print(f"Tamano: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

    gdf_proj = gdf.to_crs("EPSG:32718")
    area_km2 = gdf_proj.area.iloc[0] / 1e6
    print(f"Area aproximada: {area_km2:.0f} km2")

    # Guardar tambien distritos individuales
    gdf_distritos = gpd.GeoDataFrame(
        {"distrito": nombres},
        geometry=geometrias,
        crs="EPSG:4326"
    )
    gdf_distritos.to_file(DISTRITOS_PATH, driver="GeoJSON")
    print(f"Distritos individuales: {DISTRITOS_PATH}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  DESCARGA DE LIMITES DE LIMA METROPOLITANA + CALLAO")
    print("=" * 60)

    if OUTPUT_PATH.exists():
        print(f"\nEl archivo ya existe: {OUTPUT_PATH}")
        respuesta = input("Sobreescribir? (s/n): ")
        if respuesta.lower() != "s":
            print("Cancelado.")
            sys.exit(0)

    geometrias, nombres = descargar_via_nominatim()

    if geometrias:
        if guardar_resultado(geometrias, nombres):
            print("\n" + "=" * 60)
            print("  EXITO")
            print("=" * 60)
            print("\nProximos pasos:")
            print(f"  1. Abre en QGIS: {OUTPUT_PATH}")
            print(f"  2. Verifica que veas el contorno de Lima Metropolitana")
            print(f"  3. Si se ve correcto, ejecuta: python scripts/03_generar_tiles.py")
        else:
            sys.exit(1)
    else:
        print("\nNo se pudo obtener los datos.")
        sys.exit(1)