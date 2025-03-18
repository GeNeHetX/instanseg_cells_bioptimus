import geojson
import json
import re
from pathlib import Path
from argparse import ArgumentParser

def parse_arg():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_geojson",
        type=Path,
        default=Path(r"~/Documents/Beaujon/Thèse/"),
        help="Path to the temporary directory where the features will be saved",
        required=True,
    )
    parser.add_argument(
        "--output_geojson",
        type=Path,
        default=Path(r"/mnt/d/01_HESCohortJulien/13AG/"),
        help="Path to the WSI. Can be a .svs, .ndpi, .qptiff",
    )
    return parser.parse_args()


def clean_geojson(input_geojson):
    """Removes trailing commas from GeoJSON to prevent JSONDecodeError."""
    with open(input_geojson, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Remove trailing commas before closing brackets
    cleaned_content = re.sub(r",\s*(\]|\})", r"\1", content)

    return geojson.loads(cleaned_content)

def calculate_centroid(polygon):
    """Calculate the centroid of a polygon."""
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    return [centroid_x, centroid_y]

def convert_polygons_to_centroids(input_geojson, output_geojson):
    """Convert all polygons in a GeoJSON file to their centroids."""
    
    geojson_data = clean_geojson(input_geojson)
    
    for feature in geojson_data:
        if feature["geometry"]["type"] == "Polygon":
            polygon = feature["geometry"]["coordinates"][0]  # Extract first ring of polygon
            centroid = calculate_centroid(polygon)
            feature["geometry"] = {
                "type": "Point",
                "coordinates": centroid 
            }
    
    with open(output_geojson, "w") as f:
        json.dump(geojson_data, f, indent=4)
    
    print(f"Converted file saved as {output_geojson}")

# Example usage
# convert_polygons_to_centroids("input.geojson", "output.geojson")


if __name__ == "__main__":
    args = parse_arg()
    convert_polygons_to_centroids(args.input_geojson,args.output_geojson)
