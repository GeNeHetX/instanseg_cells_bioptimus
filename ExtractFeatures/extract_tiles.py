import geojson
import numpy

def flatten(xss) :
    return [x for x in xss]

def parse_geojson(path_geo):
    with open(path_geo, 'r') as file:
        geojson_data = geojson.load(file)

    point = [] 
    #geojson_data = geojson_data["features"]

    for cell_type in geojson_data :  
        

        if(cell_type["geometry"]["type"]=="Point") :
            point.append(cell_type["geometry"]["coordinates"])
    
    point = flatten(point)
    
    
    return point
    
 