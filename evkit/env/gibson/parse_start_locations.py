import json
import pickle

def parse_locations(json_file, output_file):
    points = []
    with open(json_file, 'r') as f:
        waypoints = json.load(f)
        for scenario in waypoints:
            points.append((scenario["startX"], scenario["startY"], scenario["startZ"]))
            points.append((scenario["goalX"], scenario["goalY"], scenario["goalZ"]))
            for waypoint in scenario["waypoints"]:
                points.append(tuple(waypoint))
    with open(output_file, 'wb') as f:
        pickle.dump(points, f)
    
parse_locations("Wiconisco.json", "Wiconisco.pkl")