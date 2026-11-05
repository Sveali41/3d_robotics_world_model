"""Extract one room from the pinned ProcTHOR generator's house schema."""

from copy import deepcopy

from shapely.geometry import Point, Polygon


def extract_room(house, room_type, reachable_positions):
    rooms = [room for room in house["rooms"] if room["roomType"] == room_type]
    if len(rooms) != 1:
        raise ValueError(f"Expected one {room_type}; found {len(rooms)}")
    room = rooms[0]
    room_id = room["id"]
    room_number = room_id.removeprefix("room|")
    polygon = Polygon([(p["x"], p["z"]) for p in room["floorPolygon"]])
    candidates = [p for p in reachable_positions
                  if polygon.contains(Point(p["x"], p["z"]))
                  and polygon.boundary.distance(Point(p["x"], p["z"])) >= 0.4]
    if not candidates:
        raise ValueError(f"No reachable initial position with wall clearance in {room_id}")
    start = min(candidates, key=lambda p: Point(p["x"], p["z"]).distance(polygon.centroid))
    result = deepcopy(house)
    result["rooms"] = [deepcopy(room)]
    # These IDs are assigned by the pinned upstream generator, including asset groups.
    result["objects"] = [obj for obj in result["objects"] if obj["id"].startswith(room_number + "|")]
    result["walls"] = [wall for wall in result["walls"] if wall["roomId"] == room_id]
    wall_ids = {wall["id"] for wall in result["walls"]}
    exterior = []
    for wall in result["walls"]:
        wall.pop("empty", None)
        if wall["id"].startswith(f"wall|{room_number}|"):
            exterior_id = wall["id"].replace(f"wall|{room_number}|", "wall|exterior|", 1)
            if exterior_id not in wall_ids:
                outer = deepcopy(wall)
                outer["id"] = exterior_id
                outer["polygon"].reverse()
                exterior.append(outer)
    result["walls"].extend(exterior)
    wall_ids.update(wall["id"] for wall in exterior)
    for key in ("doors", "windows"):
        result[key] = [opening for opening in result[key]
                       if opening["room0"] == room_id and opening["room1"] == room_id
                       and opening["wall0"] in wall_ids and opening["wall1"] in wall_ids]
    result["proceduralParameters"]["lights"] = [
        light for light in result["proceduralParameters"]["lights"]
        if light["type"] == "directional" or light["id"] == f"light_{room_number}"
    ]
    result["metadata"]["sourceRoomSpecId"] = house["metadata"]["roomSpecId"]
    result["metadata"]["roomSpecId"] = f"extracted-{room_type}"
    result["metadata"]["sourceRoomId"] = room_id
    result["metadata"]["warnings"] = {}
    result["metadata"]["agent"] = {"position": deepcopy(start),
                                      "rotation": {"x": 0, "y": 90, "z": 0},
                                      "standing": True, "horizon": 30}
    return result
