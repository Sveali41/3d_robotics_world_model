"""Room extraction must retain local interactions without dangling room references."""

from copy import deepcopy

import pytest

from domain.thor.rooms import extract_room


def test_room_extraction_preserves_objects_and_closes_removed_connections():
    polygon = [{"x": x, "y": 0, "z": z} for x, z in [(0, 0), (0, 4), (4, 4), (4, 0)]]
    house = {
        "rooms": [{"id": "room|6", "roomType": "Kitchen", "floorPolygon": polygon},
                  {"id": "room|7", "roomType": "LivingRoom", "floorPolygon": polygon}],
        "objects": [{"id": "6|0", "assetId": "Fridge", "position": {"x": 1, "y": 0, "z": 1},
                     "children": [{"id": "small|6|1", "assetId": "Tomato"}]},
                    {"id": "7|0", "assetId": "Sofa"}],
        "walls": [{"id": "wall|6|boundary", "roomId": "room|6", "polygon": polygon, "empty": True},
                  {"id": "wall|7|boundary", "roomId": "room|7", "polygon": polygon}],
        "doors": [{"room0": "room|6", "room1": "room|7", "wall0": "wall|6|boundary",
                   "wall1": "wall|7|boundary"}], "windows": [],
        "proceduralParameters": {"lights": [{"id": "sun", "type": "directional"},
                                              {"id": "light_6", "type": "point"},
                                              {"id": "light_7", "type": "point"}]},
        "metadata": {"roomSpecId": "two-rooms"},
    }
    original = deepcopy(house)
    result = extract_room(house, "Kitchen", [{"x": 2, "y": 0.9, "z": 2}])
    assert house == original
    assert result["rooms"] == house["rooms"][:1]
    assert result["objects"] == house["objects"][:1]
    assert result["doors"] == []
    assert all("empty" not in wall for wall in result["walls"])
    assert {wall["id"] for wall in result["walls"]} == {"wall|6|boundary", "wall|exterior|boundary"}
    assert len(result["proceduralParameters"]["lights"]) == 2
    assert result["metadata"]["agent"]["position"] == {"x": 2, "y": 0.9, "z": 2}


def test_room_extraction_rejects_ambiguous_room_type():
    with pytest.raises(ValueError, match="found 2"):
        extract_room({"rooms": [{"roomType": "Kitchen"}, {"roomType": "Kitchen"}]}, "Kitchen", [])
