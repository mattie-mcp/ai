import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
import logging


myID, game_map = hlt.get_init()
hlt.send_init("OverkillBot2")
logging.basicConfig(filename='pydebug.log',level=logging.DEBUG)


def find_nearest_enemy_direction(square):
    direction = NORTH
    max_distance = min(game_map.width, game_map.height) / 2
    for d in (NORTH, EAST, SOUTH, WEST):
        distance = 0
        current = square
        while current.owner == myID and distance < max_distance:
            distance += 1
            current = game_map.get_target(current, d)
        if distance < max_distance:
            direction = d
            max_distance = distance
    return direction

def heuristic(square):
    if square.owner == 0 and square.strength > 0:
        return square.production / square.strength
    else:
        # return total potential damage caused by overkill when attacking this square
        return sum(neighbor.strength for neighbor in game_map.neighbors(square) if neighbor.owner not in (0, myID))

def get_move(square):
	#returns neighbor/direction of neighbor with highest prod/strength ratio
    target, direction = max(((neighbor, direction) for direction, neighbor in enumerate(game_map.neighbors(square))
                                if neighbor.owner != myID),
                                default = (None, None),
                                key = lambda t: heuristic(t[0]))
    if target is not None and target.strength < square.strength:
        if game_map.get_target(square, direction) not in planned_moves:
            planned_moves.append(game_map.get_target(square, direction))
            return Move(square, direction)
    elif square.strength < square.production * 5:
        return Move(square, STILL)

    border = any(neighbor.owner != myID for neighbor in game_map.neighbors(square))
    if not border:
        if game_map.get_target(square, find_nearest_enemy_direction(square)) not in planned_moves:
            planned_moves.append(game_map.get_target(square, find_nearest_enemy_direction(square)))
            return Move(square, find_nearest_enemy_direction(square))

    if square.strength > 250:
        return Move(square, find_nearest_enemy_direction(square))
    return Move(square, STILL)

    
while True:
    planned_moves = []
    game_map.get_frame()
    moves = [get_move(square) for square in game_map if square.owner == myID]
    for a in planned_moves:
        logging.info(a)
    hlt.send_frame(moves)

