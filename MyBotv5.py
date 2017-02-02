import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
from collections import namedtuple


myID, game_map = hlt.get_init()
hlt.send_init("MyBotv1")

Combo = namedtuple('combo', 'direction distance')

def find_nearest_enemy_direction(square):
    direction = NORTH
    distance = 0
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
    vCombo = Combo(direction, distance)
    return vCombo

def find_nearest_friend_direction(square):
    direction = NORTH
    distance = 0
    max_distance = min(game_map.width, game_map.height) / 2
    for d in (NORTH, EAST, SOUTH, WEST):
        distance = 0
        current = square
        while current.owner != myID and distance < max_distance:
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
    target, direction = max(((neighbor, direction) for direction, neighbor in enumerate(game_map.neighbors(square))
                                if neighbor.owner != myID),
                                default = (None, None),
                                key = lambda t: heuristic(t[0]))
    if target is not None and target.strength < square.strength:
        return Move(square, direction)
    elif square.strength < square.production * 5:
        return Move(square, STILL)

    border = any(neighbor.owner != myID for neighbor in game_map.neighbors(square))
    if not border:
        #Go towards enemey
        enemy = find_nearest_enemy_direction(square)
        tempDir = enemy.direction
        tempSq = game_map.get_target(square, tempDir)
        tempStr = square.strength + tempSq.strength
        friendDir = find_nearest_friend_direction(square)

        if friendDir is not None:
            friendSq = game_map.get_target(square, friendDir)

        if tempStr <= 255 and tempSq.owner in (0, myID) and enemy.distance > 3:
            return Move(square, tempDir)
        elif friendDir is not None and friendSq is not None:
            if enemy.distance < 3 and friendSq.strength+square.strength <= 255:
                return Move(square, friendDir)
            return Move(square, STILL)
        else:
            return Move(square, STILL)
    else:
        #wait until we are strong enough to attack
        return Move(square, STILL)

    
while True:
    game_map.get_frame()
    moves = [get_move(square) for square in game_map if square.owner == myID]
    hlt.send_frame(moves)