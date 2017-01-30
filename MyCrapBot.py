import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
import sys
from collections import namedtuple
from collections import Counter

myID, game_map = hlt.get_init()
hlt.send_init("MyBot")

Decision = namedtuple('decision', 'mult direction')
Friendly = namedtuple('friendly', 'square direction')

def analyze_enemies(me):
    options = []
    enemies = game_map.neighbors(me,3)
    for direction, enemy in enumerate(enemies):
        if enemy.owner not in (0,myID) and enemy.strength*2 < me.strength and game_map.get_distance(me, enemy) == 1:
            options.append(Decision(direction, 1)) #easy to conquer
    return options

def analyze_square(me):
    neighbors = game_map.neighbors(me)
    friends = 0
    for direction, n in enumerate(neighbors):
        if n.owner == myID:
            friends += 1
            if friends > 2 and (int(n.strength) + int(me.strength)) < 50:
                return Move(me, direction)        
    if friends == 1:
        for direction, n in enumerate(neighbors):
            if n.owner != myID and (me.strength > n.strength):
                return Move(me, direction)
    for direction, n in enumerate(neighbors):
        if n.owner == 0:
            return Move(me, direction)
    return 0

def get_move(square):
    if square.strength < 5:
        return Move(square, STILL)
    role = analyze_square(square)
    if role != 0:
        return role
    options = analyze_enemies(square)
    for i in options:
        return Move(square, i.direction)
    return Move(square, random.choice((NORTH, EAST, SOUTH, WEST, STILL)))
while True:
    game_map.get_frame()
    moves = [get_move(square) for square in game_map if square.owner == myID]
    hlt.send_frame(moves)