# Team Asimov's Law: Katie Mulder, Ben LaFeldt, Mattie Phillips. We based our bot off of the Overkill bot found in the forum 
# So you’ve Improved the Random Bot. Now what? Found at http://forums.halite.io/t/so-youve-improved-the-random-bot-now-what/482.
# Our improvements are on lines 49 and 56 - 62. Comments explaining the reasoning behind the change preceed the modified
# lines of code. Our GitHub repository is at https://github.com/mattie-mcp/halite-bot .
import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random


myID, game_map = hlt.get_init()
hlt.send_init("TeamAsimov'sLawsBot")

# Find the nearest border in all cardinal directions
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

# Heuristic to account for overkill
def heuristic(square):
    if square.owner == 0 and square.strength > 0:
        return square.production / square.strength
    else:
        # return total potential damage caused by overkill when attacking this square
        return sum(neighbor.strength for neighbor in game_map.neighbors(square) if neighbor.owner not in (0, myID))

# Decide on the move for a square
def get_move(square):
	# Find a square that we do not own with the highest heuristic score in the list of neighbors
    target, direction = max(((neighbor, direction) for direction, neighbor in enumerate(game_map.neighbors(square))
                                if neighbor.owner != myID),
                                default = (None, None),
                                key = lambda t: heuristic(t[0]))
	# If we are strong enough to overtake the square, do so
    if target is not None and target.strength < square.strength:
        return Move(square, direction)
	# We changed the multiplier of square.production to 6 instead of five. After running an overnight match and testing 
	# multiple values, we determined that 6 was the optimal integer multiplier. This results in a square that waits until 
	# it possesses 6 times the production of a square before moving in the nearest enemy direction.
    elif square.strength < square.production * 6:
        return Move(square, STILL)
	
    border = any(neighbor.owner != myID for neighbor in game_map.neighbors(square))
    if not border:
		# The overkill bot moved to the nearest border, regardless if strength was lost to the strength cap. Our bot 
		# limits the amount of strength sacrificed to the cap by not moving if it would result in going over the strength cap. 
        tempDir = find_nearest_enemy_direction(square)
        tempSq = game_map.get_target(square, tempDir)
        tempStr = square.strength + tempSq.strength
        if tempStr <= 255:
            return Move(square, tempDir)
        else:
            return Move(square, STILL)
    else:
        #wait until we are strong enough to attack
        return Move(square, STILL)

# Game loop    
while True:
    game_map.get_frame()
    moves = [get_move(square) for square in game_map if square.owner == myID]
    hlt.send_frame(moves)
