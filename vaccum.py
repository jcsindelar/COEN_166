#Joseph Sindelar 1243669
#COEN 166L
#Professor Y. Liu
#October 4th, 2019

import random

class Environment(object):
    def __init__(self):
        # instantiate squares and their respective conditions
        # 0 indicates that the square is Clean 
        # 1 indicates that the square is Dirty
        self.squareState = {'L': '0', 'R': '0'}

        # randomize conditions in locations L and R
        self.squareState['L'] = random.randint(0, 1)
        self.squareState['R'] = random.randint(0, 1)


class VacuumAgent(Environment):
    def __init__(self, Environment):
        print (Environment.squareState)
        # Instantiate action counter
        Actions = 0
        # Randomly place vacuum at a location
        vacuumLocation = random.randint(0, 1)
        # if vacuum at L
        if vacuumLocation == 0:
            print ("Vacuum randomly placed at Location L.")
            # and Location L is Dirty.
            if Environment.squareState['L'] == 1:
                print ("L is Dirty.")
                # suck the dirt  and mark it clean
                Environment.squareState['L'] = 0
                Actions += 1
                print ("L has been Cleaned.")
                # move to R
                print ("Moving to Location R...")
                Actions += 1
                # if R is Dirty
                if Environment.squareState['R'] == 1:
                    print ("R is Dirty.")
                    # suck and mark clean
                    Environment.squareState['R'] = 0
                    Actions += 1
                    print ("R has been Cleaned.")
            else:
                # move to R
                Actions += 1
                print ("Moving to R...")
                # if R is Dirty
                if Environment.squareState['R'] == 1:
                    print ("R is Dirty.")
                    # suck and mark clean
                    Environment.squareState['R'] = 0
                    Actions += 1
                    print ("R has been Cleaned.")

        elif vacuumLocation == 1:
            print ("Vacuum randomly placed at R.")
            # and R is Dirty
            if Environment.squareState['R'] == 1:
                print ("R is Dirty.")
                # suck and mark clean
                Environment.squareState['R'] = 0
                Actions += 1
                print ("R has been Cleaned.")
                # move to L
                Actions += 1
                print ("Moving to L...")
                # if L is Dirty
                if Environment.squareState['L'] == 1:
                    print ("L is Dirty.")
                    # suck and mark clean
                    Environment.squareState['L'] = 0
                    Actions += 1
                    print ("L has been Cleaned.")
            else:
                # move to L
                print ("Moving to L...")
                Actions += 1
                # if L is Dirty
                if Environment.squareState['L'] == 1:
                    print ("L is Dirty.")
                    # suck and mark clean
                    Environment.squareState['L'] = 0
                    Actions += 1
                    print ("L has been Cleaned.")
        # done cleaning
        print (Environment.squareState)
        print ("Actions: " + str(Actions))

#Initialize the environment
theEnvironment = Environment()
#Pass the environment to the agent to test
theVacuum =VacuumAgent(theEnvironment)