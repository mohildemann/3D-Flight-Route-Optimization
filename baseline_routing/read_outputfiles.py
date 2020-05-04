import pandas as pd
import numpy as np
import csv

#df=pd.read_csv(r"baseline_routing\log_files\2020-04-28_21_03_14_new.csv", sep=',', skiprows=1, header = None)
#data = np.genfromtxt(r"baseline_routing\log_files\2020-04-28_21_03_14_new.csv", dtype=float, delimiter=',', names=True)

def read_output_csv(file):
    with open(file, 'r') as file:
        reader = csv.reader(file)
        all_solutions = []
        rowid = 0
        for row in reader:
            all_solutions_in_row = []
            for itemid in range(len(row)):
                if "[["  in row[itemid]:
                    all_solutions_in_row = row[itemid:]
            for solutionid in range(len(all_solutions_in_row)):
                if "threedpoints" in all_solutions_in_row[solutionid]:
                    all_solutions.append([rowid,all_solutions_in_row[solutionid].replace("[", "").replace("'", ""),
                                          float(all_solutions_in_row[solutionid+1].replace("[", "").replace("]", "")),
                                          float(all_solutions_in_row[solutionid+2].replace("[", "").replace("]", "")),
                                          float(all_solutions_in_row[solutionid+3].replace("[", "").replace("]", ""))]
                                         )
            rowid +=1

    return all_solutions

solutions = np.array(read_output_csv(r"baseline_routing\log_files\2020-04-28_21_03_14_new.csv"))
best_flighttime =  solutions[np.argmin(solutions[:,2].astype(float))]
best_energy = solutions[np.argmin(solutions[:,3].astype(float))]
best_noise = solutions[np.argmin(solutions[:,4].astype(float))]
#best_flightime_solution = np.where(float(solutions[:,3]) == best_flighttime)
pass