import arcpy
import utils as uls
import numpy as np
from problems.ThreeDSpace import ThreeDSpace, flight_Constraints
from algorithms.genetic_algorithm import GeneticAlgorithm
from arcpy import env
arcpy.CheckOutExtension('Spatial')
arcpy.env.overwriteOutput = True
import sys
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32118)
# Create a Describe object from the GDB Feature Class
#
#desc = arcpy.Describe("D:/Master_Shareverzeichnis/3DRouting/Moritz_Bk/Bk.gdb/cost_connectivity_1_3d_points")

# Print GDB FeatureClass properties
#


def main():
    env.workspace = r'C:\Users\Moritz\Desktop\Bk.gdb'
    idw = r'C:\Users\Moritz\Desktop\Bk.gdb\Idw_Refined'
    noise_map = r'C:\Users\Moritz\Desktop\Bk.gdb\Transportation_Noise'
    line_for_initialization = "example_route"
    #feature_class = r'C:\Users\Moritz\Desktop\Bk.gdb\cost_conn_zvalue_p'
    output_new_line = r'C:\Users\Moritz\Desktop\Bk.gdb\threedline'
    #geofences_restricted_airspace = "Restricted_Airspace_3D"
    geofences_restricted_airspace = r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\Restricted_Airspace_Multipar"
    geofence_point_boundary = "Restricted_Airspace_Point_Boundary_20mBuffer_10m"

    #set up flight constraints
    # legal constraints parameters
    maximum_speed_legal = 27.7777777778  # in m/s. 100 in km/h
    # air taxi specific parameters
    # This setup: (Lilium Jet, Electric VTOL Configurations Comparison 2018)
    type_aircraft = "vectored thrust eVTOL"
    weight_aircraft =  490 #(in kg)
    wing_area = 3.6 #(in m². Calculated with a wingspan of 6, Root chord 78 cm of and Tip chord of 42 cm)
    CD_from_drag_polar = (lambda CL: 0.0163 + 0.058 ** CL) #CD = Drag coefficicient, CL = Lift coefficient of plane. Obtained formula: CD = 0.0163 + 0.058 * CL²
    maximum_speed_air_taxi = 70  # (in m/s, 252 in km/h)
    acceleration_speed = 2  # (in m/s²)
    acceleration_energy = 187  # (in kW)
    deceleration_speed = -2  # (in m/s²)
    deceleration_energy = 187  # (in kW)
    minimal_cruise_energy = 28 # (in kW at speed with perfect lift/drag ratio)
    take_off_and_landing_energy = 187
    hover_energy = 187 # (in kW)
    noise_pressure_acceleration = 100
    noise_pressure_deceleration = 100
    noise_at_cruise = 100
    noise_at_hover = 100
    # flight comfort constraint
    maximum_angular_speed = 1  # (in radian/second)
    #environmental depending settings
    air_density = 1.225 #(in kg/m³) standard air pressure
    speed_of_sound = 343 # in m/s, at 20 Degrees Celsius
    gravity = 9.81 # in m/s²

    flight_constraints = flight_Constraints(type_aircraft, weight_aircraft, wing_area, CD_from_drag_polar, maximum_speed_legal, maximum_speed_air_taxi, acceleration_speed, acceleration_energy, deceleration_speed, deceleration_energy,
                                            minimal_cruise_energy, take_off_and_landing_energy, hover_energy, noise_pressure_acceleration, noise_pressure_deceleration,noise_at_cruise, noise_at_hover, maximum_angular_speed,air_density,speed_of_sound,gravity)

    uls.delete_old_objects_from_gdb("threed")

    #np.savetxt(r'D:\Master_Shareverzeichnis\3DRouting\Moritz_Bk\xyz_np.txt', init_population,fmt='%f', delimiter=';')
    #init_population = np.loadtxt(r'D:\Master_Shareverzeichnis\3DRouting\Moritz_Bk\xyz_np.txt',delimiter=';')
    #input a column for the grid value of the IDW
    # init_population_copy_z = init_population.copy()
    # init_population_copy_z = np.hstack((init_population_copy_z,init_population_copy_z[:,[2]]))

    # setup problem
    #     hypercube
    dplot = uls.Dplot()
    hypercube= [(-5, 5), (-5, 5)]
    rs = uls.get_random_state(1)
    problem_instance = ThreeDSpace(search_space=hypercube,
                                   fitness_function=uls.multi_objective_NSGA_fitness_evaluation(),
                                   IDW=idw, noisemap = noise_map, x_y_limits= 400, z_sigma=5,work_space = env.workspace, random_state = rs, init_network=line_for_initialization,
                                   sample_point_distance="350 Meters", restricted_airspace=geofences_restricted_airspace, flight_constraints= flight_constraints, geofence_point_boundary=geofence_point_boundary)
    # setup Genetic Algorithm
    p_c = 0.5
    p_m = 0.3
    n_iterations = 30
    for seed in range(1):
        # setup random state
        random_state = uls.get_random_state(seed)
        # execute Genetic Algorithm
        ga1 = GeneticAlgorithm(problem_instance=problem_instance, random_state=random_state,
                               population_size=10, selection=uls.nsga_parametrized_tournament_selection(0.3),
                               crossover=uls.n_point_crossover(3), p_c=p_c,
                               mutation=uls.parametrized_point_mutation(percentage_disturbed_chromosomes = p_m, max_disturbance_distance = 120,
                                                                        percentage_inserted_and_deleted_chromosomes = p_m, group_size=5), p_m=p_m, aimed_point_amount_factor = 1)
        ga1.initialize()
        ga1.search(n_iterations=n_iterations, report=True, log=True, dplot=None)

    # Already store initial values of x,y,z
    #Missing:
    #after cx and mutation: update the line
    #after cx, delete or insert. Update the slope, ba and distance for the splitted nodes
if __name__ == '__main__':
    main()
