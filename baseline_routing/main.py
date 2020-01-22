import arcpy
import utils as uls
from problems.ThreeDSpace import ThreeDSpace, flight_Constraints
from algorithms.genetic_algorithm import GeneticAlgorithm
from arcpy import env
arcpy.CheckOutExtension('Spatial')
arcpy.env.overwriteOutput = True
import logging
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32118)
from time import gmtime, strftime
import init
arcpy.env.workspace = init.arcpy.env.workspace

def main():
    idw = arcpy.env.workspace+"\\Idw_Projected_30"
    noise_map = arcpy.env.workspace+"\\Transportation_Noise"
    line_for_initialization = arcpy.env.workspace+"\\example_route"
    geofences_restricted_airspace = arcpy.env.workspace+"\\Restricted_Airspace"
    geofence_point_boundary = arcpy.env.workspace+"\\Restricted_Airspace_Point_Boundary"
    output_new_line = r'threedline'
    #set up flight constraints
    # legal constraints parameters
    maximum_speed_legal = 27.7777777778  # in m/s. 100 in km/h
    # air taxi specific parameters

    #for Lilium Jet flight characteristics: aircraft = "Lilium". for Ehang aircraft ="EHANG"
    aircraft = "Lilium"
    type_aircraft,weight_aircraft,wing_area,CD_from_drag_polar,maximum_speed_air_taxi,acceleration_speed,acceleration_energy,deceleration_speed,deceleration_energy,minimal_cruise_energy,take_off_and_landing_energy,hover_energy,noise_pressure_acceleration, noise_pressure_deceleration, noise_at_cruise, noise_at_hover = init.aircraft_specs(aircraft)

    # flight comfort constraint
    maximum_angular_speed = 1  # (in radian/second)
    #environmental depending settings
    air_density = 1.225 #(in kg/m³) standard air pressure
    speed_of_sound = 343 # in m/s, at 20 Degrees Celsius
    gravity = 9.81 # in m/s²

    flight_constraints = flight_Constraints(type_aircraft, weight_aircraft, wing_area, CD_from_drag_polar, maximum_speed_legal, maximum_speed_air_taxi, acceleration_speed, acceleration_energy, deceleration_speed, deceleration_energy,
                                            minimal_cruise_energy, take_off_and_landing_energy, hover_energy, noise_pressure_acceleration, noise_pressure_deceleration,noise_at_cruise, noise_at_hover, maximum_angular_speed,air_density,speed_of_sound,gravity)

    #if feature classes from previous runs shall not be deleted, uncomment
    uls.delete_old_objects_from_gdb("threed")

    # setup problem
    hypercube= [(-5, 5), (-5, 5)]
    rs = uls.get_random_state(1)
    problem_instance = ThreeDSpace(search_space=hypercube,
                                   fitness_function=uls.multi_objective_NSGA_fitness_evaluation(),
                                   IDW=idw, noisemap = noise_map, x_y_limits= 350, z_sigma=5,work_space = env.workspace, random_state = rs, init_network=line_for_initialization,
                                   sample_point_distance="400 Meters", restricted_airspace=geofences_restricted_airspace, flight_constraints= flight_constraints, geofence_point_boundary=geofence_point_boundary)

    # setup Genetic Algorithm
    p_c = 0.9
    p_m = 0.5
    n_iterations = 10
    population_size = 4
    n_crossover_points = 4
    selection_pressure = 0.5
    #params mutation
    percentage_disturbed_chromosomes = 0.2
    max_disturbance_distance = 120
    percentage_inserted_and_deleted_chromosomes = 0.25
    mutation_group_size=7

    for seed in range(1):
        # setup random state
        random_state = uls.get_random_state(seed)
        # execute Genetic Algorithm
        ga1 = GeneticAlgorithm(problem_instance=problem_instance, random_state=random_state,
                               population_size=population_size, selection=uls.nsga_parametrized_tournament_selection(selection_pressure),
                               crossover=uls.n_point_crossover(n_crossover_points), p_c=p_c,
                               mutation=uls.parametrized_point_mutation(percentage_disturbed_chromosomes = percentage_disturbed_chromosomes, max_disturbance_distance = max_disturbance_distance,
                                                                        percentage_inserted_and_deleted_chromosomes = percentage_inserted_and_deleted_chromosomes, group_size=mutation_group_size), p_m=p_m, aimed_point_amount_factor = 2)
        ga1.initialize()
        #setting up the logging
        t = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
        lgr = logging.getLogger(t)
        lgr.setLevel(logging.DEBUG)  # log all escalated at and above DEBUG
        # add a file handler
        fh = logging.FileHandler(r'baseline_routing/log_files/' + t + '_new.csv')
        fh.setLevel(logging.DEBUG)
        frmt = logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s')
        fh.setFormatter(frmt)
        lgr.addHandler(fh)
        log_event = ["rs",id(seed), __name__, "population_size", population_size,"x_y_limits", problem_instance.x_y_limits,
                     "z_sigma", problem_instance.z_sigma, "sample_point_distance", problem_instance.sample_point_distance,
                     "selection_pressure", selection_pressure,
                     "pc", p_c,"pm", p_m,"aimed_point_factor",ga1.aimed_point_amount_factor,
                     "n_crossover", n_crossover_points, "mutation_max_disturbance_distance", max_disturbance_distance,
                     "mutation_group_size", mutation_group_size, "percentage_inserted_and_deleted", percentage_inserted_and_deleted_chromosomes,
                     "percentage_disturbed", percentage_disturbed_chromosomes]
        lgr.info(','.join(list(map(str, log_event))))
        ga1.search(n_iterations=n_iterations,lgr = lgr, report=True, log=True, dplot=None)


if __name__ == '__main__':
    main()
