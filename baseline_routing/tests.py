import numpy as np
import math
import matplotlib.pyplot as plot
from solutions.solution import Solution
import arcpy
import scipy.stats as stats
import datetime
from arcpy import env
from arcpy.sa import *
arcpy.env.workspace = r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb"
#arcpy.env.outputZFlag = "Enabled"
arcpy.CheckOutExtension('Spatial')
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32118)
import sys
import utils as uls
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D






solution_1 = Solution([1,2,3])
solution_1.fitness = [10,20,33]
solution_2 = Solution([1,2,633])
solution_2.fitness = [33,15,100]
solution_3 = Solution([1,2,3])
solution_3.fitness = [5,7,9]
solution_4 = Solution([1,2,3])
solution_4.fitness = [10,5,8]
solution_5 = Solution([1,2,3])
solution_5.fitness = [2,2,2]
solution_6 = Solution([1,2,3])
solution_6.fitness = [3,9,9]
solution_7 = Solution([1,2,3])
solution_7.fitness = [9,7,8]
solution_8 = Solution([1,2,3])
solution_8.fitness = [10,7,6]
solution_9 = Solution([1,2,3])
solution_9.fitness = [11,7,5]

solution_10 = Solution([1,2,3])
solution_10.fitness = [66.91225970788894, 164.26189117226426, 4.351987548842328]
solution_11 = Solution([1,2,3])
solution_11.fitness = [64.35948970384028, 161.99029848661064, 4.391569125239813]
solution_12 = Solution([1,2,3])
solution_12.fitness = [69.78150590151472, 167.887226650484, 4.357979430257563]
solution_13 = Solution([1,2,3])
solution_13.fitness = [71.40129563343368, 175.66425826449972, 4.366203944485867]
solution_14 = Solution([1,2,3])
solution_14.fitness = [64.86894512352669, 162.2999476194521, 4.132931235453061]
solution_15 = Solution([1,2,3])
solution_15.fitness = [65.31830582019269, 158.17878726376193, 4.39624317009483]
#population = [solution_1,solution_2,solution_3,solution_4,solution_5,solution_6,solution_7,solution_8,solution_9]
population = [solution_10,solution_11,solution_12,solution_13,solution_14,solution_15]


def nsga(population,population_size, offsprings):
    population= population + offsprings
    non_dominated_sort(population)
    crowding_distance(population)
    accepted_solutions = []
    lowest_rank = 0
    # find lowest possible rank
    for solution in population:
        if solution.rank > lowest_rank:
            lowest_rank = solution.rank
    # append solutions with highest ranks to accepted solutions until the population_size is reached
    for r in range(lowest_rank):
        for solution in population:
            if solution.rank == r:
                accepted_solutions.append(solution)
        # if more accepted solutions exist than the population size allows, remove the solutions of lowest rank with lowest crowding distance
        if len(accepted_solutions) > population_size:
            accepted_solutions = np.array(accepted_solutions)
            reevalution_on_crowding_distance = []
            for sol in accepted_solutions:
                if sol.rank == r:
                    reevalution_on_crowding_distance.append(sol)
            crowding_distance(reevalution_on_crowding_distance)
            sorting_list = []
            for sol in reevalution_on_crowding_distance:
                sorting_list.append([sol.crowding_distance, sol])
            sorting_list = np.array(sorting_list)
            while accepted_solutions.shape[0] > population_size:
                lowest_crowding_distance = np.argmin(sorting_list[:,0])
                pos_to_delete_sorting_list = np.where(sorting_list[:,0] == sorting_list[np.argmin(sorting_list[:,0])][0])[0][0]
                pos_to_delete_accepted_list = np.where(accepted_solutions == sorting_list[pos_to_delete_sorting_list][1])[0][0]
                sorting_list = np.delete(sorting_list, pos_to_delete_sorting_list,0)
                accepted_solutions = np.delete(accepted_solutions, pos_to_delete_accepted_list,0)
            accepted_solutions = accepted_solutions.tolist()
            break
        elif len(accepted_solutions) == population_size:
            break
    return accepted_solutions




def crowding_distance(population):
    crowding_distance_f1 = []
    crowding_distance_f2 = []
    crowding_distance_f3 = []
    for solution in population:
        crowding_distance_f1.append([solution, solution.fitness[0]])
        crowding_distance_f2.append([solution, solution.fitness[1]])
        crowding_distance_f3.append([solution, solution.fitness[2]])

    #convert all to numpy arrays (more convenient)
    crowding_distance_f1 = np.array(crowding_distance_f1)
    crowding_distance_f2 = np.array(crowding_distance_f2)
    crowding_distance_f3 = np.array(crowding_distance_f3)

    #sort by fitness
    crowding_distance_f1 = crowding_distance_f1[crowding_distance_f1[:, 1].argsort(kind='mergesort')]
    crowding_distance_f2 = crowding_distance_f2[crowding_distance_f2[:, 1].argsort(kind='mergesort')]
    crowding_distance_f3 = crowding_distance_f3[crowding_distance_f3[:, 1].argsort(kind='mergesort')]


    for i in range(0, len(population)):
        population[i].crowding_distance = [0,0,0]
        pos_f1 = np.where(population[i] == crowding_distance_f1[:,0])
        pos_f2 = np.where(population[i] == crowding_distance_f2[:,0])
        pos_f3 = np.where(population[i] == crowding_distance_f3[:,0])

        #calculate crowding distance for first objective
        if pos_f1[0][0] == 0 or pos_f1[0][0] == len(population)-1:
            population[i].crowding_distance[0] = int(sys.float_info.max)/3
        else:
            #calculate normalized crowding distance
            population[i].crowding_distance[0] = (crowding_distance_f1[pos_f1[0][0]+1][1] -crowding_distance_f1[pos_f1[0][0]-1][1])/(np.argmax(crowding_distance_f1[:, 1])-np.argmin(crowding_distance_f1[:, 1]))

        # calculate crowding distance for second objective
        if pos_f2[0][0] == 0 or pos_f2[0][0] == len(population)-1:
            population[i].crowding_distance[1] = int(sys.float_info.max)/3
        else:
            #calculate normalized crowding distance
            population[i].crowding_distance[1] = (crowding_distance_f2[pos_f2[0][0]+1][1] -crowding_distance_f2[pos_f2[0][0]-1][1])/(np.argmax(crowding_distance_f2[:, 1])-np.argmin(crowding_distance_f2[:, 1]))

        # calculate crowding distance for third objective
        if pos_f3[0][0] == 0 or pos_f3[0][0] == len(population)-1:
            population[i].crowding_distance[2] = int(sys.float_info.max)/3
        else:
            #calculate normalized crowding distance
            population[i].crowding_distance[2] = (crowding_distance_f3[pos_f3[0][0]+1][1] -crowding_distance_f3[pos_f3[0][0]-1][1])/(np.argmax(crowding_distance_f3[:, 1])-np.argmin(crowding_distance_f3[:, 1]))

    for solution in population:
        solution.crowding_distance = sum(solution.crowding_distance)
    print()






def non_dominated_sort(population):
    #domination_count: number of solutions which dominate the solution
    #set_of_dominated_solutions: a set of solutions that the solution dominates
    # a solution A dominated another solution B if:
    # solution A is not worse than x2 in all objectives
    # solution A is better than solution B in at least one objective
    non_dominated_front = []
    pareto_fronts = []
    for solutionA in population:
        solutionA.domination_count = 0
        solutionA.set_of_dominated_solutions = []
        for solutionB in population:
            solutionA_dominates_solutionB = False
            solutionA_not_worse_than_SolutionB_in_all_objectives = True
            solutionA_better_than_SolutionB_in_one_objective = False

            #check if solution A dominates solution B
            for fitness_index in range(len(solutionA.fitness)):
                if solutionB.fitness[fitness_index] < solutionA.fitness[fitness_index]:
                    solutionA_not_worse_than_SolutionB_in_all_objectives = False
                elif solutionA.fitness[fitness_index] < solutionB.fitness[fitness_index]:
                    solutionA_better_than_SolutionB_in_one_objective = True
            if solutionA_not_worse_than_SolutionB_in_all_objectives is True and solutionA_better_than_SolutionB_in_one_objective is True:
                solutionA_dominates_solutionB = True
                solutionA.set_of_dominated_solutions.append(solutionB)

            # if B not dominated, check if solution B dominates solution A
            if solutionA_dominates_solutionB is False:
                solutionB_dominates_solutionA = False
                solutionB_not_worse_than_SolutionA_in_all_objectives = True
                solutionB_better_than_SolutionA_in_one_objective = False
                for fitness_index in range(len(solutionA.fitness)):
                    if solutionA.fitness[fitness_index] < solutionB.fitness[fitness_index]:
                        solutionB_not_worse_than_SolutionA_in_all_objectives = False
                    elif solutionB.fitness[fitness_index] < solutionA.fitness[fitness_index]:
                        solutionB_better_than_SolutionA_in_one_objective = True
                if solutionB_not_worse_than_SolutionA_in_all_objectives is True and solutionB_better_than_SolutionA_in_one_objective is True:
                    solutionB_dominates_solutionA = True
                    solutionA.domination_count += 1

        if solutionA.domination_count == 0:
            solutionA.rank = 1
            non_dominated_front.append(solutionA)
    pareto_fronts.append([0,non_dominated_front])


    for i in range(len(population)-1):
        current_domination_front = []
        for dominating_solution in pareto_fronts[i][1]:

            if len(dominating_solution.set_of_dominated_solutions) >0:

                for dominated_solution in dominating_solution.set_of_dominated_solutions:
                    dominated_solution.domination_count -= 1
                    if dominated_solution.domination_count == 0:
                        current_domination_front.append(dominated_solution)
                        dominated_solution.rank = i + 2
        if len(current_domination_front) == 0:
            break
        else:
            pareto_fronts.append([i+1,current_domination_front])
    return pareto_fronts

t = non_dominated_sort(population)
t= np.array(t)
paretofront_1 = []
for i in t[0][1]:
    paretofront_1.append(i.fitness)
paretofront_1 =  np.array(paretofront_1)
t2 = paretofront_1[:][0]
print(paretofront_1[0])



t=nsga(population[:4],4,population[4:])



x = np.random.uniform(0.2,0.9,100)
y = np.random.uniform(0.4,0.6,100)
z = np.random.uniform(0.1,0.4,100)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#x =[1,2,3,4,5,6,7,8,9,10]
##y =[5,6,2,3,13,4,1,2,4,8]
#z =[2,3,3,3,5,7,9,11,9,10]



ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_zlabel('Objective 3')

plt.show()
print()












x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
y = np.sin(x)
tck = interpolate.splrep(x, y, s=0)
xnew = np.arange(0, 2*np.pi, np.pi/50)
ynew = interpolate.splev(xnew, tck, der=0)

plt.figure()
plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.title('Cubic-spline interpolation')
plt.show()

# near_features = solution_fc
# tables = arcpy.ListTables()
# for table in tables:
#     if "Near_Table_Reorder" in table:
#         arcpy.Delete_management(table)
# out_table = "Near_Table_Reorder"
# # optional parameters
# search_radius = '1500 Meters'
# location = 'LOCATION'
# angle = 'ANGLE'
# closest = 'ALL'
# closest_count = 10
# table = arcpy.GenerateNearTable_analysis(in_features, near_features, out_table, search_radius,
#
#                                          location, angle, closest, closest_count)
def pol_boundary_to_points(fc_in, fc_out, interval):
    pnts = []
    with arcpy.da.SearchCursor(fc_in,
                               ("SHAPE@")) as curs:
        for row in curs:
            polygon = row[0]
            outline = polygon.boundary()
            d = 0
            while d < outline.length:
                pnt = outline.positionAlongLine(d, False)
                pnts.append(pnt)
                d += interval
        arcpy.CopyFeatures_management(pnts, fc_out)
def line_repair(geofences,flight_path_line, flight_path, geofence_point_boundary,placeholder_interpolated_surface, noisemap,IDW, output_name):
    def find_shortest_way_between_intersection_points(boundary_points, intersection_points):
        #Delete old tables
        tables = arcpy.ListTables()
        for table in tables:
            if "NearTable" in table:
                arcpy.Delete_management(table)
        #generate near table (self join) of point boundary. limit to three nearest points.
        out_table_point_boundary = r"memory\NearTableBoundaryPoints"
        arcpy.analysis.GenerateNearTable(boundary_points, boundary_points,
                                         out_table_point_boundary,
                                         "20 Meters", "LOCATION", "NO_ANGLE", "ALL", 4, "PLANAR")
        #convert self join near table of the point boundary points into a numpy array. Saves the two neighbor points of each point.
        array_point_boundary = arcpy.da.TableToNumPyArray(out_table_point_boundary,('*'))
        array_point_boundary = array_point_boundary.reshape(array_point_boundary.shape[0], 1)
        array_point_boundary = np.array([list(i[0]) for i in array_point_boundary])
        array_point_boundary = array_point_boundary[:, [1,2,7,8]]

        #convert near table of intersection points to the two closest point boundary points into a numpy array
        out_table_intersection_point = r"memory\NearTableIntersectionPointsToBoundaryPoints"
        arcpy.analysis.GenerateNearTable(intersection_points, boundary_points,
                                         out_table_intersection_point,
                                         "100 Meters", "LOCATION", "NO_ANGLE", "ALL", 2, "PLANAR")
        array_point_intersection_point = arcpy.da.TableToNumPyArray(out_table_intersection_point,('*'))
        array_point_intersection_point = array_point_intersection_point.reshape(array_point_intersection_point.shape[0], 1)
        array_point_intersection_point = np.array([list(i[0]) for i in array_point_intersection_point])

        pointid_nearest_to_first_intersection_point = array_point_intersection_point[0,[2,7,8]]
        pointid_2ndnearest_to_first_intersection_point = array_point_intersection_point[1,[2,7,8]]

        pointid_nearest_to_2nd_intersection_point = array_point_intersection_point[2,[2,7,8]]
        pointid_2ndnearest_to_2nd_intersection_point = array_point_intersection_point[3,[2,7,8]]

        #"walk" in both directions of the geofence point boundary. stop if second intersection point is met.
        array_first_direction = np.array(pointid_nearest_to_first_intersection_point).reshape(1,3)
        array_other_direction = np.array(pointid_2ndnearest_to_first_intersection_point).reshape(1,3)
        for i in range(array_point_boundary.shape[0]):
            #going first direction
            result_first_direction = np.where(
                (array_point_boundary[:, 0] == array_first_direction[-1][0])
                & (np.in1d(array_point_boundary[:, 1], array_first_direction[:, 0]) == False)
                & (np.in1d(array_point_boundary[:, 1], array_other_direction[:, 0]) == False))


            if len(result_first_direction[0]) >= 1:

                # look for the wanted id in the solution.representation and add it to the ordered list
                value_to_append = array_point_boundary[int(result_first_direction[0][0]),[1,2,3]]
                array_first_direction = np.vstack([array_first_direction, value_to_append])
                if (value_to_append[0] == pointid_nearest_to_2nd_intersection_point[0] or value_to_append[0] == pointid_2ndnearest_to_2nd_intersection_point[0]):
                    return array_first_direction

            # going other direction
            result_other_direction = np.where(
                (array_point_boundary[:,0] == array_other_direction[-1][0])
                & (np.in1d(array_point_boundary[:, 1], array_other_direction[:, 0]) == False)
                & (np.in1d(array_point_boundary[:, 1], array_first_direction[:, 0]) == False)
                )
            if len(result_other_direction[0]) >= 1:
                # look for the wanted id in the solution.representation and add it to the ordered list
                value_to_append_other_direction = array_point_boundary[int(result_other_direction[0][0]), [1, 2, 3]]
                array_other_direction = np.vstack([array_other_direction, value_to_append_other_direction])
                if (value_to_append_other_direction[0] == pointid_nearest_to_2nd_intersection_point[0] or value_to_append_other_direction[0] == pointid_2ndnearest_to_2nd_intersection_point[0]):
                    return array_other_direction


        if i >= array_point_boundary.shape[0]:
            print("linerepair didnt work due to problems with the point boundary replacement")

    arcpy.analysis.Intersect([flight_path_line, geofences], r"memory\intL", "ALL", None,
                             "LINE")
    fields = ["OBJECTID", "SHAPE@X", "SHAPE@Y"]
    # explode multi point to single point
    arcpy.MultipartToSinglepart_management(r"memory\intL", r"memory\sintL")
    arcpy.management.FeatureVerticesToPoints(r"memory\sintL",
                                            r"memory\sintP",
                                             "BOTH_ENDS")
    _counter = 0
    new_coordinates = []
    with arcpy.da.SearchCursor(
            r"memory\sintP",
            fields) as cursor:
        for row in cursor:
            new_coordinates.append([row[0], row[1], row[2]])
    new_coordinates = np.array(new_coordinates)
    uls.delete_old_objects_from_gdb("selected_boundary_points")
    # loop in stepsize 2: one is the first point before geofence intersection, one the point at the end of the intersection
    arcpy.CreateFeatureclass_management(arcpy.env.workspace, "selected_boundary_points", "POINT",template = geofence_point_boundary)

    for i in range(0, len(new_coordinates - 2), 2):
        _counter += 1
        #createRectangleBetweenPoints(new_coordinates[i][1:], new_coordinates[i + 1][1:],
                                     #r"memory\rect_" + str(_counter))
        Selected_Intersection_points = arcpy.SelectLayerByAttribute_management(r"memory\sintP", "NEW_SELECTION",
                                                '"OBJECTID" IN ({0})'.format(', '.join(map(str, [i+1, i+2]))))
        arcpy.CopyFeatures_management(Selected_Intersection_points,
                                      r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\testingpoints"+str(_counter))
        matchcount = int(arcpy.GetCount_management(Selected_Intersection_points)[0])
        Selected_geofences = arcpy.management.SelectLayerByLocation(geofences, "WITHIN_A_DISTANCE_GEODESIC",
                                               Selected_Intersection_points, "20 Meters", "NEW_SELECTION", "NOT_INVERT")
        #arcpy.CopyFeatures_management(Selected_geofences,
                                      #r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\testgeofencesel" + str(
                                          #_counter))

        matchcountpols = int(arcpy.GetCount_management(Selected_geofences)[0])
        pol_boundary_to_points(Selected_geofences,r"memory\bound_p_" + str(_counter), 10)

        #select the two corresponding intersection points
        points_to_insert = find_shortest_way_between_intersection_points(r"memory\bound_p_" + str(_counter), Selected_Intersection_points)
        uls.create_xy_FCFromPoints(points_to_insert[:,1], points_to_insert[:,2], r"memory\selected_boundary_points")

        #insert the points of the geofence point boundary which are the nearest to the output points
        # positioned exactly on the geofence boundary (which lie 10 meters outside of the geofence)
        #generate near table to the point boundary which is 1o meters away from the geofence
        out_table_point_boundary = r"memory\NearGeofenceBoundaryPoints"
        arcpy.analysis.GenerateNearTable(r"memory\selected_boundary_points", geofence_point_boundary,
                                         out_table_point_boundary,
                                         "50 Meters", "NO_LOCATION", "NO_ANGLE", "ALL", 1, "PLANAR")
        # convert self join near table of the point boundary points into a numpy array. Saves the two neighbor points of each point.
        array_point_boundary = arcpy.da.TableToNumPyArray(out_table_point_boundary, ('*'))
        array_point_boundary = array_point_boundary.reshape(array_point_boundary.shape[0], 1)
        array_point_boundary = np.array([list(i[0]) for i in array_point_boundary])
        #get the index of the point boundaries
        array_point_boundary = array_point_boundary[:, 2]
        array_point_boundary = array_point_boundary.astype(int)
        Selected_geofence_boundary_points = arcpy.SelectLayerByAttribute_management(geofence_point_boundary, "ADD_TO_SELECTION",
                                                                               '"OBJECTID" IN ({0})'.format(', '.join(
                                                                                   map(str, array_point_boundary))))
        arcpy.CopyFeatures_management(Selected_geofence_boundary_points, r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\testpoints")
        matchcount_boundary_points = int(arcpy.GetCount_management(Selected_geofence_boundary_points)[0])

        arcpy.Append_management([Selected_geofence_boundary_points],"selected_boundary_points")

        arcpy.SelectLayerByAttribute_management(Selected_Intersection_points, "CLEAR_SELECTION")
        arcpy.SelectLayerByAttribute_management(Selected_geofences, "CLEAR_SELECTION")
        arcpy.SelectLayerByAttribute_management(Selected_geofence_boundary_points, "CLEAR_SELECTION")

    uls.extractValues_many_Rasters("selected_boundary_points",
                                   r" {} int_z; {} noise; {} int_z1; {} grid_z;{} grid_z".format(
                                       placeholder_interpolated_surface, noisemap,
                                       placeholder_interpolated_surface, IDW, IDW))

    arcpy.FeatureTo3DByAttribute_3d("selected_boundary_points",
                                    r"memory\selected_boundary_points_3D", "int_z")

    arcpy.CopyFeatures_management(r"memory\selected_boundary_points_3D",
    r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\testgeofencesemerge")
    arcpy.Merge_management([r"memory\selected_boundary_points_3D", flight_path],
                           output_name)

    ##start edit
    # deleting the points between the intP points
    selection_for_deletion = arcpy.management.SelectLayerByLocation(output_name,
                                                                    "WITHIN", geofences, None,
                                                                    "NEW_SELECTION","NOT_INVERT")
    if int(arcpy.GetCount_management(selection_for_deletion).getOutput(0)) > 0:
        arcpy.DeleteFeatures_management(selection_for_deletion)

    ##end edit

    combined_3D_representation = uls.point3d_fc_to_np_array(
        output_name,
        additional_fields=["grid_z", "noise"])
    arcpy.Delete_management("in_memory")
    return combined_3D_representation

        #merge flight path points with additional points for repairing

test=line_repair(geofences=r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\Restricted_Airspace_Multipar",
            flight_path_line=r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\threedpoints_1575707447_ph_sml_for_testing",
            flight_path=r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\threedpoints_1575707447_ph_3d_for_testing",
            geofence_point_boundary=r"C:\Users\Moritz\Desktop\Bk.gdb\Geofence_Points_For_Relocation_10m"
            ,placeholder_interpolated_surface= r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\interpolated_surface_for_testing",
            noisemap= r"C:\Users\Moritz\Desktop\Bk.gdb\Transportation_Noise",
            IDW=r"C:\Users\Moritz\Desktop\Bk.gdb\Idw_Projected_30",
            output_name=r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\testmerge_linerepair")
print()
#input_near = "Geofence_Points_For_Relocation_10m"
#points_for_z = r"Restricted_Airspace_Buffer_Lines_GeneratePointsAlongLines"

#arcpy.AddField_management(r"C:\Users\Moritz\Desktop\Bk.gdb\Geofence_Points_For_Relocation_10m", "grid_z", "FLOAT", 2,
                          #field_alias="grid_z", field_is_nullable="NULLABLE")

#cur = arcpy.UpdateCursor(r"C:\Users\Moritz\Desktop\Bk.gdb\Geofence_Points_For_Relocation_10m",["OBJECTID","grid_z"])
#cur2 = arcpy.SearchCursor(r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\Geofence_Points_For_Relocati",["IN_FID","NEAR_FID"])
#cur3 = arcpy.SearchCursor(r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\Restricted_Airspace_Buffer_Lines_GeneratePointsAlongLines",["OBJECTID","grid_z"])
# with arcpy.da.UpdateCursor(r"C:\Users\Moritz\Desktop\Bk.gdb\Geofence_Points_For_Relocation_10m", ["OBJECTID","grid_z"]) as cur:
#     for row in cur:
#         with arcpy.SearchCursor(r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\Geofence_Points_For_Relocati",["IN_FID","NEAR_FID"]) as cur2:
#             for row2 in cur2:
#                 with  arcpy.SearchCursor(r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\Restricted_Airspace_Buffer_Lines_GeneratePointsAlongLines",["OBJECTID","grid_z"]) as cur3:
#                     if row[0] == row2[0]:
#                         for row3 in cur3:
#                             if row3[0] == row2[0] and row3[1] == row2[1]:
#                                 row[1] = row3[1]
#                                 cur.updateRow(row)


def select_points_at_geofence_boundary(target_fc, join_fc, out_fc):
    #hier muss weiter gemacht werden: spatial join+delete
    tempLayer = str(out_fc) + "temp"
    arcpy.SpatialJoin_analysis(target_features=target_fc,join_features= join_fc,out_feature_class= out_fc ,match_option= "WITHIN")
    arcpy.MakeFeatureLayer_management(out_fc, tempLayer)
    arcpy.SelectLayerByLocation_management(in_layer=tempLayer,overlap_type="COMPLETELY_WITHIN",select_features=join_fc,search_distance=None,
                                           selection_type="NEW_SELECTION",invert_spatial_relationship= "NOT_INVERT")
    arcpy.DeleteFeatures_management(tempLayer)


def repairLinesInRestrictedAirspace(flightpath_line, flightpath_points, flightpath_numpy_representation,geofences_restricted_airspace,geofence_point_boundary):
    arcpy.analysis.Intersect([flightpath_line,geofences_restricted_airspace], str(flightpath_points)+"intersectionPoints","ALL", None, "POINT")
    fields = ["OBJECTID", "SHAPE@X", "SHAPE@Y"]
    #explode multi point to single point
    arcpy.MultipartToSinglepart_management( flightpath_points+"intersectionPoints",flightpath_points+ "singleIntersectionPoints")
    # set local variables
    in_features = flightpath_points+ "singleIntersectionPoints"
    near_features = geofence_point_boundary
    # find features only within search radius
    search_radius = "500 Meters"
    # find location nearest features
    location = "NO_LOCATION"
    # avoid getting angle of neares features
    angle = "NO_ANGLE"
    # execute the function
    arcpy.Near_analysis(in_features, near_features, search_radius, location, angle)
    print()
    #loop two-step-wise
        #get NEAR_FID if first and second
        #check which is smaller and which is bigger

geofence_point_boundary = "Geofence_Points_For_Relocation_10m"
fc_geofence = r"C:\Users\Moritz\Desktop\Bk.gdb\Restricted_Airspace"
testline = r"C:\Users\Moritz\Desktop\Bk.gdb\threedline_1575461531"
testpoints = r"C:\Users\Moritz\Desktop\Bk.gdb\threedpoints_1575461531_ph_3d"
repairLinesInRestrictedAirspace(testline, testpoints, 2, fc_geofence,geofence_point_boundary)



    # _counter = 0
    # new_coordinates = []
    # with arcpy.da.SearchCursor(
    #         flightpath_points+"singleIntersectionPoints",
    #         fields) as cursor:
    #     for row in cursor:
    #         in_features = row
    #         reorder_is_valid = True
    #         near_features = solution_fc
    #         tables = arcpy.ListTables()
    #         for table in tables:
    #             if "Near_Table_Reorder" in table:
    #                 arcpy.Delete_management(table)
    #         out_table = "Near_Table_Reorder"
    #         # optional parameters
    #         search_radius = '1500 Meters'
    #         location = 'LOCATION'
    #         angle = 'ANGLE'
    #         closest = 'ALL'
    #         closest_count = 10
    #         table = arcpy.GenerateNearTable_analysis(in_features, near_features, out_table, search_radius,
    #                                                  location, angle, closest, closest_count)
    #         arr = arcpy.da.TableToNumPyArray(out_table, ('*'))
    #         arr = arr.reshape(arr.shape[0], 1)
    #         arr = np.array([list(i[0]) for i in arr])
    #
    #
    #
    #
    #         new_coordinates.append([row[0], row[1], row[2]])
    # new_coordinates = np.array(new_coordinates)
    # for i in range(0, len(new_coordinates - 2), 2):




        # createRectangleBetweenPoints(new_coordinates[i][1:], new_coordinates[i + 1][1:],"rect_"+str(_counter))
        # pol_boundary_to_points("rect_"+str(_counter), "bound_p_"+str(_counter), 10)
        # select_points_at_geofence_boundary("bound_p_"+str(_counter), geofences_restricted_airspace, "rep_p_"+str(_counter))
        # update_coordinates = []
        # arcpy.analysis.Near("rep_p_"+str(_counter), geofence_point_boundary, "50 Meters", "LOCATION",
        #                     "NO_ANGLE", "PLANAR")
        # fields = ["OBJECTID", "NEAR_X", "NEAR_Y"]
        # with arcpy.da.SearchCursor("rep_p_"+str(_counter), fields) as cursor:
        #     for row in cursor:
        #         update_coordinates.append([row[0], row[1], row[2]])
        # update_coordinates = np.array(update_coordinates).reshape(len(update_coordinates), 3)
        # fields = ["OBJECTID", "SHAPE@X", "SHAPE@Y"]
        # with arcpy.da.UpdateCursor("rep_p_"+str(_counter), fields) as cursor:
        #     # get ready to update each row
        #     to_delete1 = []
        #     for row in cursor:
        #         #delete points that are not within specified minimum range
        #         for i in range(len(update_coordinates)):
        #             if row[0] == update_coordinates[i][0]:
        #                 # if the values in the near table are -1 in nearx and neary position, then there was no near point in the specified near distance (500m).
        #                 # in this case, the point can be deleted
        #                 if update_coordinates[i][1] == -1 or update_coordinates[i][2] == -1:
        #                     cursor.deleteRow()
        #                     del_pos1 = np.where(update_coordinates == row[0])
        #                     del_pos1 = del_pos1[0][0]
        #                     to_delete1.append(del_pos1)
        #         update_coordinates = np.delete(update_coordinates, to_delete1, axis=0)
        #         #update flightpath representation
        #         #update point fc
        #         #missing: z-Werte der Punkte bestimmen

def pol_boundary_to_points(fc_in, fc_out, interval):
    pnts = []
    with arcpy.da.SearchCursor(fc_in,
                               ("SHAPE@")) as curs:
        for row in curs:
            polygon = row[0]
            outline = polygon.boundary()
            d = 0
            while d < outline.length:
                pnt = outline.positionAlongLine(d, False)
                pnts.append(pnt)
                d += interval
        arcpy.CopyFeatures_management(pnts, fc_out)

def delete_points_in_geofence(target_fc, join_fc, out_fc):
    #hier muss weiter gemacht werden: spatial join+delete
    tempLayer = str(out_fc) + "temp"
    arcpy.SpatialJoin_analysis(target_features=target_fc,join_features= join_fc,out_feature_class= out_fc ,match_option= "WITHIN")
    arcpy.MakeFeatureLayer_management(out_fc, tempLayer)
    arcpy.SelectLayerByLocation_management(in_layer=tempLayer,overlap_type="COMPLETELY_WITHIN",select_features=join_fc,search_distance=None,
                                           selection_type="NEW_SELECTION",invert_spatial_relationship= "NOT_INVERT")
    arcpy.DeleteFeatures_management(tempLayer)

fc_point_out = r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\test_p"
fc_in = r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\Restricted_Airspace_Inte4"
pol_boundary_to_points(fc_in, fc_point_out, 10)
fc_geofence = r"C:\Users\Moritz\Desktop\Bk.gdb\Restricted_Airspace"
delete_points_in_geofence(fc_point_out, fc_geofence, r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\Goodpoints")
print()
def createRectangleBetweenPoints(a, b, rect_fc_name):
    arcpy.env.outputZFlag = "Disabled"
    pointA = arcpy.Point(float(a[0]),float(a[1]))
    ptGeometryA = arcpy.PointGeometry(pointA, arcpy.env.outputCoordinateSystem)
    pointB = arcpy.Point(float(b[0]), float(b[1]))
    ptGeometryB = arcpy.PointGeometry(pointB, arcpy.env.outputCoordinateSystem)
    ang_and_dist = ptGeometryA.angleAndDistanceTo(ptGeometryB, "GEODESIC")
    ptRect1 = ptGeometryA.pointFromAngleAndDistance(ang_and_dist[0] + 90, ang_and_dist[1])
    ptRect2 = ptGeometryA.pointFromAngleAndDistance(ang_and_dist[0] - 90, ang_and_dist[1])
    ptRect3 = ptGeometryB.pointFromAngleAndDistance(ang_and_dist[0] + 90, ang_and_dist[1])
    ptRect4 = ptGeometryB.pointFromAngleAndDistance(ang_and_dist[0] - 90, ang_and_dist[1])
    array = arcpy.Array()
    array.add(ptRect1.firstPoint)
    array.add(ptRect2.firstPoint)
    array.add(ptRect4.firstPoint)
    array.add(ptRect3.firstPoint)
    features = []
    polygon = arcpy.Polygon(array, arcpy.env.outputCoordinateSystem)
    features.append(polygon)
    arcpy.CopyFeatures_management(features, rect_fc_name)
    arcpy.env.outputZFlag = "Enabled"
    #ptGeometry.angleAndDistanceTo(ptGeometry2, "GEODESIC")


arcpy.analysis.Intersect([r"D:\Master_Shareverzeichnis\1.Semester\Flighttaxi_project\MyProject15\MyProject15.gdb\threedpoints_1571072368_PointsToLine",r"D:\Master_Shareverzeichnis\1.Semester\Flighttaxi_project\MyProject15\MyProject15.gdb\test_3d_geofence"], r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\intersection_points3", "ALL", None, "POINT")
fields = ["OBJECTID", "SHAPE@X", "SHAPE@Y"]
arcpy.MultipartToSinglepart_management(r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\intersection_points3",
                                       r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\intersection_singlepoints3")
_counter = 0
new_coordinates = []
with arcpy.da.SearchCursor(r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\intersection_singlepoints3", fields) as cursor:
    for row in cursor:
        new_coordinates.append([row[0], row[1], row[2]])
new_coordinates = np.array(new_coordinates)
for i in range(0,len(new_coordinates-2),2):
    print(angleBetweenPoints(new_coordinates[i][1:],new_coordinates[i+1][1:]))




solution_1 = Solution([1,2,3])
solution_1.fitness = [10,20,33]
solution_2 = Solution([1,2,633])
solution_2.fitness = [33,15,100]
solution_3 = Solution([1,2,3])
solution_3.fitness = [5,7,9]
solution_4 = Solution([1,2,3])
solution_4.fitness = [10,5,8]
solution_5 = Solution([1,2,3])
solution_5.fitness = [2,2,2]
solution_6 = Solution([1,2,3])
solution_6.fitness = [3,9,9]
solution_7 = Solution([1,2,3])
solution_7.fitness = [9,7,8]
solution_8 = Solution([1,2,3])
solution_8.fitness = [10,7,6]
solution_9 = Solution([1,2,3])
solution_9.fitness = [11,7,5]

solution_10 = Solution([1,2,3])
solution_10.fitness = [66.91225970788894, 164.26189117226426, 4.351987548842328]
solution_11 = Solution([1,2,3])
solution_11.fitness = [64.35948970384028, 161.99029848661064, 4.391569125239813]
solution_12 = Solution([1,2,3])
solution_12.fitness = [69.78150590151472, 167.887226650484, 4.357979430257563]
solution_13 = Solution([1,2,3])
solution_13.fitness = [71.40129563343368, 175.66425826449972, 4.366203944485867]
solution_14 = Solution([1,2,3])
solution_14.fitness = [64.86894512352669, 162.2999476194521, 4.132931235453061]
solution_15 = Solution([1,2,3])
solution_15.fitness = [65.31830582019269, 158.17878726376193, 4.39624317009483]
#population = [solution_1,solution_2,solution_3,solution_4,solution_5,solution_6,solution_7,solution_8,solution_9]
population = [solution_10,solution_11,solution_12,solution_13,solution_14,solution_15]


def nsga(population,population_size, offsprings):
    population= population + offsprings
    non_dominated_sort(population)
    crowding_distance(population)
    accepted_solutions = []
    lowest_rank = 0
    # find lowest possible rank
    for solution in population:
        if solution.rank > lowest_rank:
            lowest_rank = solution.rank
    # append solutions with highest ranks to accepted solutions until the population_size is reached
    for r in range(lowest_rank):
        for solution in population:
            if solution.rank == r:
                accepted_solutions.append(solution)
        # if more accepted solutions exist than the population size allows, remove the solutions of lowest rank with lowest crowding distance
        if len(accepted_solutions) > population_size:
            accepted_solutions = np.array(accepted_solutions)
            reevalution_on_crowding_distance = []
            for sol in accepted_solutions:
                if sol.rank == r:
                    reevalution_on_crowding_distance.append(sol)
            crowding_distance(reevalution_on_crowding_distance)
            sorting_list = []
            for sol in reevalution_on_crowding_distance:
                sorting_list.append([sol.crowding_distance, sol])
            sorting_list = np.array(sorting_list)
            while accepted_solutions.shape[0] > population_size:
                lowest_crowding_distance = np.argmin(sorting_list[:,0])
                pos_to_delete_sorting_list = np.where(sorting_list[:,0] == sorting_list[np.argmin(sorting_list[:,0])][0])[0][0]
                pos_to_delete_accepted_list = np.where(accepted_solutions == sorting_list[pos_to_delete_sorting_list][1])[0][0]
                sorting_list = np.delete(sorting_list, pos_to_delete_sorting_list,0)
                accepted_solutions = np.delete(accepted_solutions, pos_to_delete_accepted_list,0)
            accepted_solutions = accepted_solutions.tolist()
            break
        elif len(accepted_solutions) == population_size:
            break
    return accepted_solutions




def crowding_distance(population):
    crowding_distance_f1 = []
    crowding_distance_f2 = []
    crowding_distance_f3 = []
    for solution in population:
        crowding_distance_f1.append([solution, solution.fitness[0]])
        crowding_distance_f2.append([solution, solution.fitness[1]])
        crowding_distance_f3.append([solution, solution.fitness[2]])

    #convert all to numpy arrays (more convenient)
    crowding_distance_f1 = np.array(crowding_distance_f1)
    crowding_distance_f2 = np.array(crowding_distance_f2)
    crowding_distance_f3 = np.array(crowding_distance_f3)

    #sort by fitness
    crowding_distance_f1 = crowding_distance_f1[crowding_distance_f1[:, 1].argsort(kind='mergesort')]
    crowding_distance_f2 = crowding_distance_f2[crowding_distance_f2[:, 1].argsort(kind='mergesort')]
    crowding_distance_f3 = crowding_distance_f3[crowding_distance_f3[:, 1].argsort(kind='mergesort')]


    for i in range(0, len(population)):
        population[i].crowding_distance = [0,0,0]
        pos_f1 = np.where(population[i] == crowding_distance_f1[:,0])
        pos_f2 = np.where(population[i] == crowding_distance_f2[:,0])
        pos_f3 = np.where(population[i] == crowding_distance_f3[:,0])

        #calculate crowding distance for first objective
        if pos_f1[0][0] == 0 or pos_f1[0][0] == len(population)-1:
            population[i].crowding_distance[0] = int(sys.float_info.max)/3
        else:
            #calculate normalized crowding distance
            population[i].crowding_distance[0] = (crowding_distance_f1[pos_f1[0][0]+1][1] -crowding_distance_f1[pos_f1[0][0]-1][1])/(np.argmax(crowding_distance_f1[:, 1])-np.argmin(crowding_distance_f1[:, 1]))

        # calculate crowding distance for second objective
        if pos_f2[0][0] == 0 or pos_f2[0][0] == len(population)-1:
            population[i].crowding_distance[1] = int(sys.float_info.max)/3
        else:
            #calculate normalized crowding distance
            population[i].crowding_distance[1] = (crowding_distance_f2[pos_f2[0][0]+1][1] -crowding_distance_f2[pos_f2[0][0]-1][1])/(np.argmax(crowding_distance_f2[:, 1])-np.argmin(crowding_distance_f2[:, 1]))

        # calculate crowding distance for third objective
        if pos_f3[0][0] == 0 or pos_f3[0][0] == len(population)-1:
            population[i].crowding_distance[2] = int(sys.float_info.max)/3
        else:
            #calculate normalized crowding distance
            population[i].crowding_distance[2] = (crowding_distance_f3[pos_f3[0][0]+1][1] -crowding_distance_f3[pos_f3[0][0]-1][1])/(np.argmax(crowding_distance_f3[:, 1])-np.argmin(crowding_distance_f3[:, 1]))

    for solution in population:
        solution.crowding_distance = sum(solution.crowding_distance)
    print()






def non_dominated_sort(population):
    #domination_count: number of solutions which dominate the solution
    #set_of_dominated_solutions: a set of solutions that the solution dominates
    # a solution A dominated another solution B if:
    # solution A is not worse than x2 in all objectives
    # solution A is better than solution B in at least one objective
    non_dominated_front = []
    pareto_fronts = []
    for solutionA in population:
        solutionA.domination_count = 0
        solutionA.set_of_dominated_solutions = []
        for solutionB in population:
            solutionA_dominates_solutionB = False
            solutionA_not_worse_than_SolutionB_in_all_objectives = True
            solutionA_better_than_SolutionB_in_one_objective = False

            #check if solution A dominates solution B
            for fitness_index in range(len(solutionA.fitness)):
                if solutionB.fitness[fitness_index] < solutionA.fitness[fitness_index]:
                    solutionA_not_worse_than_SolutionB_in_all_objectives = False
                elif solutionA.fitness[fitness_index] < solutionB.fitness[fitness_index]:
                    solutionA_better_than_SolutionB_in_one_objective = True
            if solutionA_not_worse_than_SolutionB_in_all_objectives is True and solutionA_better_than_SolutionB_in_one_objective is True:
                solutionA_dominates_solutionB = True
                solutionA.set_of_dominated_solutions.append(solutionB)

            # if B not dominated, check if solution B dominates solution A
            if solutionA_dominates_solutionB is False:
                solutionB_dominates_solutionA = False
                solutionB_not_worse_than_SolutionA_in_all_objectives = True
                solutionB_better_than_SolutionA_in_one_objective = False
                for fitness_index in range(len(solutionA.fitness)):
                    if solutionA.fitness[fitness_index] < solutionB.fitness[fitness_index]:
                        solutionB_not_worse_than_SolutionA_in_all_objectives = False
                    elif solutionB.fitness[fitness_index] < solutionA.fitness[fitness_index]:
                        solutionB_better_than_SolutionA_in_one_objective = True
                if solutionB_not_worse_than_SolutionA_in_all_objectives is True and solutionB_better_than_SolutionA_in_one_objective is True:
                    solutionB_dominates_solutionA = True
                    solutionA.domination_count += 1

        if solutionA.domination_count == 0:
            solutionA.rank = 1
            non_dominated_front.append(solutionA)
    pareto_fronts.append([0,non_dominated_front])


    for i in range(len(population)-1):
        current_domination_front = []
        for dominating_solution in pareto_fronts[i][1]:

            if len(dominating_solution.set_of_dominated_solutions) >0:

                for dominated_solution in dominating_solution.set_of_dominated_solutions:
                    dominated_solution.domination_count -= 1
                    if dominated_solution.domination_count == 0:
                        current_domination_front.append(dominated_solution)
                        dominated_solution.rank = i + 2
        if len(current_domination_front) == 0:
            break
        else:
            pareto_fronts.append([i+1,current_domination_front])
    return pareto_fronts


nsga(population[:4],4,population[4:])




def circle_relocation(pointselection):
    fields = ["OBJECTID", "SHAPE@X", "SHAPE@Y"]
    coordinates = []
    with arcpy.da.SearchCursor(pointselection, fields) as cursor:
        for row in cursor:
            coordinates.append([row[0], row[1], row[2]])
    endpoints = [[coordinates[0][1],coordinates[0][2]],[coordinates[-1][1],coordinates[-1][2]]]
    distance_between_endpoints = math.sqrt(pow((endpoints[0][0] - endpoints[1][0]), 2) + pow((endpoints[0][1] - endpoints[1][1]),2))
    middle_point = [(endpoints[0][0] + endpoints[1][0])/2,(endpoints[0][1] + endpoints[1][1])/2]

    def points_on_circumference(center, radius, number_of_points):
        return [
            [
                center[0] + (math.cos(2 * math.pi / number_of_points * x) * radius),  # x
                center[1] + (math.sin(2 * math.pi / number_of_points * x) * radius)  # y

            ] for x in range(0, number_of_points + 1)]

    # for j in range(len(coordinates)):
    #     coordinates[j][1] = coordinates[j][1] + (distance_between_endpoints/2) * y[j]
    #     coordinates[j][2] = coordinates[j][2] + (distance_between_endpoints/2) * y[j]
    coordinates = np.array(coordinates)
    ring = points_on_circumference(middle_point, distance_between_endpoints/2, int( math.pi * distance_between_endpoints / 10))
    ring = np.array(ring)
    #uls.create_xy_FCFromPoints(coordinates[:,1], coordinates[:,2], r"D:\Master_Shareverzeichnis\1.Semester\Flighttaxi_project\MyProject15\MyProject15.gdb\testcirclerepair_one_part")
    uls.create_xy_FCFromPoints(ring[:, 0], ring[:, 1],
                               r"D:\Master_Shareverzeichnis\1.Semester\Flighttaxi_project\MyProject15\MyProject15.gdb\testcircle")
circle_relocation(r"D:\Master_Shareverzeichnis\1.Semester\Flighttaxi_project\MyProject15\MyProject15.gdb\circle_relocation_testpoints_one_part" )


#legal constraints parameters
maximum_speed_legal = 27.7777777778 #in m/s. 100 in km/h

#air taxi specific parameters (Lilium Jet, Electric VTOL Configurations Comparison 2018)
maximum_speed_air_taxi = 70 #(in m/s, 252 in km/h)
acceleration_speed = 2 #(in m/s)
acceleration_energy = 1.82 #(in kWh)
deceleration_speed = 2 #(in m/s)
deceleration_energy = 1.82 #(in kWh)

#flight comfort constraint
maximum_angular_speed = 1 #(in radian/second)



import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
y = np.sin(x)
tck = interpolate.splrep(x, y, s=0)
xnew = np.arange(0, 2*np.pi, np.pi/50)
ynew = interpolate.splev(xnew, tck, der=0)

plt.figure()
plt.plot(x, y, 'x', xnew, ynew, xnew, np.sin(xnew), x, y, 'b')
plt.legend(['Linear', 'Cubic Spline', 'True'])
plt.axis([-0.05, 6.33, -1.05, 1.05])
plt.title('Cubic-spline interpolation')
plt.show()