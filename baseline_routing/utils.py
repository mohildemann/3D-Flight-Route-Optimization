import numpy as np
import arcpy
import math
from math import sqrt
import sys
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
import random
import scipy.stats as stats
import datetime
from arcpy import env
from arcpy.sa import *
arcpy.env.workspace = r'C:\Users\Moritz\Desktop\Bk.gdb'
arcpy.env.outputZFlag = "Enabled"
arcpy.CheckOutExtension('Spatial')
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32118)
arcpy.env.overwriteOutput = True
from scipy.special import logsumexp
import copy
#arcpy.env.gpuId = 1






##General tools
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def remove_duplicate_points(unique_vals, input_data):
    u, indices = np.unique(unique_vals, axis=0,return_index=True)
    array_wo_duplicates = input_data[indices]
    array_wo_duplicates = array_wo_duplicates[array_wo_duplicates[:,0].argsort(kind='mergesort')]
    mask = ~np.isin(input_data[:, 0],array_wo_duplicates[:, 0])
    indices_for_deletion = input_data[mask][:,0]
    return array_wo_duplicates, indices_for_deletion

def remove_points_from_pointfc(pointfc, indices):
    with arcpy.da.UpdateCursor(pointfc, "OBJECTID") as cursor:
        for row in cursor:
            if row[0] in indices:
                cursor.deleteRow()

### Initialization
def get_random_state(seed):
    return np.random.RandomState(seed)

def check_fields_x_y_z_gridz(solution, IDW):
    if solution.representation.shape[1]<5:
        output_z_vals = r'C:\Users\Moritz\Desktop\Bk.gdb\zvals'
        solution = check_synchronization(solution)
        extractValuesRaster(solution.PointFCName, IDW, output_z_vals)
        grid_z_array = arcpy.da.FeatureClassToNumPyArray(output_z_vals, "RASTERVALU")
        grid_z_array = np.array([i[0] for i in grid_z_array]).reshape(len(grid_z_array),1)
        solution.representation = np.hstack([solution.representation,grid_z_array])
    return solution.representation

def random_pointmove_in_boundary(solution_fc,IDW, random_state, x_y_limit, sigma):
    #memory for xy
    output_xy_vals = r'C:\Users\Moritz\Desktop\Bk.gdb\output_xy_vals'
    #memory for z
    output_z_vals = r'C:\Users\Moritz\Desktop\Bk.gdb\zvals'
    infc = arcpy.GetParameterAsText(0)
    #D = arcpy.Describe(solution)
    fields =  ["OID@", "SHAPE@X","SHAPE@Y","SHAPE@Z","grid_z"]
    numpy_p = np.array([]).reshape(0,5)
    with arcpy.da.SearchCursor(solution_fc, fields) as cursor:
        for row in cursor:
            p =np.array([row[0],row[1] ,row[2],row[3],row[4]])
            numpy_p = np.vstack([numpy_p,p])
    #get the random x and y positions in buffer
    for i in range(len(numpy_p)):
        numpy_p[i][1] = random_state.uniform(numpy_p[i][1] - x_y_limit, numpy_p[i][1] + x_y_limit)
        numpy_p[i][2] = random_state.uniform(numpy_p[i][2] - x_y_limit, numpy_p[i][2] + x_y_limit)
    #convert to fc to make faster extraction possible
    create_xy_FCFromPoints(numpy_p[:,1],numpy_p[:,2],output_xy_vals)
    extractValuesRaster(output_xy_vals, IDW, output_z_vals)
    with arcpy.da.SearchCursor(output_z_vals, "RASTERVALU") as cursor:
        i = 0
        for row in cursor:
            numpy_p[i][4]=row[0]
            i = 1 + 1

    with arcpy.da.UpdateCursor(solution_fc, fields) as cursor:
        # get ready to update each row
        i = 0
        for row in cursor:
            if row[0] in numpy_p:
                # Setting a boundary which is defined by a percentage of the minimum distance to the idw height and the distance to the boarder of 700 ft.
                #guarantee that the height does not get higher than 700 ft and that the lower limit is smaller than the upper limit
                if numpy_p[i][3] >= 213:
                    numpy_p[i][3] = 212
                if numpy_p[i][4] >= 213:
                    numpy_p[i][4] = 212.9
                lower, upper = numpy_p[i][4], 213
                mu = numpy_p[i][3]
                X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                N = stats.norm(loc=mu, scale=sigma)
                random_z = stats.truncnorm.rvs((((lower - mu)) / sigma), (upper - mu) / sigma, loc=mu, scale=sigma, size=1)
                numpy_p[i][3] = random_z[0]
                if random_z <= numpy_p[i][4]:
                    numpy_p[i][3] = random_z[0]
                if random_z[0] > 213:
                     numpy_p[i][3] = 213
                random_z = None
                #Update
                row[0], row[1], row[2], row[3], row[4] = numpy_p[i][0],numpy_p[i][1],numpy_p[i][2],numpy_p[i][3],numpy_p[i][4]
                cursor.updateRow(row)
                i= i+1
    #calculate slopes
    #slopes_dist_ba =calculateSlopes_eucDist_bAngle(solution_fc)
    #add slopes to np array
    #numpy_p=np.concatenate((numpy_p, slopes_dist_ba), axis=1)
    #numpy columns  ='ID','X', 'Y', 'Z', 'grid_z','slope', eucl distance and bending angle
    return numpy_p

def check_synchronization(solution):
    result_count = arcpy.GetCount_management(solution.PointFCName)
    count_P = int(result_count.getOutput(0))
    if solution.representation.shape[0] != count_P:
        print("FC does not fit thre solution's representation. Pointcount: " + str(count_P) + ", repr_length: " + str(solution.representation.shape[0]))
        solution.representation, indices = remove_duplicate_points(solution.representation[:, 1:3], solution.representation)
        remove_points_from_pointfc(solution.PointFCName, indices)
        fields = ["OID@", "SHAPE@X", "SHAPE@Y", "SHAPE@Z"]
        numpy_p = np.array([]).reshape(0, 4)
        with arcpy.da.SearchCursor(solution.PointFCName, fields) as cursor:
            for row in cursor:
                p = np.array([row[0], row[1], row[2], row[3]])
                numpy_p = np.vstack([numpy_p, p])
        solution.representation = numpy_p
    return solution

def equalize_and_repair_representation_and_fc(solution,IDW,endpoints,sigma, geofences_restricted_airspace, geofence_point_boundary):
    output_xy_vals = r'C:\Users\Moritz\Desktop\Bk.gdb\output_xy_vals'
    if arcpy.Exists(output_xy_vals):
        arcpy.Delete_management(output_xy_vals)
    solution.representation = check_endpoints(solution.representation, endpoints)
    createFCFromPoints(solution.representation[:,1].tolist(), solution.representation[:,2].tolist(), solution.representation[:,3].tolist(), output_xy_vals)
    solution.representation = repairPointsInRestrictedAirspace(output_xy_vals, solution.representation, geofences_restricted_airspace, geofence_point_boundary)
    solution.representation, indices = remove_duplicate_points(solution.representation[:, 1:3], solution.representation)
    remove_points_from_pointfc(output_xy_vals, indices)
    #reorder feature class and representation in order thst they are in the correct geographical order
    solution.representation, reorder_is_valid = reorder_points(output_xy_vals, solution.representation, endpoints,searchradius=1500)
    arcpy.AddField_management(output_xy_vals, "grid_z", "FLOAT", 9, "", "", "grid_z")
    extractValuesRaster(output_xy_vals, IDW, solution.PointFCName)
    arcpy.Delete_management(output_xy_vals)
    solution = check_synchronization(solution)
    grid_z_array = arcpy.da.FeatureClassToNumPyArray(solution.PointFCName, "RASTERVALU")
    grid_z_array = np.array([i[0] for i in grid_z_array]).reshape(len(grid_z_array), 1)
    solution.representation = np.hstack([solution.representation, grid_z_array])
    fields = ["OID@", "SHAPE@X", "SHAPE@Y", "SHAPE@Z", "grid_z"]
    already_used = []
    with arcpy.da.UpdateCursor(solution.PointFCName, fields) as cursor:
        # get ready to update each row
        i = 0
        rows_to_delete = []
        for row in cursor:
            # Setting a boundary which is defined by a percentage of the minimum distance to the idw height and the distance to the boarder of 700 ft.
            # guarantee that the height does not get higher than 700 ft and that the lower limit is smaller than the upper limit
            if solution.representation[i][3] >= 213:
                solution.representation[i][3] = 212
            if solution.representation[i][4] >= 213:
                solution.representation[i][4] = 212.9
            if solution.representation[i][3] < solution.representation[i][4]:
                lower, upper = solution.representation[i][4], 213
                mu = solution.representation[i][3]
                X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
                N = stats.norm(loc=mu, scale=sigma)
                random_z = stats.truncnorm.rvs((((lower - mu)) / sigma), (upper - mu) / sigma, loc=mu, scale=sigma,
                                               size=1)
                solution.representation[i][3] = random_z[0]
                if random_z <= solution.representation[i][4]:
                    solution.representation[i][3] = random_z[0]
                if random_z[0] > 213:
                    solution.representation[i][3] = 213
                random_z = None
            # Update
            row[0], row[1], row[2], row[3], row[4] = solution.representation[i][0], solution.representation[i][1], solution.representation[i][2], solution.representation[i][3], \
                                                     solution.representation[i][4]
            cursor.updateRow(row)
            already_used.append(solution.representation[i][0])
            i = i + 1
    result_count = arcpy.GetCount_management(solution.PointFCName)
    count_P = int(result_count.getOutput(0))
    if solution.representation.shape[0] != count_P:
        print("FC does not fit thre solution's representation. Pointcount: " + str(count_P) + ", repr_length: "+ str(solution.representation.shape[0]) )
    return solution.representation

def check_endpoints(solution_representation, endpoints):
    endpoints_problem_instance = endpoints
    range_len = 4
    if solution_representation.shape[1] >4:
        range_len = 5
    if solution_representation[0][1] != endpoints_problem_instance[0][0] or solution_representation[0][2] != endpoints_problem_instance[0][1]:
        for i in range(1,range_len):
            solution_representation[0][i] = endpoints_problem_instance[0][i-1]
    if solution_representation[-1][1] != endpoints_problem_instance[-1][0] or solution_representation[-1][2] != endpoints_problem_instance[-1][1]:
        for i in range(1,range_len):
            solution_representation[-1][i] = endpoints_problem_instance[-1][i-1]
    return solution_representation

# Function to find the circle on
# which the given three points lie
# contributed by Ryuga on https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
def findCircle_and_radius(x1, y1, x2, y2, x3, y3):
    x12 = x1 - x2
    x13 = x1 - x3
    y12 = y1 - y2
    y13 = y1 - y3
    y31 = y3 - y1
    y21 = y2 - y1
    x31 = x3 - x1
    x21 = x2 - x1
    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)
    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)
    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)
    f = logsumexp(((sx13) * (x12) + (sy13) * (x12) + (sx21) * (x13) +  (sy21) * (x13)) // ((2 * ((y31) * (x12) - (y21) * (x13)))))

    g = (((sx13) * (y12) + (sy13) * (y12) +
          (sx21) * (y13) + (sy21) * (y13)) //
         (2 * ((x31) * (y12) - (x21) * (y13))))

    c = (-pow(x1, 2) - pow(y1, 2) -
         2 * g * x1 - 2 * f * y1)
    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    h = -g
    k = -f
    sqr_of_r = h * h + k * k - c
    # r is the radius
    r = sqrt(sqr_of_r)
    return r


def calculateSlopes_eucDist_bAngle(input_Points, fieldName_NewZ = "SHAPE@Z"):
    #print("Calculating slopes and distance...")
    arcpy.AddField_management(input_Points, "Slope","DOUBLE")
    arcpy.AddField_management(input_Points, "Euc_Distance","DOUBLE")
    arcpy.AddField_management(input_Points, "Bending_angle","DOUBLE")
    arcpy.AddField_management(input_Points, "POINT_X2","DOUBLE")
    arcpy.AddField_management(input_Points, "POINT_Y2","DOUBLE")
    arcpy.AddField_management(input_Points, "POINT_Z2","DOUBLE")
    l_moment_X = []
    l_moment_Y = []
    l_moment_Z = []

    with arcpy.da.SearchCursor(input_Points, ["SHAPE@X", "SHAPE@Y",fieldName_NewZ]) as cursor:
        for row in cursor:
            l_moment_X.append(row[0])
            l_moment_Y.append(row[1])
            l_moment_Z.append(row[2])
    flag = 1
    result_count = arcpy.GetCount_management(input_Points)
    count_P = int(result_count.getOutput(0))
    with arcpy.da.UpdateCursor(input_Points, ["OBJECTID","POINT_X2", "POINT_Y2", "POINT_Z2"]) as cursor2:
        for row in cursor2:
            if flag <  count_P:
                row[1] = l_moment_X[flag]
                row[2] = l_moment_Y[flag]
                row[3] = l_moment_Z[flag]
            elif flag == count_P:
                row[1] = 0
                row[2] = 0
                row[3] = 0
            flag += 1
            cursor2.updateRow(row)
    a_slope = [0]
    a_distances = [0]
    a_bending_angles = [0]

    flag_2 = 1
    with arcpy.da.SearchCursor(input_Points,["OBJECTID", "SHAPE@X", "POINT_X2", "SHAPE@Y","POINT_Y2", fieldName_NewZ, "POINT_Z2"]) as cursor:
        for row in cursor:
            if flag_2 < count_P :
                if row[1] == row[2]:
                    slope_value = 0
                    distance_value = 0
                    degrees_b_a = 0
                else:
                    slope_value = ((row[5]-row[6]) / pow(pow((row[1]-row[2]),2) + pow((row[3]-row[4]),2), 0.5))*100
                    distance_value = math.sqrt(pow((row[1]-row[2]),2) + pow((row[3]-row[4]),2) + pow((row[5]-row[6]),2))
                    if (row[2]-row[1]) == 0 or (row[4]-row[3]) == 0:
                        degrees_b_a = 0
                    else:
                        radian_b_a = math.atan((row[2]-row[1])/(row[4]-row[3]))
                        degrees_b_a = radian_b_a * 180 /math.pi
                a_slope.append(slope_value)
                a_distances.append(distance_value)
                a_bending_angles.append(degrees_b_a)
                flag_2 = flag_2 +1
    flag3 = 0
    with arcpy.da.UpdateCursor(input_Points,["OBJECTID", "SHAPE@X", "POINT_X2", "SHAPE@Y","POINT_Y2", fieldName_NewZ, "POINT_Z2", "Slope", "Euc_Distance", "Bending_angle"]) as cursor:
        for row in cursor:
            row[7] = a_slope[flag3]
            row[8] = a_distances[flag3]
            row[9] = a_bending_angles[flag3]
            flag3 += 1
            cursor.updateRow(row)

    numpy_slopes_eucl_ba = np.column_stack([a_slope, a_distances, a_bending_angles])
    a_slope = None
    a_distances = None
    a_bending_angles = None
    return numpy_slopes_eucl_ba
###Geoprocessing utils

def createFCFromPoints(array_x, array_y,array_z, output_NP):
    #print("Create New Feature Class From Points")
    if arcpy.Exists(output_NP):
        arcpy.Delete_management(output_NP)
    sr = arcpy.env.outputCoordinateSystem
    points = [arcpy.PointGeometry(arcpy.Point(array_x[f], array_y[f],array_z[f]),sr, True) for f in range(len(array_x))]
    arcpy.CopyFeatures_management(points, output_NP)

def create_xy_FCFromPoints(array_x, array_y, output_NP):
    if arcpy.Exists(output_NP):
        arcpy.Delete_management(output_NP)
    #print("Create New Feature Class From Points")
    sr = arcpy.env.outputCoordinateSystem
    points = [arcpy.PointGeometry(arcpy.Point(array_x[f], array_y[f]),sr, True) for f in range(len(array_x))]
    arcpy.CopyFeatures_management(points, output_NP)


def repair_points(nr_repairtrials,points,idw,random_state):
    repaired_points = []
    points_to_delete = []
    i= 0
    p1=len(repaired_points)
    p2=len(points)
    for point in points:
        j=0
        while np.isin(point, repaired_points).all() == False and j<=nr_repairtrials:
            j = j+1
            x=point[1]
            y=point[2]
            z=point[3]
            randx = random_state.uniform(point[1] - 20, point[1] + 20)
            point[1] = random_state.uniform(point[1] - 20, point[1] + 20)
            point[2] = random_state.uniform(point[2] - 20, point[2] + 20)
            point[4] =capturesingleValueZ(idw, point[1], point[2])
            point[3] = point[4] + 5
            if 213-point[3]>10 and point[3]>point[4]:
                repaired_points.append(point)
                #print("Point repaired")
            elif np.isin(point, repaired_points).all() == True:
                repaired_points.append(point)
                #print("Point repaired")
        if j>= nr_repairtrials:
            points_to_delete.append(point)
            print("Point needs to be deleted")
    if len(repaired_points)==len(points):
        valid = True
    else:
        valid = False
    repaired_points = np.array([repaired_points]).reshape(len(repaired_points),8)
    points_to_delete = np.array([points_to_delete]).reshape(len(points_to_delete), 8)
    return repaired_points,points_to_delete, valid

def delete_old_objects_from_gdb(wildcard):

    for objFeatureClass in arcpy.ListFeatureClasses():
        desc = arcpy.Describe(objFeatureClass)
        if wildcard in desc.Name:
            try:
                arcpy.Delete_management(objFeatureClass)
            except:
                print("Delete of fc didnt work")
    rasters = arcpy.ListRasters("*", "ALL")
    for raster in rasters:
        if wildcard in raster:
            try:
                arcpy.Delete_management(raster)
            except:
                print("Delete of raster didnt work")
def extractValuesRaster(feature_New, idw, output_extract):
    #print("Extract Z Value from the Raster using the newfeature class")
    if len(arcpy.ListFields(feature_New, "RASTERVALU")) > 0:
        arcpy.DeleteField_management(feature_New, ["RASTERVALU"])
    if arcpy.Exists(output_extract):
        arcpy.Delete_management(output_extract)
    arcpy.gp.ExtractValuesToPoints_sa(feature_New, idw, output_extract, 'NONE', 'ALL')

def extractValues_many_Rasters(fc, idw_w_fieldnames):
    arcpy.sa.ExtractMultiValuesToPoints(fc, idw_w_fieldnames, "NONE")

def calculate_added_noise(solution_representation, aircraft_noise_array, ground_noise_array):
    # step 1: calculate noise pressure change to position at the ground. Formula of calculating sound pressure: abs(20 log (R1/R2)), R1 = 1m, R2 = current height of aircraft
    #noise_at_ground = 20 * math.log(solution_representation[:,2])
    def f(x):
        return abs(20 * math.log(1/x,10))
    solution_representation[:, 3] = np.where(solution_representation[:, 3]==0, 0.01, solution_representation[:, 3])
    f2  = np.vectorize(f)
    noise_vector = f2(solution_representation[:, 3])
    noise_vector = np.array(noise_vector).reshape(len(noise_vector),1)
    #print()
    #step 2: calculate current noise of aircraft - volume change at the ground
    noise_at_ground = aircraft_noise_array - noise_vector
    #step 3: calculate added noise. Formula for adding several noise sources: 10 log10 *  sum(for each noise source: 10 exp(Noise / 10))
    def noise_addition(x,y):
        combined_noise = 10 * math.log((math.pow(10, x / 10))+(math.pow(10, y / 10)),10)
        return combined_noise
    vectorized_noise = np.vectorize(noise_addition)
    v = vectorized_noise(noise_at_ground,ground_noise_array.reshape(ground_noise_array.shape[0],1))
    added_noise_array = abs(ground_noise_array.reshape(ground_noise_array.shape[0],1) - v)
    mask = np.all(np.isnan(added_noise_array), axis=1)
    added_noise_array = added_noise_array[~mask]
    avg_added_noise = np.mean(added_noise_array)
    if math.isnan(avg_added_noise):
        print("nan is added noise 3")
    return avg_added_noise

def captureValueZ(raster_lay, values_x, values_y):
    # Slow Approach for many points
    new_Z_Values = []
    for index in range(len(values_x)):
        z_new_val = arcpy.GetCellValue_management(raster_lay, str(values_x[index]) + " " + str(values_y[index]))
        new_Z_Values.append(z_new_val.getOutput(0))
    #print(new_Z_Values)
    new_Z_f = [float(i.replace(",", ".")) for i in new_Z_Values]
    return new_Z_f

def pointsToLine(point_feature, output_Line_f):
    arcpy.PointsToLine_management(point_feature, output_Line_f, None, "OBJECTID", "NO_CLOSE")

def lineToPoints(line_feature, output_points, distance):
    distance = str(distance) + " Meters"
    arcpy.management.GeneratePointsAlongLines(line_feature, output_points, "DISTANCE", distance, None, "END_POINTS")

def smoothline(line_fc_name, output_line, smooth_factor, airspace_restriction):
    smooth_factor = str(smooth_factor) + " Meters"
    arcpy.cartography.SmoothLine(line_fc_name, output_line, "PAEK", smooth_factor, "FIXED_CLOSED_ENDPOINT", "NO_CHECK", airspace_restriction)

def interpolate_points_to_spline_surface(input_points, out_raster, cellsize):
    arcpy.AddField_management(input_points, "n_height", "FLOAT", 9, "", "", "n_height")
    with arcpy.da.UpdateCursor(input_points, ["SHAPE@Z","n_height"]) as cursor:
        # get ready to update each row
        i = 0
        rows_to_delete = []
        for row in cursor:
            row[1] = row[0]
            cursor.updateRow(row)
    arcpy.Spline_3d(input_points, "n_height", out_raster, cellsize, "REGULARIZED", 0.1, 12)

def point3d_fc_to_np_array(point_fc_name, additional_fields = None):
    fields = ["OBJECTID","SHAPE@X", "SHAPE@Y", "SHAPE@Z"]
    if additional_fields is not None:
        fields.extend(additional_fields)
    x_y_z_array = arcpy.da.FeatureClassToNumPyArray(point_fc_name, fields)
    x_y_z_array = np.array([[i[j] for j in range(len(fields))] for i in x_y_z_array]).reshape(
        len(x_y_z_array), len(fields))
    return  x_y_z_array

def calculate_max_speed_for_given_radius_and_max_g_force(max_g_force, radius):
    #calculate maximal velocity for given maximum gforce and given radius
    vmax = math.sqrt(max_g_force*radius*9.81)
    return vmax

def calculate_speed_limits(flight_points_x_y_z_array, maximum_angular_speed,max_g_force,maximum_speed_air_taxi, maximum_speed_legal):
    speed_limit_list = []
    #speed limit starting point is 0
    speed_limit_list.append(0)
    maxv_by_angular_speed = []
    nan_values = []
    for i in range(1,flight_points_x_y_z_array.shape[0]-1):
        radius = findCircle_and_radius(flight_points_x_y_z_array[i-1][0],flight_points_x_y_z_array[i-1][1],flight_points_x_y_z_array[i][0],flight_points_x_y_z_array[i][1],flight_points_x_y_z_array[i+1][0],flight_points_x_y_z_array[i+1][1])
        if np.isnan(radius):
            nan_values.append(i)
            radius = sys.float_info.max
        velocity_limit_by_max_g_force = calculate_max_speed_for_given_radius_and_max_g_force(max_g_force,radius)
        velocity_limit_by_angular_speed = maximum_angular_speed * radius
        maxv_by_angular_speed.append(velocity_limit_by_angular_speed)
        speed_limit_list.append(min(velocity_limit_by_angular_speed,maximum_speed_air_taxi, maximum_speed_legal))
    speed_limit_list.append(0)
    maxv_by_angular_speed= np.array(maxv_by_angular_speed)
    speed_limit_list = np.array(speed_limit_list).reshape(len(speed_limit_list),1)
    flight_points_x_y_z_speedlimit_array = np.hstack([flight_points_x_y_z_array,speed_limit_list])
    #evtl: slope berechnen und ebenfalls max_velocity_berechnen
    return flight_points_x_y_z_speedlimit_array

def calculate_speed_and_energy_consumption(flight_points_x_y_z_maxv, flight_constraints):
    def get_distance_between_points(p1,p2) :
        #point: (x,y,z)
        return math.sqrt(
            pow((p1[0] - p2[0]), 2)
            + pow((p1[1] - p2[1]), 2)
            + pow((p1[2] - p2[2]), 2)) / 2

    def get_time_for_evenly_accelerated_movement_w_given_distance(distance, acceleration):
        return math.sqrt((2*distance)/abs(acceleration))

    def get_velocity_after_given_time_w_given_acceleration(v0, time, acceleration):
        return v0 + (time * acceleration)

    def get_velocity_after_given_distance_w_given_acceleration(v0, distance, acceleration):
        #calculates the velocity after travelling the given distance with the specified acceleration
        if 2* acceleration * distance + math.pow(v0,2) <= 0:
            acceleration = abs(acceleration)
        if v0 > 0:
            updated_velocity = math.sqrt(2* acceleration * distance + math.pow(v0,2))
        else:
            try:
                updated_velocity = math.sqrt(2 * acceleration * distance)
            except:
                updated_velocity = 0.01
        return updated_velocity

    def get_distance_needed_for_given_acceleration_or_deceleration(v1,v2,acceleration):
        return abs(((0.5 * math.pow(v1,2)) - (0.5 * math.pow(v2,2)))/ acceleration)

    def calc_angle_of_climb(p1, p2):
        if p1[2] == p2[2] or (p1[1] == p2[1] and p1[0] == p2[0]):
            slope_value = 0
        else:
            slope_value = ((p1[2] - p2[2]) / pow(pow((p1[0] - p2[0]), 2) + pow((p1[1] - p2[1]), 2), 0.5)) * 100
        return slope_value

    array_current_velocity = []
    array_current_noise = []
    array_current_velocity.append(0)
    array_current_noise.append(flight_constraints.noise_at_hover)
    array_current_energy_consumption = []
    array_distance_to_next_point =[]
    for i in range(flight_points_x_y_z_maxv.shape[0]-1):
        # compute distance to next point. save distance in array array_distance_to_next_point
        distance = get_distance_between_points(flight_points_x_y_z_maxv[i],flight_points_x_y_z_maxv[i+1])
        array_distance_to_next_point.append(distance)
        #check if we can accelerate at the current point because allowed speed of next point is higher:
        if flight_points_x_y_z_maxv[i+1][4] > array_current_velocity[-1]:
            #accelerate with given acceleration speed
            time = get_time_for_evenly_accelerated_movement_w_given_distance(distance, flight_constraints.acceleration_speed)
            velocity = get_velocity_after_given_distance_w_given_acceleration(array_current_velocity[-1], distance, flight_constraints.acceleration_speed)
            # check if possible reached speed after acceleration is higher then max allowed speed at current position
            if velocity > flight_points_x_y_z_maxv[i+1][4]:
                velocity = flight_points_x_y_z_maxv[i+1][4]
            array_current_velocity.append(velocity)
            #calculate the energy consumption for the needed time for accelerating between last point to current point
            energy_consumed_in_kWh = flight_constraints.acceleration_energy * (time/3600)
            array_current_energy_consumption.append(energy_consumed_in_kWh)
            array_current_noise.append(flight_constraints.noise_pressure_acceleration)

        elif flight_points_x_y_z_maxv[i+1][4] == array_current_velocity[-1] and flight_points_x_y_z_maxv[i+1][4] == np.amax(flight_points_x_y_z_maxv[:,4]):
            array_current_velocity.append(array_current_velocity[-1])
            #fly at current speed
            if flight_constraints.type_aircraft == "multicoptor":
                energy = flight_constraints.hover_energy
            else:
                angle_of_climb = calc_angle_of_climb(flight_points_x_y_z_maxv[i], flight_points_x_y_z_maxv[i+1])
                try:
                    energy = flight_constraints.calculate_required_energy_at_level_speed(angle_of_climb,array_current_velocity[-1])
                    if energy > flight_constraints.hover_energy:
                        energy = flight_constraints.hover_energy
                except:
                    angle = angle_of_climb
                    vel = array_current_velocity[-1]
            time = distance / array_current_velocity[-1]
            energy_consumed_in_kWh = energy * (time/3600)
            array_current_energy_consumption.append(energy_consumed_in_kWh)
            array_current_noise.append(flight_constraints.noise_at_cruise)
        else:
            #allowed speed at next point is lower then current speed
            #decelerate with given deceleration speed. calculate distance which is needed for decelerating for the speed delta. Find previous point where that distance is reached. Update the velocities for this point range.
            needed_distance_for_deceleration =  get_distance_needed_for_given_acceleration_or_deceleration(array_current_velocity[-1],flight_points_x_y_z_maxv[i+1][4],flight_constraints.deceleration_speed)
            update_distance = distance

            # add a value with the maximum allowed speed level at the next point and add the energy consumed by decelerating to it
            array_current_velocity.append(get_velocity_after_given_distance_w_given_acceleration(flight_points_x_y_z_maxv[i+1][4], distance,
                                                                       flight_constraints.deceleration_speed))
            try:
                time = distance / array_current_velocity[-1]
            except:
                time = 0.000001
            energy_consumed_in_kWh = flight_constraints.deceleration_energy * (time / 3600)
            array_current_energy_consumption.append(energy_consumed_in_kWh)
            array_current_noise.append(flight_constraints.noise_pressure_deceleration)
            for j in range(i):
                #if the needed distance for decelerating to the next point is higher then the total distance of going backwards, then we need to go back.
                # we go back to the last saved velocities and update them.
                # Approach: from the allowed maximal speed that makes it necesary to decelerate, we acelerate from that point with the given deceleration speed going backwards in the array.
                velocity = get_velocity_after_given_distance_w_given_acceleration(array_current_velocity[i-j+1], array_distance_to_next_point[i-j], abs( flight_constraints.deceleration_speed))
                # check if possible reached speed after acceleration is higher then max allowed speed at current position
                if velocity >= array_current_velocity[i-j]:
                    break
                array_current_velocity[i-j] =velocity
                # calculate the energy consumption for the needed time for accelerating between last point to current point
                energy_consumed_in_kWh = flight_constraints.acceleration_energy * (time / 3600)
                array_current_energy_consumption[i-j] = energy_consumed_in_kWh
                array_current_noise[i-j] = flight_constraints.noise_pressure_deceleration

    #add energy consumption and velocity for last point
    time = get_time_for_evenly_accelerated_movement_w_given_distance(array_distance_to_next_point[-1], flight_constraints.deceleration_speed)
    energy_consumed_in_kWh = flight_constraints.acceleration_energy * time / 3600
    array_current_energy_consumption.append(energy_consumed_in_kWh)
    #array_current_velocity.append(0)
    #array_current_noise.append(flight_constraints.noise_at_hover)
    #convert to numpy
    array_current_energy_consumption = np.array(array_current_energy_consumption).reshape(len(array_current_energy_consumption),1)
    array_current_velocity = np.array(array_current_velocity).reshape( len(array_current_velocity),1)
    array_distance_to_next_point = np.array(array_distance_to_next_point ).reshape( len(array_distance_to_next_point),1)
    array_current_noise = np.array(array_current_noise).reshape(len(array_current_noise),1)
    total_energy_consumption = np.sum(array_current_energy_consumption)
    total_distance = np.sum(array_distance_to_next_point)
    avg_velocity = np.mean(array_current_velocity)
    flight_time = total_distance / avg_velocity
    return array_distance_to_next_point, array_current_velocity, array_current_energy_consumption, array_current_noise, total_energy_consumption, flight_time

def extractValues_Z(fc_Obj, field_Z):
    # Fast Approach for many points
    #("Extract the new Z Values")
    val_z_New = []
    with arcpy.da.SearchCursor(fc_Obj,field_Z) as cursor:
        for row in cursor:
            val_z_New.append(row[0])
    return val_z_New

def capturesingleValueZ(raster_lay, value_x, value_y):
    z_new_val = arcpy.GetCellValue_management(raster_lay, str(value_x) + " " + str(value_y))
    #print(new_Z_Values)
    new_Z_f = float(z_new_val.getOutput(0).replace(",", "."))
    return new_Z_f

def updateFeaturePoint(feature_update, new_array, new_fcname):
    #because of dynamic length, an update cursor is not possible. So first delete rows and then insert with cursor
    template = arcpy.CopyFeatures_management(feature_update, new_fcname)
    arcpy.Delete_management(feature_update)
    arcpy.CreateFeatureclass_management(arcpy.env.workspace, feature_update, template =template,has_m="ENABLED", has_z="ENABLED")
    fields = ["SHAPE@X", "SHAPE@Y","SHAPE@Z"]
    cursor = arcpy.da.InsertCursor(feature_update, fields)
    for x in range(new_array.shape[0]):
        cursor.insertRow((new_array[x][0], new_array[x][1], new_array[x][2]))
    # Delete cursor object
    del cursor
    #print("Feature updated!!!")
    return new_fcname


def delete_fc_from_old_generation(all_fc_names, array_to_keep):
    for fc_class in all_fc_names:
        if fc_class not in array_to_keep:
            arcpy.Delete_management(fc_class)

def updateLine(feature_update, feature_line):
    a_z_values = []
    with arcpy.da.SearchCursor(feature_update,["SHAPE@Z"]) as cursor:
        for row in cursor:
            a_z_values.append(row[0])
    arcpy.XYToLine_management(feature_update, feature_line, 'SHAPE@X', 'SHAPE@Y', 'POINT_X2', 'POINT_Y2', 'GEODESIC', '#', "PROJCS['NAD_1983_StatePlane_New_York_Long_Island_FIPS_3104_Feet',GEOGCS['GCS_North_American_1983',DATUM['D_North_American_1983',SPHEROID['GRS_1980',6378137.0,298.257222101]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Lambert_Conformal_Conic'],PARAMETER['False_Easting',984250.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',-74.0],PARAMETER['Standard_Parallel_1',40.66666666666666],PARAMETER['Standard_Parallel_2',41.03333333333333],PARAMETER['Latitude_Of_Origin',40.16666666666666],UNIT['Foot_US',0.3048006096012192]];-120039300 -96540300 3048,00609601219;-100000 10000;-100000 10000;3,28083333333333E-03;0,001;0,001;IsHighPrecision")
    arcpy.AddField_management(feature_line, "Z", "DOUBLE",has_z = "ENABLED")
    flag_s1 = 0
    with arcpy.da.UpdateCursor(feature_line, "Z") as cursor:
        for row in cursor:
            row[0] = a_z_values[flag_s1]
            flag_s1 += 1
            cursor.updateRow(row)
    expression = "OID = " + str(arcpy.GetCount_management(feature_line))

    with arcpy.da.UpdateCursor(feature_line, "OID", where_clause = expression) as cursor:
        for row in cursor:
            cursor.deleteRow()
            #print("deleted")

def reorder_points(solution_fc, solution_repr,endpoints, searchradius):
    in_features = solution_fc
    reorder_is_valid = True
    near_features = solution_fc
    tables = arcpy.ListTables()
    for table in tables:
        if "Near_Table_Reorder" in table:
            arcpy.Delete_management(table)
    out_table = arcpy.env.workspace+"\\Near_Table_Reorder"
    # optional parameters
    search_radius = str(searchradius)+ " Meters"
    location = 'LOCATION'
    angle = 'ANGLE'
    closest = 'ALL'
    closest_count = 10
    table = arcpy.GenerateNearTable_analysis(in_features, near_features, out_table, search_radius,
                                             location, angle, closest, closest_count)
    arr = arcpy.da.TableToNumPyArray(out_table,('*'))
    arr = arr.reshape(arr.shape[0], 1)
    arr = np.array([list(i[0]) for i in arr])
    #df = pd.DataFrame(arr,dtype=float, columns=['ID', 'IN_FID','NEAR_ID', 'NEAR_DIST', 'NEAR_RANK', 'IN_X', 'IN_Y', 'NEAR_X','NEAR_Y','NEAR_ANGLE'])
    # Columns of near table
    # 1 = IN_FID
    # 2 = NEAR_ID
    # 3 = NEAR_DIST
    # 4 = NEAR_RANK
    # 5 = IN_X
    # 6 = IN_Y
    # 7 = NEAR_X
    # 8 = NEAR_Y
    # 9 = NEAR_ANGLE
    ordered_list = []
    #search position of starting node
    #searching start node (uncertainty due to tolerance in point features does allow equal match)
    x_y_cols_int = solution_repr[:, [1,2]].astype(int)
    result = np.where(
        (x_y_cols_int[:,0] == int(endpoints[0][0])) & (x_y_cols_int[:,1] ==int(endpoints[0][1])))
    if isinstance(result, tuple):
        result = result[0][0]
    first_point =solution_repr[result]
    rank_id = 1
    ordered_list = np.array(first_point).reshape(1,solution_repr.shape[1])
    #logic:
    # loop until the starting and endpoint are included in the list.
    # start with starting point
    #   loop until condition found: the neighbor point is the closest point (lowest rank) that does not exist in the ordered list yet
    #   so if the nearest point already exists in the array, the next closest point is tested
    counter = 0
    while (int(endpoints[-1][0]) not in ordered_list[:,1].astype(int))is True or (int(endpoints[-1][1]) not in ordered_list[:,2].astype(int)) is True:
        rank_id = 1
        while rank_id <= closest_count:
            result = np.where((arr[:,1] == ordered_list[-1][0]) & (arr[:,4] <= rank_id)& (np.in1d(arr[:,2], ordered_list[:,0]) ==False))
            if len(result[0])>=1:
                #look for the wanted id in the solution.representation and add it to the ordered list
                try:
                    value_to_append = np.squeeze( solution_repr[np.where(solution_repr[:,0]==arr[result[0][0]][2])],axis=0)
                    ordered_list = np.vstack([ordered_list, value_to_append])
                except:
                    s = solution_repr[np.where(solution_repr[:,0]==arr[result[0][0]][2])]
                break
            else:
                rank_id += 1
        counter = counter + 1
        if counter >= (solution_repr.shape[0]*10):
            solution_repr = ordered_list
            print("Reorder didnt work")
            createFCFromPoints(ordered_list[:, 1].tolist(), ordered_list[:, 2].tolist(),
                               ordered_list[:, 3].tolist(), "threedptest")
            reorder_is_valid = False
            break
    if int(endpoints[-1][0]) in ordered_list[:,1].astype(int) and int(endpoints[-1][1]) in ordered_list[:,2].astype(int):
        solution_repr = ordered_list
        solution_repr[:,0] = np.array([i + 1 for i in range(solution_repr.shape[0])])
        if arcpy.Exists(solution_fc):
            arcpy.Delete_management(solution_fc)
        createFCFromPoints(solution_repr[:,1].tolist(), solution_repr[:,2].tolist(), solution_repr[:,3].tolist(), solution_fc)
    else:
        print("something went wrong with reordering. Endpoint not included")
        reorder_is_valid = False
    return solution_repr, reorder_is_valid

def getTheNewLine(feature_update, geofences_vector, out_sort_Line):
    out_identity = r'in_memory\test_points_identity'
    #out_sort_Line = r'D:\Geotech_Classes\GIS_App\Copy_Project\Moritz_Bk\Bk.gdb\test_new_points'
    expression = "h_buffer_m >= 300"
    arcpy.Identity_analysis(feature_update, geofences_vector, out_identity, 'ALL', '#', 'NO_RELATIONSHIPS')
    with arcpy.da.UpdateCursor(out_identity, "h_buffer_m", where_clause = expression) as cursor:
        for row in cursor:
            cursor.deleteRow()
            #print("deleted")
    arcpy.Sort_management(out_identity, out_sort_Line, 'FID_' + (feature_update.split("\\")[-1]) + ' ASCENDING', 'UR')
    return out_sort_Line

def select_points_in_restricted_airspace(flightpath_points, geofences_restricted_airspace):
    Selection = arcpy.SelectLayerByLocation_management(flightpath_points, "within",
                                                       geofences_restricted_airspace,None, "NEW_SELECTION", "NOT_INVERT")
    matchcount = int(arcpy.GetCount_management(Selection)[0])
    print(matchcount)
    return Selection, matchcount

def move_points_out_of_restrcited_airspace(points_in_restricted_airspace,flightpaths_points,flightpath_numpy_representation, geofence_point_boundary):
    # The points were created by making a 10m buffer of the totally restricted air space (above 213 m).
    # These were conversed to lines. Points along these lines (all 50m) were created)
    new_coordinates = []
    arcpy.analysis.Near(points_in_restricted_airspace, geofence_point_boundary, "500 Meters", "LOCATION",
                        "NO_ANGLE", "PLANAR")
    fields = ["OBJECTID", "NEAR_X", "NEAR_Y"]
    with arcpy.da.SearchCursor(points_in_restricted_airspace, fields) as cursor:
        for row in cursor:
            new_coordinates.append([row[0], row[1], row[2]])
    new_coordinates = np.array(new_coordinates).reshape(len(new_coordinates), 3)
    fields = ["OBJECTID", "SHAPE@X", "SHAPE@Y"]
    with arcpy.da.UpdateCursor(flightpaths_points, fields) as cursor:
        # get ready to update each row
        to_delete1 = []
        to_delete2 = []
        for row in cursor:
            for i in range(len(new_coordinates)):
                if row[0] == new_coordinates[i][0]:
                    #if the values in the near table are -1 in nearx and neary position, then there was no near point in the specified near distance (500m).
                    # in this case, the point can be deleted
                    if new_coordinates[i][1] == -1 or new_coordinates[i][2] == -1:
                        cursor.deleteRow()
                        del_pos1 = np.where(new_coordinates == row[0])
                        del_pos1 = del_pos1[0][0]
                        to_delete1.append(del_pos1)
                        del_pos2 = np.where(flightpath_numpy_representation[:,0] == row[0])
                        del_pos2 = del_pos2[0][0]
                        to_delete2.append(del_pos2)
                    #update the feature class

    new_coordinates = np.delete(new_coordinates, to_delete1, axis=0)
    flightpath_numpy_representation = np.delete(flightpath_numpy_representation, to_delete2, axis=0)
    with arcpy.da.UpdateCursor(flightpaths_points, fields) as cursor:
        # get ready to update each row
        for row in cursor:
            for i in range(new_coordinates.shape[0]):
                if row[0] == new_coordinates[i][0]:
                    row[1] = new_coordinates[i][1]
                    row[2] = new_coordinates[i][2]
                    pos = np.where(flightpath_numpy_representation == row[0])
                    pos = pos[0][0]
                    #update the numpy representation of the flightpaths (solution.representation)
                    flightpath_numpy_representation[pos][1] = new_coordinates[i][1]
                    flightpath_numpy_representation[pos][2] = new_coordinates[i][2]
                    cursor.updateRow(row)
    return flightpath_numpy_representation

def repairPointsInRestrictedAirspace(flightpath_points, flightpath_numpy_representation,geofences_restricted_airspace,geofence_point_boundary):

    flightpaths_points_in_restricted_airspace, pointcount_in_restricted_airspace = select_points_in_restricted_airspace(flightpath_points,
                                                                                     geofences_restricted_airspace)
    new_representation = move_points_out_of_restrcited_airspace(flightpaths_points_in_restricted_airspace,flightpath_points,flightpath_numpy_representation, geofence_point_boundary)
    arcpy.management.SelectLayerByAttribute(flightpath_points, "CLEAR_SELECTION")
    return new_representation

def repairLinesInRestrictedAirspace(flightpath_line, flightpath_points,flightpath_numpy_representation, geofences_restricted_airspace):
    arcpy.analysis.Intersect([flightpath_line,geofences_restricted_airspace], flightpath_points+"intP","ALL", None, "POINT")
    fields = ["OBJECTID", "SHAPE@X", "SHAPE@Y"]
    #explode multi point to single point
    arcpy.MultipartToSinglepart_management( flightpath_points+"intP", flightpath_points+"sintP")
    _counter = 0
    new_coordinates = []
    with arcpy.da.SearchCursor(
            flightpath_points+"sintP",
            fields) as cursor:
        for row in cursor:
            new_coordinates.append([row[0], row[1], row[2]])
    new_coordinates = np.array(new_coordinates)
    geofence_point_boundary = "Geofence_Points_For_Relocation_10m"
    #loop in stepsize 2: one is the first point before geofence intersection, one the point at the end of the intersection
    for i in range(0, len(new_coordinates - 2), 2):
        _counter += 1
        createRectangleBetweenPoints(new_coordinates[i][1:], new_coordinates[i + 1][1:],flightpath_points+"rect_"+str(_counter))
        pol_boundary_to_points(flightpath_points + "rect_" + str(_counter),
                               flightpath_points + "bound_p_" + str(_counter), 10)
        #calculate distance of the two points building the rectangle:
        dist = math.sqrt(pow((new_coordinates[i][1] - new_coordinates[i+1][1]), 2)
            + pow((new_coordinates[i][2] - new_coordinates[i+1][2]), 2))
        #calculate max. possible distance the point can be away from the border
        max_possible_dist = math.sqrt(pow(dist,2) + pow(dist/2,2))
        #missing 1: intersect with geofence
        arcpy.analysis.Intersect("threedpoints_1575473371_ph_3drect_0 #;Restricted_Airspace #",
                                 r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\t_3", "ALL", None,
                                 "INPUT")

        #missing 2: multipart to Singlepart:
        arcpy.management.MultipartToSinglepart("t_3_PolToLine",
                                               r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\t_3__MultipartToSin")

        #missing 3: split line at point
        arcpy.management.SplitLineAtPoint("t_3__MultipartToSin", "intersection_singlepoints3", r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\t_3__MultipartToSin_SplitLin", "20 Meters")

        #missing 4: check which line of the splitted ones connects to the points used for creating the rectangle
        #generate near table
        arcpy.analysis.GenerateNearTable("mega_single_points", "t_3__MultipartToSin_SplitLin",
                                         r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\Geofence_Points_For_Reloca2",
                                         "50 Meters", "NO_LOCATION", "NO_ANGLE", "ALL", None, "PLANAR")
        #calculate which NEAR_FID has the smalles average NEAR_DIST

        #missing 5: split points at lines

        #missing 6: move to point boundary

        pol_boundary_to_points(flightpath_points+"rect_"+str(_counter),flightpath_points+ "bound_p_"+str(_counter), 10)
        select_points_at_geofence_boundary(flightpath_points+"bound_p_"+str(_counter), geofences_restricted_airspace, flightpath_points+"rep_p_"+str(_counter))
        update_coordinates = []
        arcpy.analysis.Near(flightpath_points+"rep_p_"+str(_counter), geofence_point_boundary, str(int(max_possible_dist))+ " Meters", "LOCATION",
                            "NO_ANGLE", "PLANAR")
        fields = ["OBJECTID", "NEAR_X", "NEAR_Y", "NEAR_FID"]
        fields_point_boundary = ["OBJECTID", "grid_z", "noise"]
        with arcpy.da.SearchCursor(flightpath_points+"rep_p_"+str(_counter), fields) as cursor:
            for row in cursor:
                # get the grid_z and noise from the boundary points
                with arcpy.da.SearchCursor(geofence_point_boundary, fields_point_boundary) as cursor2:
                    for row2 in cursor2:
                        if row2[0] == row[3]:
                            #row[0] = id, row[1] = x, row[2] = y of closest borderpoint
                            # row2[1] = grid_z, row2[2] = noise
                            update_coordinates.append([row[0], row[1], row[2],row2[1],row2[2]])



        update_coordinates = np.array(update_coordinates).reshape(len(update_coordinates), 5)
        update_coordinates, indices = remove_duplicate_points(update_coordinates[:,1:], update_coordinates)
        fields = ["OBJECTID", "SHAPE@X", "SHAPE@Y"]
        #the following loop updates the points to the closest point of the geofence boundary
        with arcpy.da.UpdateCursor(flightpath_points+"rep_p_"+str(_counter), fields) as cursor:
            # get ready to update each row
            to_delete1 = []
            for row in cursor:
                #delete points that are not within specified minimum range
                for i in range(len(update_coordinates)):
                    if row[0] == update_coordinates[i][0]:
                        # if the values in the near table are -1 in nearx and neary position, then there was no near point in the specified near distance.
                        # in this case, the point can be deleted
                        if update_coordinates[i][1] == -1 or update_coordinates[i][2] == -1:
                            cursor.deleteRow()
                            del_pos1 = np.where(update_coordinates == row[0])
                            del_pos1 = del_pos1[0][0]
                            to_delete1.append(del_pos1)
                update_coordinates = np.delete(update_coordinates, to_delete1, axis=0)


        #insert the points into the shapefile (not the placeholder numpy representation)
        fields = ["SHAPE@X", "SHAPE@Y", "SHAPE@Z", "grid_z","int_z", "noise"]
        cursor = arcpy.da.InsertCursor(flightpath_points , fields)
        max_index = np.max(update_coordinates[:,0])
        for i in range(update_coordinates.shape[0]):
            #insert in point fc
            cursor.insertRow((update_coordinates[i][1],update_coordinates[i][2],update_coordinates[i][3],update_coordinates[i][3],update_coordinates[i][3], update_coordinates[i][4]))
            flightpath_numpy_representation = np.insert(flightpath_numpy_representation, int(update_coordinates[i][0]),[update_coordinates[i][0]+i,update_coordinates[i][1],update_coordinates[i][2],update_coordinates[i][3],update_coordinates[i][3], update_coordinates[i][4]], axis=0)
        # Delete cursor object
        del cursor
        #reindex in order to have unique ids
        flightpath_numpy_representation[:, 0] = np.array([i + 1 for i in range(flightpath_numpy_representation.shape[0])])
    return flightpath_numpy_representation


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

def select_points_at_geofence_boundary(target_fc, join_fc, out_fc):
    #hier muss weiter gemacht werden: spatial join+delete
    tempLayer = str(out_fc) + "temp"
    arcpy.SpatialJoin_analysis(target_features=target_fc,join_features= join_fc,out_feature_class= out_fc ,match_option= "WITHIN")
    arcpy.MakeFeatureLayer_management(out_fc, tempLayer)
    arcpy.SelectLayerByLocation_management(in_layer=tempLayer,overlap_type="COMPLETELY_WITHIN",select_features=join_fc,search_distance=None,
                                           selection_type="NEW_SELECTION",invert_spatial_relationship= "NOT_INVERT")
    arcpy.DeleteFeatures_management(tempLayer)



def createRectangleBetweenPoints(a, b, rect_fc_name):
    arcpy.env.outputZFlag = "Disabled"
    pointA = arcpy.Point(float(a[0]), float(a[1]))
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

def copy_old_generation(population):
    copy_population = []
    for solution in population:
        copy_population.append(solution)
    return copy_population

def line_repair(geofences,flight_path_line, flight_path, geofence_point_boundary,placeholder_interpolated_surface, noisemap,IDW,endpoints, output_name):
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
        out_table_intersection_point = "NearTableIntersectionPointsToBoundaryPoints"
        arcpy.analysis.GenerateNearTable(intersection_points, boundary_points,
                                         out_table_intersection_point,
                                         "200 Meters", "LOCATION", "NO_ANGLE", "ALL", 2, "PLANAR")
        array_point_intersection_point = arcpy.da.TableToNumPyArray(out_table_intersection_point,('*'))
        array_point_intersection_point = array_point_intersection_point.reshape(array_point_intersection_point.shape[0], 1)
        array_point_intersection_point = np.array([list(i[0]) for i in array_point_intersection_point])

        pointid_nearest_to_first_intersection_point = array_point_intersection_point[0,[2,7,8]]
        pointid_2ndnearest_to_first_intersection_point = array_point_intersection_point[1,[2,7,8]]

        try:
            pointid_nearest_to_2nd_intersection_point = array_point_intersection_point[2,[2,7,8]]
        except:
            print()
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
    delete_old_objects_from_gdb("selected_boundary_points")
    # loop in stepsize 2: one is the first point before geofence intersection, one the point at the end of the intersection
    arcpy.CreateFeatureclass_management(arcpy.env.workspace, "selected_boundary_points", "POINT",template = geofence_point_boundary)

    #take care of the starting and endpoint
    fields = ["SHAPE@X","SHAPE@Y","SHAPE@Z","int_z","noise","grid_z"]
    arcpy.CreateFeatureclass_management("memory", "start_point", "POINT",template=flight_path,has_z = "ENABLED")
    with arcpy.da.InsertCursor(r"memory\start_point",fields) as iCur:
        iCur.insertRow([endpoints[0][0],endpoints[0][1],endpoints[0][2],endpoints[0][2],"50",endpoints[0][3]])


    arcpy.CreateFeatureclass_management("memory", "end_point", "POINT", template=flight_path,has_z = "ENABLED")
    with arcpy.da.InsertCursor(r"memory\end_point",fields) as iCur:
        iCur.insertRow([endpoints[-1][0],endpoints[-1][1],endpoints[-1][2],endpoints[-1][2],"50",endpoints[-1][3]])


    for i in range(0, len(new_coordinates - 2), 2):
        _counter += 1
        #createRectangleBetweenPoints(new_coordinates[i][1:], new_coordinates[i + 1][1:],
                                     #r"memory\rect_" + str(_counter))
        Selected_Intersection_points = arcpy.SelectLayerByAttribute_management(r"memory\sintP", "NEW_SELECTION",
                                                '"OBJECTID" IN ({0})'.format(', '.join(map(str, [i+1, i+2]))))
        # arcpy.CopyFeatures_management(Selected_Intersection_points,
        #                               r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\testintpoints"+str(_counter))
        matchcountIntersection_points = int(arcpy.GetCount_management(Selected_Intersection_points)[0])
        Selected_geofences = arcpy.management.SelectLayerByLocation(geofences, "WITHIN_A_DISTANCE_GEODESIC",
                                               Selected_Intersection_points, "20 Meters", "NEW_SELECTION", "NOT_INVERT")
        # arcpy.CopyFeatures_management(Selected_geofences,
        #                               r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\testgeofencesel" + str(
        #                                   _counter))

        matchcountpols = int(arcpy.GetCount_management(Selected_geofences)[0])
        pol_boundary_to_points(Selected_geofences,r"memory\bound_p_" + str(_counter), 10)

        #select the two corresponding intersection points
        points_to_insert = find_shortest_way_between_intersection_points(r"memory\bound_p_" + str(_counter), Selected_Intersection_points)
        if points_to_insert is not None:
            create_xy_FCFromPoints(points_to_insert[:,1], points_to_insert[:,2], r"memory\selected_boundary_points")

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
            #arcpy.CopyFeatures_management(Selected_geofence_boundary_points, r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\testpoints")
            matchcount_boundary_points = int(arcpy.GetCount_management(Selected_geofence_boundary_points)[0])

            arcpy.Append_management([Selected_geofence_boundary_points],"selected_boundary_points")

            arcpy.SelectLayerByAttribute_management(Selected_Intersection_points, "CLEAR_SELECTION")
            arcpy.SelectLayerByAttribute_management(Selected_geofences, "CLEAR_SELECTION")
            arcpy.SelectLayerByAttribute_management(Selected_geofence_boundary_points, "CLEAR_SELECTION")

    # selection_for_deletion = arcpy.management.SelectLayerByLocation("selected_boundary_points",
    #                                                                 "WITHIN", geofences, None,
    #                                                                 "NEW_SELECTION", "NOT_INVERT")
    # if int(arcpy.GetCount_management(selection_for_deletion).getOutput(0)) > 0:
    #     arcpy.DeleteFeatures_management(selection_for_deletion)
    # arcpy.SelectLayerByAttribute_management(selection_for_deletion, "CLEAR_SELECTION")

    extractValues_many_Rasters("selected_boundary_points",
                                   r" {} int_z; {} noise; {} int_z1; {} grid_z;{} grid_z".format(
                                       placeholder_interpolated_surface, noisemap,
                                       placeholder_interpolated_surface, IDW, IDW))

    arcpy.FeatureTo3DByAttribute_3d("selected_boundary_points",
                                    r"memory\selected_boundary_points_3D", "int_z")
    #arcpy.CopyFeatures_management(r"memory\selected_boundary_points_3D",
    #r"C:\Users\Moritz\Documents\ArcGIS\Projects\Testing\Testing.gdb\testgeofencesemerge")

    #
    arcpy.Merge_management([r"memory\start_point",r"memory\selected_boundary_points_3D", flight_path, r"memory\end_point"],
                           output_name)
    #delete the old points in the restricted airspace
    selection_for_deletion = arcpy.management.SelectLayerByLocation(output_name,
                                                                    "WITHIN_A_DISTANCE", geofences, "0 Meters",
                                                                    "NEW_SELECTION", "NOT_INVERT")
    if int(arcpy.GetCount_management(selection_for_deletion).getOutput(0)) > 0:
        arcpy.DeleteFeatures_management(selection_for_deletion)
    #insert endpoints  at the right positions
    combined_3D_representation = point3d_fc_to_np_array(
        output_name,
        additional_fields=["grid_z", "noise"])

    #make sure first point is starting point and last point is ending point
    # result = np.where(
    #     (combined_3D_representation[:, 0] == endpoints[0][0]) & (combined_3D_representation[:, 1] == endpoints[0][1]))
    # if len(result[0])>=1:
    #     copy_first_row = combined_3D_representation[0]
    #     combined_3D_representation[0] = combined_3D_representation[result[0][0]]
    #     combined_3D_representation[result[0][0]] = copy_first_row
    # else:
    #     np.insert(combined_3D_representation, 0, [endpoints[0][0], endpoints[0][1],endpoints[0][2], endpoints[0][3], 50], axis=0)
    # result = np.where(
    #     (combined_3D_representation[:, 0] == endpoints[1][0]) & (combined_3D_representation[:, 1] == endpoints[1][1]))
    # if len(result[0]) >= 1:
    #     copy_last_row = combined_3D_representation[-1]
    #     combined_3D_representation[-1] = combined_3D_representation[result[0][0]]
    #     combined_3D_representation[result[0][0]] = copy_last_row
    # else:
    #     combined_3D_representation = np.vstack([combined_3D_representation, [endpoints[-1][0], endpoints[-1][1],endpoints[-1][2], endpoints[-1][3], 50]])



    #index_column = np.array([i + 1 for i in range(combined_3D_representation.shape[0])]).reshape(combined_3D_representation.shape[0],1)
    #combined_3D_representation = np.hstack([index_column,combined_3D_representation])
    arcpy.Delete_management("memory")
    arcpy.Delete_management("in_memory")
    arcpy.SelectLayerByAttribute_management(selection_for_deletion, "CLEAR_SELECTION")
    return combined_3D_representation


# def repairLine(feature_Class_points, buffer_feature, new_points_line_path):
#     new_line = r'in_memory\draft_line'
#     arcpy.AddXY_management(feature_Class_points)
#     arcpy.XYToLine_management(feature_Class_points, new_line, 'POINT_X', 'POINT_Y', 'POINT_X2', 'POINT_Y2', 'GEODESIC', '#', "PROJCS['NAD_1983_StatePlane_New_York_Long_Island_FIPS_3104_Feet',GEOGCS['GCS_North_American_1983',DATUM['D_North_American_1983',SPHEROID['GRS_1980',6378137.0,298.257222101]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Lambert_Conformal_Conic'],PARAMETER['False_Easting',984250.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',-74.0],PARAMETER['Standard_Parallel_1',40.66666666666666],PARAMETER['Standard_Parallel_2',41.03333333333333],PARAMETER['Latitude_Of_Origin',40.16666666666666],UNIT['Foot_US',0.3048006096012192]];-120039300 -96540300 3048,00609601219;-100000 10000;-100000 10000;3,28083333333333E-03;0,001;0,001;IsHighPrecision")
#     f_layer = "f_line"
#     arcpy.MakeFeatureLayer_management(new_line, f_layer)
#     arcpy.SelectLayerByLocation_management(f_layer, 'CROSSED_BY_THE_OUTLINE_OF', buffer_feature, '#', 'NEW_SELECTION', 'NOT_INVERT')
#     a_index_touch = []
#     with arcpy.da.SearchCursor(f_layer,["OID"]) as cursor:
#         for row in cursor:
#             a_index_touch.append(row[0])
#
#     a_z_values = []
#     with arcpy.da.SearchCursor(feature_Class_points,["Z"]) as cursor:
#         for row in cursor:
#             a_z_values.append(row[0])
#     arcpy.MakeFeatureLayer_management(new_line, f_layer)
#     arcpy.SelectLayerByAttribute_management(f_layer, "CLEAR_SELECTION")
#     a_new_X = []
#     a_new_Y = []
#     a_new_Z = []
#     flag_repair1 = 0
#
#     with arcpy.da.SearchCursor(new_line, ["OID", 'POINT_X', 'POINT_Y']) as cursor:
#         for row in cursor:
#             if row[0] in a_index_touch:
#
#                 #print(str(flag_repair1) + "Altern point")
#                 arcpy.SelectLayerByAttribute_management(f_layer, 'NEW_SELECTION', 'OID =' + str(row[0]))
#                 out_momen_points = "in_memory\points_intersected"
#                 arcpy.Intersect_analysis([f_layer, buffer_feature], out_momen_points, 'ALL', '#', 'POINT')
#                 arcpy.AddXY_management(out_momen_points)
#                 a_new_X.append(row[1])
#                 a_new_Y.append(row[2])
#                 a_new_Z.append(a_z_values[flag_repair1])
#                 with arcpy.da.SearchCursor(out_momen_points, ["OID", 'POINT_X', 'POINT_Y']) as cursor:
#                     for row_s in cursor:
#                         a_new_X.append(row_s[1])
#                         a_new_Y.append(row_s[2])
#                         #a_new_Z.append(a_z_values[flag_repair1])
#                         if flag_repair1 + 1 < len(a_z_values):
#                             if a_z_values[flag_repair1] >= a_z_values[flag_repair1 + 1]:
#                                 a_new_Z.append(a_z_values[flag_repair1])
#                             else:
#                                 a_new_Z.append(a_z_values[flag_repair1+1])
#                         else:
#                             a_new_Z.append(a_z_values[flag_repair1])
#             else:
#                 a_new_X.append(row[1])
#                 a_new_Y.append(row[2])
#                 a_new_Z.append(a_z_values[flag_repair1])
#
#             flag_repair1 +=1
#     #new_points_calculated = r'D:\Geotech_Classes\GIS_App\Copy_Project\Moritz_Bk\Bk.gdb\draft_new_points'
#     createFCFromPoints(a_new_X, a_new_Y, a_new_Z, new_points_line_path)
#     #print('New points created and calculated')
#     return new_points_line_path


###GA Utils
def getKey(item):
    return item[1]

def accumalitive_fitness(population):
    total_fitness = 0
    for i in range (0, len(population) , 1):
        total_fitness += population[i].fitness * i
    return total_fitness



#Keeps best of the population
def elitism(population, elite):
    population_fitness = [(individual, individual.fitness) for individual in population]
    sorted_population = sorted(population_fitness, key=getKey, reverse=False)
    sorted_population.pop(0)
    add_elite = (elite, elite.fitness)
    sorted_population.append(add_elite)
    population_new = [individual[0] for individual in sorted_population]
    return population_new

#Same like elitism, but keeps n best
def parametrized_n_eliteranism(n_replacements):
    def n_eliteranism(population, elite):
        population_new = elitism(population, elite)
        for i in range (n_replacements - 1):
            population_new = elitism(population_new, elite)
        return population_new
    return n_eliteranism

##Selection algorithms
#tournament selection: defines random tournament members of the size of the tournament pressure -> The best gets selected
def nsga_parametrized_tournament_selection(pressure):
    def tournament_selection(population, minimization, random_state):
        tournament_pool_size = int(len(population)*pressure)
        tournament_pool = random_state.choice(population, size=tournament_pool_size, replace=False)
        tournament_winner = tournament_pool[0]
        for solution in tournament_pool:
            if solution.rank < tournament_winner.rank:
                tournament_winner = solution
            elif solution.rank == tournament_winner.rank:
                if solution.crowding_distance > tournament_winner.crowding_distance:
                    tournament_winner = solution
        return tournament_winner
    return tournament_selection

##Crossovers
def one_point_crossover(p1_r, p2_r, random_state):
    len_ = p1_r.shape[0]
    len_2 = p2_r.shape[0]
    point = random_state.randint(1, len_ - 1)
    if len_ == len_2:
        off1_r = np.concatenate((p1_r[0:point], p2_r[point:len_2]))
        off2_r = np.concatenate((p2_r[0:point], p1_r[point:len_]))
    else:
        distance_matrix = []
        for i in range(len_2):
            distance_matrix.append([math.sqrt(
                pow((p1_r[point][1] - p2_r[i][1]), 2)
                + pow((p1_r[point][2] - p2_r[i][2]), 2)
                + pow((p1_r[point][3] - p2_r[i][3]), 2)) / 2])
        distance_matrix = np.array(distance_matrix)
        crossover_point_off2 = np.where(distance_matrix == np.amin(distance_matrix))
        crossover_point_off2 = crossover_point_off2[0][0]
        off1_r = np.concatenate((p1_r[0:point], p2_r[crossover_point_off2:len_2]))
        off2_r = np.concatenate((p2_r[0:crossover_point_off2], p1_r[point:len_]))
    off1_r[:, 0] = np.array([i + 1 for i in range(off1_r.shape[0])])
    off2_r[:, 0] = np.array([i + 1 for i in range(off2_r.shape[0])])
    return off1_r, off2_r

def  n_point_crossover(n_crossover):
    def n_point_crossover(p1_r, p2_r, random_state):
        p1_r, p2_r = p1_r.representation[:,:5], p2_r.representation[:,:5]
        off1_r, off2_r = one_point_crossover(p1_r, p2_r, random_state)
        for i in range(n_crossover - 1):
            off1_r, off2_r = one_point_crossover(off1_r, off2_r, random_state)
        return off1_r, off2_r
    return n_point_crossover

def geometric_crossover(p1_r, p2_r, random_state):
    p1_r_repr, p2_r_repr = p1_r.representation[:,:5], p2_r.representation[:,:5]
    len_ = min(p1_r_repr.shape[0], p2_r_repr.shape[0])
    point = (random_state.randint(1,10)/10)
    for i in range(len_):
        if i > 0 and i < len_-1:
            #Crossover in x-axis
            p1_r_repr[i][1] = ((p1_r_repr[i][1]*point)+((p2_r_repr[i][1])*(1-point)))
            p2_r_repr[i][1] = (((p1_r_repr[i][1]*(1-point))+(p2_r_repr[i][1])*point))
            # Crossover in y-axis
            p1_r_repr[i][2] = ((p1_r_repr[i][2] * point) + ((p2_r_repr[i][2]) * (1 - point)))
            p2_r_repr[i][2] = (((p1_r_repr[i][2] * (1 - point)) + (p2_r_repr[i][2]) * point))
            # Crossover in z-axis
            p1_r_repr[i][3] = ((p1_r_repr[i][3] * point) + ((p2_r_repr[i][3]) * (1 - point)))
            p2_r_repr[i][3] = (((p1_r_repr[i][3] * (1 - point)) + (p2_r_repr[i][3]) * point))
            #get the new grid_z value of new position
    ##update the feature classes to changes
    p1_r, p2_r = p1_r_repr, p2_r_repr
    return p1_r, p2_r

##Mutations

def parametrized_point_mutation(percentage_disturbed_chromosomes,max_disturbance_distance, percentage_inserted_and_deleted_chromosomes, group_size):
    def ball_mutation(solution, random_state):
        print(" repr before mutation"+str(solution.representation.shape))
        number_to_delete_and_insert = int(float(percentage_inserted_and_deleted_chromosomes * solution.representation.shape[0]))
        def disturbance():
            for i in range(solution.representation.shape[0]):
                # Disturb all points except starting and destination point
                if i > 0 and i < solution.representation.shape[0]-1:
                    if random_state.uniform() < percentage_disturbed_chromosomes:
                        # disturbance in x and y
                        # calculate random distance of disturbance between -max_disturbance_distance and max_disturbance_distance in x and y
                        solution.representation[i][1] = random_state.uniform(solution.representation[i][1] - max_disturbance_distance, solution.representation[i][1] +max_disturbance_distance)
                        solution.representation[i][2] = random_state.uniform(solution.representation[i][2] - max_disturbance_distance, solution.representation[i][2] + max_disturbance_distance)
                        # disturbance in z
                        # info: solution.representation[i][3] = Z, solution.representation[i][4] = GridZ
                        # Random altitude between minimal allowed height at that point (gridz) and current z plus half distance between minimal flight height and current z.
                        # Parameter of half distance is chosen to keep a tendency for low point heights
                        z_point_mut = random_state.uniform(
                            low=solution.representation[i][4],
                            high=solution.representation[i][3] + ((solution.representation[i][3]-solution.representation[i][4])/2) )
                        #making sure the new random height is lower than the maximal flight height
                        if z_point_mut < 213:
                            solution.representation[i][3] = z_point_mut
                        else:
                            solution.representation[i][3] = 212
            return solution
        def group_disturbance(group_size):
            for i in range(solution.representation.shape[0]):
                #sine curve formular variables
                a = 0.5
                b = 2 * np.pi
                c = 2 * np.pi
                d = a

                # Disturb all points except starting and destination point
                if i > 0 and i < solution.representation.shape[0]-group_size - 1:
                    if random_state.uniform() < percentage_disturbed_chromosomes / group_size:
                        list_to_mutate = []
                        for j in range(i, i+group_size):
                            list_to_mutate.append(solution.representation[j])
                        list_to_mutate = np.array(list_to_mutate)
                        #create the random mutation distances
                        x_y_disturbance = random_state.uniform(-max_disturbance_distance, max_disturbance_distance)
                        min_z = np.amin(list_to_mutate[:, 4], axis=0)
                        z_disturbance = random_state.uniform(
                            low=min_z,
                            high=solution.representation[i+int(group_size/2)][3] + (
                                    (solution.representation[i+int(group_size/2)][3] - solution.representation[i][4]) / 2))
                        start_point = list_to_mutate[0]
                        end_point = list_to_mutate[-1]
                        position_on_vector_from_start_to_endpoint = []
                        for j in range(list_to_mutate.shape[0]):
                            dist_to_start_point = math.sqrt(
                                pow((list_to_mutate[j][1] - start_point[1]), 2)
                                + pow((list_to_mutate[j][2] - start_point[2]), 2)
                                + pow((list_to_mutate[j][3] - start_point[3]), 2))
                            dist_to_end_point = math.sqrt(
                                pow((list_to_mutate[j][1] - end_point[1]), 2)
                                + pow((list_to_mutate[j][2] - end_point[2]), 2)
                                + pow((list_to_mutate[j][3] - end_point[3]), 2))
                            x_val = dist_to_start_point / (dist_to_start_point+ dist_to_end_point)
                            position_on_vector_from_start_to_endpoint.append(x_val)
                        x = np.array(position_on_vector_from_start_to_endpoint)
                        #formula for calculation of new positions
                        y = a * np.sin(b * (x - c)) + d
                        for j in range(group_size):
                            solution.representation[i+j][1] =solution.representation[i+j][1]+  x_y_disturbance * y[j]
                            solution.representation[i+j][2] = solution.representation[i+j][2]+ x_y_disturbance * y[j]
                            if solution.representation[i+j][3]  + z_disturbance * y[j] > solution.representation[i+j][4] and solution.representation[i+j][3]  + z_disturbance * y[j] > solution.representation[i+j][4] < 213:
                                solution.representation[i+j][3] = solution.representation[i+j][3] + z_disturbance * y[j]
            return solution


        ## Insert a random point in a threed ball around the centre between points
        def insertion(nr_to_insert):
            # create array with distance to neitghbor points
            distance_to_last_point = []
            distance_to_last_point.append([0, float('-inf')])
            for i in range(solution.representation.shape[0]):
                # Disturb all points except starting and destination point
                if i > 0 and i < solution.representation.shape[0] - 1:
                    #calculate distance to previous point
                    distance_to_last_point.append([i, math.sqrt(
                        pow((solution.representation[i][1] - solution.representation[i- 1][1]), 2)
                        + pow((solution.representation[i][2] - solution.representation[i- 1][2]), 2)
                        + pow((solution.representation[i][3] - solution.representation[i- 1][3]), 2)) ])
            distance_to_last_point.append([i + 1, float('-inf')])
            sort_by_distance = np.array(distance_to_last_point)
            rank_by_distance = sort_by_distance[:, 1].argsort()
            indizes_for_inserting = []
            values_for_inserting = []
            len_ = solution.representation.shape[0]
            for i in range(len_):
                # Disturb all points except starting and destination point
                rank = np.where(rank_by_distance==i)
                rank = rank[0][0]
                if i > 0 and i < solution.representation.shape[0] - 1:
                    if rank >= (solution.representation.shape[0]-nr_to_insert):
                        indizes_for_inserting.append(i)
                        values_for_inserting.append([i,(solution.representation[i-1][1]+solution.representation[i][1])/2,(solution.representation[i-1][2]+solution.representation[i][2])/2,(solution.representation[i-1][3]+solution.representation[i][3])/2])
            solution.representation = solution.representation[:,:4]
            solution.representation = np.insert(solution.representation, indizes_for_inserting, np.array(values_for_inserting), axis = 0)
            #reindex
            solution.representation[:, 0] = np.array([i + 1 for i in range(solution.representation.shape[0])])
            return solution

        def deletion(nr_to_delete):
            #create array with distance to neitghbor points
            avg_distance_to_neighbor_points = []
            avg_distance_to_neighbor_points.append([0,float('inf')])
            for i in range(solution.representation.shape[0]):
                if i > 0 and i < solution.representation.shape[0]-1:
                    avg_distance_to_neighbor_points.append([i, math.sqrt(
                        pow((solution.representation[i - 1][1] - solution.representation[i + 1][1]), 2)
                        + pow((solution.representation[i - 1][2] - solution.representation[i + 1][2]), 2)
                        + pow((solution.representation[i - 1][3] - solution.representation[i + 1][3]), 2))/2])
            avg_distance_to_neighbor_points.append([i+1,float('inf')])
            sort_by_distance = np.array(avg_distance_to_neighbor_points)
            rank_by_distance = sort_by_distance[:, 1].argsort()
            #Delete points with the lowest distance, nr depends on the parameter of deleted_and_inserted_chromosomes
            indizes_for_delete = []
            for i in range(solution.representation.shape[0]):
                # Disturb all points except starting and destination point
                rank = np.where(rank_by_distance == i)
                rank = rank[0][0]
                if i > 0 and i < solution.representation.shape[0]-1:
                    if rank < nr_to_delete-1:
                        indizes_for_delete.append(i)
            solution.representation = np.delete(solution.representation, indizes_for_delete, axis=0)
            #reindex
            solution.representation[:,0] = np.array([i+1 for i in range(solution.representation.shape[0])])
            return solution
        if number_to_delete_and_insert > 0:
            solution = deletion(number_to_delete_and_insert)
        solution = group_disturbance(group_size)
        #solution = disturbance()
        if number_to_delete_and_insert > 0:
            solution = insertion(number_to_delete_and_insert)
        return solution.representation
    return ball_mutation

##Visualization utils
class Dplot():
    def background_plot(self, hypercube, function_):
        dim1_min = hypercube[0][0]
        dim1_max = hypercube[0][1]
        dim2_min = hypercube[1][0]
        dim2_max = hypercube[1][1]

        x0 = np.arange(dim1_min, dim1_max, 0.1)
        x1 = np.arange(dim2_min, dim2_max, 0.1)
        x0_grid, x1_grid = np.meshgrid(x0, x1)

        x = np.array([x0_grid, x1_grid])
        y_grid = function_(x)

        # plot
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.ax.set_xlim(dim1_min, dim1_max)
        self.ax.set_ylim(dim2_min, dim2_max)
        self.ax.plot_surface(x0_grid, x1_grid, y_grid, rstride=1, cstride=1, color="green", alpha=0.15) # cmap=cm.coolwarm,

    def iterative_plot(self, points, z, best=None):
        col = "k" if best is None else np.where(z == best, 'r', 'k')
        size = 75 if best is None else np.where(z == best, 150, 75)
        self.scatter = self.ax.scatter(points[0], points[1], z, s=size, alpha=0.75, c=col)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.scatter.remove()

def rastrigin(point):
    a = len(point) * 10 if len(point.shape) <= 2 else point.shape[0] * 10
    return a + np.sum(point ** 2 - 10. * np.cos(np.pi * 2. * point), axis=0)

def multi_objective_NSGA_fitness_evaluation():
    def fitness_evaluation(problem_instance, solution):
        sol_rep_w_speed_limits = calculate_speed_limits(solution.placeholder_representation[:,1:5],problem_instance.flight_constraints.maximum_angular_speed,3,problem_instance.flight_constraints.maximum_speed_air_taxi,
                               problem_instance.flight_constraints.maximum_speed_legal)
        array_distances, array_velocities, array_energy_consumptions, array_noises, total_energy_consumption, flight_time = calculate_speed_and_energy_consumption(sol_rep_w_speed_limits,problem_instance.flight_constraints)
        avg_added_noise = calculate_added_noise(solution.placeholder_representation, array_noises, solution.placeholder_representation[:,5])

        # pd.DataFrame(np.array(array_velocities)).to_csv(r"D:\Master_Shareverzeichnis\Masterthesis\Ergebnisse\velocities.csv")
        # pd.DataFrame(np.array(array_energy_consumptions)).to_csv(r"D:\Master_Shareverzeichnis\Masterthesis\Ergebnisse\energie.csv")
        # pd.DataFrame(np.array(array_noises)).to_csv(r"D:\Master_Shareverzeichnis\Masterthesis\Ergebnisse\noisearray.csv")

        return [flight_time, total_energy_consumption, avg_added_noise]
    return fitness_evaluation


def multi_objective_fitness_evaluation( weight_slope, weight_euc_dist, weight_bend_angle):
    def fitness_evaluation(solution):
        solution.representation = np.hstack([solution.representation,calculateSlopes_eucDist_bAngle(solution.PointFCName)])
        route = solution.representation
        #fitness evaluation with slope, euclidian distance and bending angle
        def calc_slope_fitness():
            minimal_possible_avg_slope =  ((route[0,3] - route[-1,3]) / pow(pow((route[0,1] - route[-1,1]), 2) + pow((route[0,2] - route[-1,2]), 2), 0.5)) * 100
            #its not possible to get to the destination without an average slope bigger than 90
            maximal_possible_abs_avg_slope = 89
            avg_slope= np.sum(abs(route[:,5]))/route.shape[0]
            slope_fitness = (float(avg_slope )-float(minimal_possible_avg_slope))/float(maximal_possible_abs_avg_slope)
            return slope_fitness

        def calc_distance_fitness():
            # "OBJECTID" 0, "SHAPE@X" 1, "POINT_X2" 2, "SHAPE@Y" 3, "POINT_Y2" 4, fieldName_NewZ 5, "POINT_Z2" 6
            minimal_possible_3d_eucl_distance = math.sqrt(pow((route[0,1] - route[-1,1]), 2) + pow((route[0,2] - route[-1,2]), 2) + pow((route[0,3] - route[-1,3]), 2))
            # it is not really possible to define a maximum distance.
            maximal_3d_eucl_distance =  5 * minimal_possible_3d_eucl_distance
            sum_distance  = np.sum(route[:,6])
            distance_fitness = (float(sum_distance)-float(minimal_possible_3d_eucl_distance))/float(maximal_3d_eucl_distance)
            return distance_fitness

        def calc_bendangle_fitness():
            #minimal_possible_bendangle = 0
            maximal_possible_bendangle = 89
            avg_bending_angle = np.sum(abs(route[:, 7])) / route.shape[0]
            ba_fitness =  float(avg_bending_angle) / float(maximal_possible_bendangle)
            return ba_fitness

        slope_fitness = calc_slope_fitness()
        bend_angle_fitness = calc_bendangle_fitness()
        euc_distance_fitness = calc_distance_fitness()
        overall_fitness =  np.average(np.array([slope_fitness, bend_angle_fitness,euc_distance_fitness]),weights = [weight_slope, weight_bend_angle, weight_euc_dist])
        return overall_fitness
    return fitness_evaluation

def sphere_function(point):
    return np.sum(np.power(point, 2.), axis=0)#len(point.shape) % 2 - 1)



def non_dominated_sort(population, minimization = True):
    #domination_count: number of solutions which dominate the solution
    #set_of_dominated_solutions: a set of solutions that the solution dominates
    # a solution A dominated another solution B if:
    # solution A is not worse than x2 in all objectives
    # solution A is better than solution B in at least one objective
    non_dominated_front = []
    pareto_fronts = []
    for solutionA in population:
        #delete for recalculation
        if hasattr(solutionA, 'rank'):
            del solutionA.rank
            del solutionA.domination_count
            del solutionA.set_of_dominated_solutions
        solutionA.domination_count = 0
        solutionA.set_of_dominated_solutions = []
        for solutionB in population:
            solutionA_dominates_solutionB = False
            solutionA_not_worse_than_SolutionB_in_all_objectives = True
            solutionA_better_than_SolutionB_in_one_objective = False
            if minimization is True:
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
            #in case of maximization
            else:
                # check if solution A dominates solution B
                for fitness_index in range(len(solutionA.fitness)):
                    if solutionB.fitness[fitness_index] > solutionA.fitness[fitness_index]:
                        solutionA_not_worse_than_SolutionB_in_all_objectives = False
                    elif solutionA.fitness[fitness_index] > solutionB.fitness[fitness_index]:
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
                        if solutionA.fitness[fitness_index] > solutionB.fitness[fitness_index]:
                            solutionB_not_worse_than_SolutionA_in_all_objectives = False
                        elif solutionB.fitness[fitness_index] > solutionA.fitness[fitness_index]:
                            solutionB_better_than_SolutionA_in_one_objective = True
                    if solutionB_not_worse_than_SolutionA_in_all_objectives is True and solutionB_better_than_SolutionA_in_one_objective is True:
                        solutionB_dominates_solutionA = True
                        solutionA.domination_count += 1

        if solutionA.domination_count == 0:
            solutionA.rank = 1
            non_dominated_front.append(solutionA)
    pareto_fronts.append([0,non_dominated_front])

    for i in range(len(population) - 1):
        current_domination_front = []
        for dominating_solution in pareto_fronts[i][1]:

            if len(dominating_solution.set_of_dominated_solutions) > 0:

                for dominated_solution in dominating_solution.set_of_dominated_solutions:
                    dominated_solution.domination_count -= 1
                    if dominated_solution.domination_count == 0:
                        current_domination_front.append(dominated_solution)
                        dominated_solution.rank = i + 2
        if len(current_domination_front) == 0:
            break
        else:
            pareto_fronts.append([i + 1, current_domination_front])
    return pareto_fronts

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
            population[i].crowding_distance[0] = int(sys.float_info.max)/8
        else:
            #calculate normalized crowding distance
            population[i].crowding_distance[0] = (crowding_distance_f1[pos_f1[0][0]+1][1] -crowding_distance_f1[pos_f1[0][0]-1][1])/(np.argmax(crowding_distance_f1[:, 1])-np.argmin(crowding_distance_f1[:, 1]))

        # calculate crowding distance for second objective
        if pos_f2[0][0] == 0 or pos_f2[0][0] == len(population)-1:
            population[i].crowding_distance[1] = int(sys.float_info.max)/8
        else:
            #calculate normalized crowding distance
            population[i].crowding_distance[1] = (crowding_distance_f2[pos_f2[0][0]+1][1] -crowding_distance_f2[pos_f2[0][0]-1][1])/(np.argmax(crowding_distance_f2[:, 1])-np.argmin(crowding_distance_f2[:, 1]))

        # calculate crowding distance for third objective
        if pos_f3[0][0] == 0 or pos_f3[0][0] == len(population)-1:
            population[i].crowding_distance[2] = int(sys.float_info.max)/8
        else:
            #calculate normalized crowding distance
            population[i].crowding_distance[2] = (crowding_distance_f3[pos_f3[0][0]+1][1] -crowding_distance_f3[pos_f3[0][0]-1][1])/(np.argmax(crowding_distance_f3[:, 1])-np.argmin(crowding_distance_f3[:, 1]))

    for solution in population:
        solution.crowding_distance = sum(solution.crowding_distance)
        if math.isnan(solution.crowding_distance):
            print("is nan")