import utils as uls
import numpy as np
from problems.problem import Problem
from solutions.solution import Solution
import arcpy
import pickle
import math
import time
import re
arcpy.env.workspace = r'C:\Users\Moritz\Desktop\Bk.gdb'
arcpy.env.outputZFlag = "Enabled"
arcpy.CheckOutExtension('Spatial')
arcpy.CheckOutExtension('3D')
arcpy.env.outputCoordinateSystem = arcpy.SpatialReference(32118)
import copy
arcpy.env.gpuId = 1

class ThreeDSpace(Problem):
    def __init__(self, search_space, fitness_function, IDW,noisemap, x_y_limits,z_sigma,work_space,random_state,init_network, sample_point_distance,restricted_airspace,flight_constraints,geofence_point_boundary, minimization=True):
        Problem.__init__(self, search_space, fitness_function, minimization)
        self.IDW = IDW
        self.noisemap = noisemap
        self.x_y_limits = x_y_limits
        self.z_sigma = z_sigma
        self.work_space = work_space
        self.random_state = random_state
        self.init_network = init_network
        self.sample_point_distance = sample_point_distance
        self.endpoints = self._get_endpoints()
        self.restricted_airspace = restricted_airspace
        self.flight_constraints = flight_constraints
        self.geofence_point_boundary = geofence_point_boundary

    def create_initial_route(self):
        output_new_Feature = self.work_space + "\\threedpoints"
        output_xy_vals = self.work_space + "\\output_xy_vals"
        output_z_vals = self.work_space + "\\zvals"
        point_feature_class = arcpy.management.GeneratePointsAlongLines(self.init_network, output_xy_vals,
                                              "DISTANCE", self.sample_point_distance, None, "END_POINTS")
        uls.extractValuesRaster(point_feature_class, self.IDW, output_z_vals)
        fields = ['SHAPE@X', 'SHAPE@Y', 'RASTERVALU']
        val_x = []
        val_y = []
        val_z = []
        # Loop to save the values of x,y,z from the initial feature class
        with arcpy.da.SearchCursor(output_z_vals, fields) as cursor:
            for row in cursor:
                val_x.append(row[0])
                val_y.append(row[1])
                val_z.append(row[2])
        # Once we have the new random arrays X and Y call the next funtion
        uls.createFCFromPoints(val_x, val_y, val_z, output_new_Feature)
        arcpy.AddField_management(output_new_Feature, "grid_z", "FLOAT", 9, "", "", "threedpoints")
        with arcpy.da.UpdateCursor(output_new_Feature, "grid_z") as cursor:
            i = 0
            for row in cursor:
                row[0] = val_z[i]
                i += 1
                cursor.updateRow(row)
        uls.pointsToLine(output_new_Feature, "threedline")
        return output_new_Feature,"threedline"

    def evaluate(self, solution):
        # check if all points are not in a restricted airspace and above IDW
        # validate here
        solution = self._validate(solution)
        if solution.valid:
            solution.fitness= self.fitness_function(self, solution)
        else:
            if self.minimization:
                solution.fitness = [np.iinfo(np.int32).max,np.iinfo(np.int32).max,np.iinfo(np.int32).max]
            else:
                solution.fitness = [0,0,0]
        return solution

    def _validate(self, solution):
        def preparation_for_evaluation(solution):
            solution.representation = uls.equalize_and_repair_representation_and_fc(solution, self.IDW,
                                                                                    self.endpoints, 3,
                                                                                    self.restricted_airspace, self.geofence_point_boundary)
            solution = uls.check_synchronization(solution)
            solution.placeholder_smooth_line = solution.PointFCName + "_ph_sml"
            solution.placeholder_repaired_points = solution.PointFCName + "_3d_rep"
            solution.placeholder_line_to_points = solution.PointFCName + "_ph_ltp"
            solution.placeholder_dense_points = solution.PointFCName + "_ph_dp"
            solution.placeholder_interpolated_surface = arcpy.env.workspace + "\\" + solution.PointFCName + "_ph_ints"
            solution.placeholder_dense_points_min_height = solution.PointFCName + "_ph_minh"
            solution.placeholder_dense_points_interpolated = solution.PointFCName + "_ph_dpi"
            solution.placeholder_dense_points_interpolated_3d = solution.PointFCName + "_ph_3d"

            uls.pointsToLine(solution.PointFCName, solution.LineFCName)
            uls.smoothline(solution.LineFCName, solution.placeholder_smooth_line, 100, self.restricted_airspace)
            uls.lineToPoints(solution.placeholder_smooth_line, solution.placeholder_dense_points, 10)
            # create field

            uls.interpolate_points_to_spline_surface(solution.PointFCName, solution.placeholder_interpolated_surface,
                                                     20)
            # get interpolated height at current cell of IDW (corresponding z in solution.representation)
            uls.extractValues_many_Rasters(solution.placeholder_dense_points,
                                           r" {} grid_z1; {} grid_z2; {} noise; {} int_z;{} grid_z".format(
                                               solution.placeholder_interpolated_surface, self.noisemap,
                                               solution.placeholder_interpolated_surface, self.IDW, self.IDW))
            arcpy.FeatureTo3DByAttribute_3d(solution.placeholder_dense_points,
                                            solution.placeholder_dense_points_interpolated_3d, "int_z")

            solution.placeholder_representation = uls.point3d_fc_to_np_array(
                solution.placeholder_dense_points_interpolated_3d,
                additional_fields=["grid_z", "noise"])

            #index_col = np.array([i + 1 for i in range(solution.placeholder_representation.shape[0])]).reshape(
                #solution.placeholder_representation.shape[0], 1)
            #solution.placeholder_representation = np.hstack([index_col, solution.placeholder_representation])

            #solution.placeholder_representation = uls.repairPointsInRestrictedAirspace(solution.placeholder_dense_points_interpolated_3d, solution.placeholder_representation,
            #                                                           self.restricted_airspace, self.geofence_point_boundary)
            #uls.pointsToLine(solution.placeholder_dense_points_interpolated_3d,solution.placeholder_repaired_line)
            solution.placeholder_representation = uls.line_repair(self.restricted_airspace,solution.placeholder_smooth_line ,
            solution.placeholder_dense_points_interpolated_3d,
            self.geofence_point_boundary,
            placeholder_interpolated_surface= solution.placeholder_interpolated_surface,
            noisemap= self.noisemap,
            IDW=self.IDW,
            endpoints = self.endpoints,
            output_name=solution.placeholder_repaired_points)

            # uls.repairLinesInRestrictedAirspace(solution.placeholder_smooth_line,
            #                                 solution.placeholder_dense_points_interpolated_3d,solution.placeholder_representation,
            #                                 self.restricted_airspace)
            # sol_test.representation = uls.reorder_points(sol_test.PointFCName, sol_test.representation, self.endpoints)
            #solution.placeholder_representation = uls.check_endpoints(solution.placeholder_representation, self.endpoints)
            solution.placeholder_representation, indices = uls.remove_duplicate_points(solution.placeholder_representation[:, 1:3],
                                                              solution.placeholder_representation)
            uls.remove_points_from_pointfc(solution.placeholder_repaired_points, indices)
            solution.placeholder_representation, valid_reorder = uls.reorder_points(solution.placeholder_repaired_points, solution.placeholder_representation,
                                                     self.endpoints, searchradius=100)
            if valid_reorder is False:
                valid_reorder = 1
            else:
                valid_reorder = 0
            return solution, valid_reorder


        # valid = False
        # total_constraint_violation = 0
        #
        # def validate_line_surface_intersection(line):
        #     cv_line_intersetcs_surface = 0
        #     # "l_int_w_IDW" output of line segments that intersect with IDW
        #     # "p_int_w_IDW" output of point segments that intersect with IDW
        #     arcpy.gp.Intersect3DLineWithSurface_3d (line, self.IDW, "l_int_w_IDW", "p_int_w_IDW")
        #     amnt_points_intersected = int(arcpy.GetCount_management("p_int_w_IDW").getOutput(0))
        #     if amnt_points_intersected == 0:
        #         cv_line_intersetcs_surface = 0
        #         valid = True
        #     else:
        #         cv_line_intersetcs_surface = 1
        #         #uls.repair_line(line, self.IDW)
        #     return cv_line_intersetcs_surface

        def validate_point_height(routepoints):
            i = 0
            constraint_violation_zpoints = 0
            constraint_violation=[]
            # Get the points that are higher than 213 m or lie lower than minimal flight height
            i = 0
            height_violations = 0
            while i < len(routepoints) and constraint_violation_zpoints == 0:
                if routepoints[i][3] > 213 or routepoints[i][3] < routepoints[i][4]:
                    height_violations = height_violations + 1
                i = i+1
            if height_violations == 0:
                constraint_violation_zpoints = 0
            else:
                    constraint_violation_zpoints = 1
            return routepoints,constraint_violation_zpoints

        solution, violation_reorder = preparation_for_evaluation(solution)
        routepoints,violation_point_height =validate_point_height(solution.representation)
        total_constraint_violation = violation_point_height + violation_reorder
        if total_constraint_violation == 0:
            valid = True
        else:
            valid = False
        solution.representation = routepoints
        solution.cv = total_constraint_violation
        solution.valid = valid
        uls.delete_old_objects_from_gdb("_ph")
        return solution


    def sample_search_space(self, random_state):
        #get_timestamp
        ts = str(time.time())
        ts =re.sub(r'[.]','',ts)[:10]
        #create memories for points and lines in workspace
        points,line = self.create_initial_route()
        solutionpoints = arcpy.CopyFeatures_management(points, "threedpoints"+"_"+str(ts))
        arcpy.CopyFeatures_management(line, "threedline" + "_" + str(ts))
        solution = Solution(uls.random_pointmove_in_boundary(solutionpoints, self.IDW, random_state, self.x_y_limits, self.z_sigma))
        solution.PointFCName = "threedpoints" + "_" + str(ts)
        uls.pointsToLine(solution.PointFCName,"threedline" + "_" + str(ts))
        solution.LineFCName = "threedline" + "_" + str(ts)
        return solution

    def _get_endpoints(self):
        endpoints = []
        cursor = arcpy.da.SearchCursor(self.init_network, ["OBJECTID", "SHAPE@"])
        for row in cursor:
            zvals = uls.captureValueZ(self.IDW, [row[1].firstPoint.X,row[1].lastPoint.X], [row[1].firstPoint.Y,row[1].lastPoint.Y])
            endpoints = [[row[1].firstPoint.X, row[1].firstPoint.Y, zvals[0],zvals[0]], [row[1].lastPoint.X, row[1].lastPoint.Y, zvals[-1],zvals[-1]]]
        del cursor
        return endpoints

class flight_Constraints():
    def __init__(self, type_aircraft, weight_aircraft, wing_area, CD_from_drag_polar,maximum_speed_legal, maximum_speed_air_taxi, acceleration_speed, acceleration_energy,
                 deceleration_speed, deceleration_energy, minimal_cruise_energy, take_off_and_landing_energy, hover_energy,noise_pressure_acceleration, noise_pressure_deceleration,noise_at_cruise, noise_at_hover, maximum_angular_speed = 1,
                 air_density=1.225, speed_of_sound = 343, gravity = 9.81):
        self.type_aircraft = type_aircraft
        self.weight_aircraft = weight_aircraft
        self.wing_area = wing_area
        self.CD_from_drag_polar = CD_from_drag_polar
        self.maximum_speed_legal = maximum_speed_legal
        self.maximum_speed_air_taxi = maximum_speed_air_taxi
        self.acceleration_speed = acceleration_speed
        self.acceleration_energy = acceleration_energy
        self.deceleration_speed = deceleration_speed
        self.deceleration_energy = deceleration_energy
        self.minimal_cruise_energy = minimal_cruise_energy
        self.take_off_and_landing_energy = take_off_and_landing_energy
        self.hover_energy = hover_energy
        self.noise_pressure_acceleration = noise_pressure_acceleration
        self.noise_pressure_deceleration=noise_pressure_deceleration
        self.noise_at_cruise= noise_at_cruise
        self.noise_at_hover = noise_at_hover
        self.maximum_angular_speed = maximum_angular_speed
        self.air_density = air_density
        self.speed_of_sound = speed_of_sound
        self.gravity = gravity

    def calculate_required_energy_at_level_speed(self, angle_of_climb, velocity):
        #calculates the required energy for a given aircraft with a given speed and a given angle of climb (Flight dynamics for steady climb)
        # Step 1: Lift required Lc = W * cos(a), W=Aircraft weight * Gravity, to convert kg to Newton, a = ascending/descending angle in degrees
        W = self.weight_aircraft* self.gravity
        Lc = W * math.cos(angle_of_climb)
        # Step 2: Calculate Lift coefficient Clc in climb: Clc = Lc / (.5 * p * V² * S), p = air density, V = velocity, S = Wing area
        Clc = Lc / (0.5 * self.air_density * math.pow(velocity,2) * self.wing_area)
        #Step 3: calculate drag coefficient Cdc with drag polar and calculated Clc
        Cdc = self.CD_from_drag_polar(Clc)
        # Step 4: calculate drag in climb. Dc = 1/2 * p * V² * S * Cdc
        Dc = 0.5 * self.air_density * math.pow(velocity,2) * self.wing_area * Cdc
        #Step 5: calculate thrust required in climb (Trc): Trc = W sin γ + Dc
        Trc = W * math.sin(angle_of_climb) + Dc
        #Step 6: calculate power required in climb (Prc): Prc = (Trc*V)/1000 ,in kW
        Prc = (Trc * velocity) / 1000
        return Prc