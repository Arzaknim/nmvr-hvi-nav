import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

class Fuzzy_inference:
    """
    5-input Mamdani fuzzy logic.

    Input (all distances):
        FL : front-left
        L  : left
        C  : center (front)
        R  : right
        FR : front-right

    Output:
        steering correction in [-1.0, 1.0]
        negative -> turn RIGHT
        positive -> turn LEFT
        zero     -> go straight
    """

    def __init__(self, max_dist=1):
        self.max_dist = float(max_dist)

        # Universe of distances and steering
        self.x_dist = np.arange(0.0, self.max_dist + 0.001, 0.01)
        self.x_steer = np.arange(-1.0, 1.0 + 0.001, 0.01)

        self.dist_type = "trimf"
        self.steer_type = "trimf"

        self.dist_near_list = [0.0, 0.0, 0.3]
        self.dist_med_list = [0.15, 0.45, 0.75]
        self.dist_far_list = [0.6, self.max_dist, self.max_dist]
        #
        self.turn_hard_right_lst = [-1.00, -1.00, -0.50]
        self.turn_soft_right_lst = [-0.75, -0.40, -0.05]
        self.go_straight_lst = [-0.20, 0.00, 0.20]
        self.turn_soft_left_lst = [0.05, 0.40, 0.75]
        self.turn_hard_left_lst = [0.50, 1.00, 1.00]

        # self.dist_near_list = [0, 0.33]
        # self.dist_med_list = [self.max_dist/2, 0.33]
        # self.dist_far_list = [self.max_dist, 0.33]

        # self.turn_hard_right_lst = [-1, 0.20]
        # self.turn_soft_right_lst = [-0.5, 0.20]
        # self.go_straight_lst = [0.0, 0.20]
        # self.turn_soft_left_lst = [0.5, 0.20]
        # self.turn_hard_left_lst = [1, 0.20]

        # Distance membership functions
        self.dist_near = self.fuzz_Mem_Func(self.x_dist, self.dist_type, self.dist_near_list)
        self.dist_med = self.fuzz_Mem_Func(self.x_dist, self.dist_type, self.dist_med_list)
        self.dist_far = self.fuzz_Mem_Func(self.x_dist, self.dist_type, self.dist_far_list)

        # Steering membership functions
        self.turn_hard_right = self.fuzz_Mem_Func(self.x_steer, self.steer_type, self.turn_hard_right_lst)
        self.turn_soft_right = self.fuzz_Mem_Func(self.x_steer, self.steer_type, self.turn_soft_right_lst)
        self.go_straight = self.fuzz_Mem_Func(self.x_steer, self.steer_type, self.go_straight_lst)
        self.turn_soft_left = self.fuzz_Mem_Func(self.x_steer, self.steer_type, self.turn_soft_left_lst)
        self.turn_hard_left = self.fuzz_Mem_Func(self.x_steer, self.steer_type, self.turn_hard_left_lst)


    # membership function utility
    def fuzz_Mem_Func(self, var, typeOfMf, lst):
        if typeOfMf == 'trimf':
            return fuzz.trimf(var, lst)
        elif typeOfMf == 'gaussmf':
            mean, sigma = lst
            return fuzz.gaussmf(var, mean, sigma)
        elif typeOfMf == 'gauss2mf':
            mean1, sigma1, mean2, sigma2 = lst
            return fuzz.gauss2mf(var, mean1, sigma1, mean2, sigma2)
        elif typeOfMf == 'trapmf':
            return fuzz.trapmf(var, lst)
        elif typeOfMf == 'gbellmf':
            a, b, c = lst
            return fuzz.gbellmf(var, a, b, c)

    def fuzz_plot_mf(self, x_var, var_, var_types, varName):
        print(f'The following plot shows the {varName}')
        fig, ax = plt.subplots(figsize=(8, 3))
        for i in range(len(var_)):
            ax.plot(x_var, var_[i], linewidth=1.5, label=var_types[i])
        ax.set_title(varName)
        ax.legend()
        plt.show()

    def fuzz_plot_outputMf(self, x_var, rule, output_used):
        zerolike = np.zeros_like(x_var)
        fig, ax0 = plt.subplots(figsize=(8, 3))
        for i in range(len(rule)):
            ax0.fill_between(x_var, zerolike, rule[i], facecolor='orange', alpha=0.7)
            ax0.plot(x_var, output_used[i], linewidth=0.5, linestyle='--')
        ax0.set_title('Output membership activity')

        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()
        plt.show()

    def fuzz_Interplot_mem(self, x_var, lst, singleton_value):
        memvalue = []
        for i in range(len(lst)):
            memvalue.append(fuzz.interp_membership(x_var, lst[i], singleton_value))
        return memvalue

    def fuzz_aggregation(self, rule):
        l = len(rule) - 1
        npfmax = np.fmax(rule[l - 1], rule[l])
        for i in range(len(rule) - 2):
            l = (len(rule) - 1) - (i + 1)
            npfmax = np.fmax(rule[l - 1], npfmax)
        return npfmax

    def fuzz_defuzz(self, x_var, R_combined):
        output = fuzz.defuzz(x_var, R_combined, 'centroid')
        output_activation = fuzz.interp_membership(x_var, R_combined, output)
        lst = [output, output_activation]
        return lst

    def fuzz_output(self, x_var, var, output, output_activation, R_combined):
        fig, ax0 = plt.subplots(figsize=(8, 3))
        zerolike = np.zeros_like(x_var)
        for i in range(len(var)):
            ax0.plot(x_var, var[i], linewidth=0.5, linestyle='--')
        ax0.fill_between(x_var, zerolike, R_combined, facecolor='Orange', alpha=0.7)
        ax0.plot([output, output], [0, output_activation], 'k', linewidth=1.5, alpha=0.9)
        ax0.set_title('Aggregated membership and result (line)')

        for ax in (ax0,):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()
        plt.show()

    def plot_distance_mfs(self):
        """Plot the generic distance MFs (near/med/far) used for all sectors."""
        self.fuzz_plot_mf(
            self.x_dist,
            [self.dist_near, self.dist_med, self.dist_far],
            ["near", "medium", "far"],
            "Distance"
        )

    def plot_output_mfs(self):
        """Plot the steering correction output MFs."""
        self.fuzz_plot_mf(
            self.x_steer,
            [
                self.turn_hard_right,
                self.turn_soft_right,
                self.go_straight,
                self.turn_soft_left,
                self.turn_hard_left
            ],
            ["hard_right", "soft_right", "straight", "soft_left", "hard_left"],
            "Steering"
        )

    # 5-input Mamdani evaluation
    def eval(self, FL, L, C, R, FR, debug=False):
        """
        Evaluate fuzzy safety for one state.
        Inputs:
            FL, L, C, R, FR : float distances
        Returns:
            correction : float in [-1.0, 1.0]
        """

        # Clip distances into the universe range
        FL = float(np.clip(FL, 0.0, self.max_dist))
        L  = float(np.clip(L,  0.0, self.max_dist))
        C  = float(np.clip(C,  0.0, self.max_dist))
        R  = float(np.clip(R,  0.0, self.max_dist))
        FR = float(np.clip(FR, 0.0, self.max_dist))

        # Fuzzification
        def mu_dist(d):
            mu_n = fuzz.interp_membership(self.x_dist, self.dist_near, d)
            mu_m = fuzz.interp_membership(self.x_dist, self.dist_med,  d)
            mu_f = fuzz.interp_membership(self.x_dist, self.dist_far,  d)
            return mu_n, mu_m, mu_f

        FL_near, FL_med, FL_far = mu_dist(FL)
        L_near,  L_med,  L_far  = mu_dist(L)
        C_near,  C_med,  C_far  = mu_dist(C)
        R_near,  R_med,  R_far  = mu_dist(R)
        FR_near, FR_med, FR_far = mu_dist(FR)

        # fuzzy conditions
        front_near = max(FL_near, C_near, FR_near)
        front_med = C_med
        front_far = C_far
        left_blocked  = max(L_near,  FL_near)
        right_blocked = max(R_near,  FR_near)
        left_dist  = min(FL, L)
        right_dist = min(FR, R)

        # Rules (Mamdani)

        aggregated = np.zeros_like(self.x_steer)

        # Safety threshold: if distance <= 0.4, treat that side as not safe
        safety_threshold = 0.4
        left_safe = left_dist > safety_threshold
        right_safe = right_dist > safety_threshold

        if front_near > 0.0:
            if left_safe and not right_safe:
                rule1 = np.fmin(front_near, self.turn_hard_left)

            elif right_safe and not left_safe:
                rule1 = np.fmin(front_near, self.turn_hard_right)

            elif left_safe and right_safe:
                if left_dist > right_dist:
                    rule1 = np.fmin(front_near, self.turn_hard_left)
                else:
                    rule1 = np.fmin(front_near, self.turn_hard_right)

            else:
                rule1 = np.fmin(front_near, self.turn_hard_right)

            aggregated = np.fmax(aggregated, rule1)

        rule2 = np.fmin(left_blocked, self.turn_soft_right)
        aggregated = np.fmax(aggregated, rule2)

        rule3 = np.fmin(right_blocked, self.turn_soft_left)
        aggregated = np.fmax(aggregated, rule3)

        if front_med > 0.0:
            if left_dist > right_dist:
                rule4 = np.fmin(front_med, self.turn_soft_left)
            else:
                rule4 = np.fmin(front_med, self.turn_soft_right)
            aggregated = np.fmax(aggregated, rule4)

        rule5 = np.fmin(front_far, self.go_straight)
        aggregated = np.fmax(aggregated, rule5)

        # Defuzzification
        if np.sum(aggregated) == 0.0:
            correction = 0.0
            output_activation = 0.0
        else:
            correction = fuzz.defuzz(self.x_steer, aggregated, 'centroid')
            output_activation = fuzz.interp_membership(self.x_steer, aggregated, correction)

        if debug:
            self.fuzz_output(
                self.x_steer,
                [
                    self.turn_hard_right,
                    self.turn_soft_right,
                    self.go_straight,
                    self.turn_soft_left,
                    self.turn_hard_left
                ],
                correction,
                output_activation,
                aggregated
            )

        return float(correction)

    def model(self):
        self.plot_distance_mfs()
        self.plot_output_mfs()






# import numpy as np
# import skfuzzy as fuzz
# import matplotlib.pyplot as plt
#
# class Fuzzy_inference:
#     """
#     5-input Mamdani fuzzy logic.
#
#     Input (all distances):
#         FL : front-left
#         L  : left
#         C  : center (front)
#         R  : right
#         FR : front-right
#
#     Output:
#         steering correction in [-1.0, 1.0]
#         negative -> turn RIGHT
#         positive -> turn LEFT
#         zero     -> go straight
#     """
#
#     def __init__(self, max_dist=1):
#         self.max_dist = float(max_dist)
#
#         # Universe of distances and steering
#         self.x_dist = np.arange(0.0, self.max_dist + 0.001, 0.01)
#         self.x_steer = np.arange(-1.0, 1.0 + 0.001, 0.01)
#
#         # Triangular membership functions
#         self.dist_type = "trimf"
#         self.steer_type = "trimf"
#
#         # Distance MF definitions
#         self.dist_near_list = [0.0, 0.0, 0.20]
#         self.dist_med_list = [0.15, 0.40, 0.70]
#         self.dist_far_list = [0.55, self.max_dist, self.max_dist]
#
#         # Steering MF definitions (triangular)
#         self.turn_hard_right_lst = [-1.00, -1.00, -0.50]
#         self.turn_soft_right_lst = [-0.75, -0.40, -0.05]
#         self.go_straight_lst = [-0.20, 0.00, 0.20]
#         self.turn_soft_left_lst = [0.05, 0.40, 0.75]
#         self.turn_hard_left_lst = [0.50, 1.00, 1.00]
#
#         # Distance membership functions
#         self.dist_near = self.fuzz_Mem_Func(self.x_dist, self.dist_type, self.dist_near_list)
#         self.dist_med = self.fuzz_Mem_Func(self.x_dist, self.dist_type, self.dist_med_list)
#         self.dist_far = self.fuzz_Mem_Func(self.x_dist, self.dist_type, self.dist_far_list)
#
#         # Steering membership functions
#         self.turn_hard_right = self.fuzz_Mem_Func(self.x_steer, self.steer_type, self.turn_hard_right_lst)
#         self.turn_soft_right = self.fuzz_Mem_Func(self.x_steer, self.steer_type, self.turn_soft_right_lst)
#         self.go_straight = self.fuzz_Mem_Func(self.x_steer, self.steer_type, self.go_straight_lst)
#         self.turn_soft_left = self.fuzz_Mem_Func(self.x_steer, self.steer_type, self.turn_soft_left_lst)
#         self.turn_hard_left = self.fuzz_Mem_Func(self.x_steer, self.steer_type, self.turn_hard_left_lst)
#
#
#     # membership function utility
#     def fuzz_Mem_Func(self, var, typeOfMf, lst):
#         if typeOfMf == 'trimf':
#             return fuzz.trimf(var, lst)
#         elif typeOfMf == 'gaussmf':
#             mean, sigma = lst
#             return fuzz.gaussmf(var, mean, sigma)
#         elif typeOfMf == 'gauss2mf':
#             mean1, sigma1, mean2, sigma2 = lst
#             return fuzz.gauss2mf(var, mean1, sigma1, mean2, sigma2)
#         elif typeOfMf == 'trapmf':
#             return fuzz.trapmf(var, lst)
#         elif typeOfMf == 'gbellmf':
#             a, b, c = lst
#             return fuzz.gbellmf(var, a, b, c)
#
#     # plotting utilities ------------------------------------------------
#
#     def fuzz_plot_mf(self, x_var, var_, var_types, varName):
#         print(f'The following plot shows the {varName}')
#         fig, ax = plt.subplots(figsize=(8, 3))
#         for i in range(len(var_)):
#             ax.plot(x_var, var_[i], linewidth=1.5, label=var_types[i])
#         ax.set_title(varName)
#         ax.legend()
#         plt.show()
#
#     def fuzz_output(self, x_var, var, output, output_activation, R_combined):
#         fig, ax0 = plt.subplots(figsize=(8, 3))
#         zerolike = np.zeros_like(x_var)
#
#         # Show all MFs
#         for i in range(len(var)):
#             ax0.plot(x_var, var[i], linewidth=0.5, linestyle='--')
#
#         # Aggregated fuzzy output
#         ax0.fill_between(x_var, zerolike, R_combined, facecolor='Orange', alpha=0.7)
#
#         # Crisp centroid
#         ax0.plot([output, output], [0, output_activation], 'k', linewidth=1.5, alpha=0.9)
#         ax0.set_title('Aggregated membership and centroid')
#
#         for ax in (ax0,):
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#
#         plt.tight_layout()
#         plt.show()
#
#     def plot_distance_mfs(self):
#         self.fuzz_plot_mf(
#             self.x_dist,
#             [self.dist_near, self.dist_med, self.dist_far],
#             ["near", "medium", "far"],
#             "Distance"
#         )
#
#     def plot_output_mfs(self):
#         self.fuzz_plot_mf(
#             self.x_steer,
#             [
#                 self.turn_hard_right,
#                 self.turn_soft_right,
#                 self.go_straight,
#                 self.turn_soft_left,
#                 self.turn_hard_left
#             ],
#             ["hard_right", "soft_right", "straight", "soft_left", "hard_left"],
#             "Steering"
#         )
#
#     # Mamdani evaluation ------------------------------------------------
#
#     # Mamdani evaluation ------------------------------------------------
#
#     def eval(self, FL, L, C, R, FR, debug=False):
#         """
#         Evaluate fuzzy safety for one state.
#         Returns:
#             (correction, log)
#         """
#         log = {}
#
#         # Clip distances to range
#         FL = float(np.clip(FL, 0.0, self.max_dist))
#         L  = float(np.clip(L,  0.0, self.max_dist))
#         C  = float(np.clip(C,  0.0, self.max_dist))
#         R  = float(np.clip(R,  0.0, self.max_dist))
#         FR = float(np.clip(FR, 0.0, self.max_dist))
#
#         # Fuzzification --------------------------------------------------
#         def mu_dist(d):
#             mu_n = fuzz.interp_membership(self.x_dist, self.dist_near, d)
#             mu_m = fuzz.interp_membership(self.x_dist, self.dist_med,  d)
#             mu_f = fuzz.interp_membership(self.x_dist, self.dist_far,  d)
#             return mu_n, mu_m, mu_f
#
#         FL_near, FL_med, FL_far = mu_dist(FL)
#         L_near,  L_med,  L_far  = mu_dist(L)
#         C_near,  C_med,  C_far  = mu_dist(C)
#         R_near,  R_med,  R_far  = mu_dist(R)
#         FR_near, FR_med, FR_far = mu_dist(FR)
#
#         log["fuzzified"] = {
#             "FL": (FL_near, FL_med, FL_far),
#             "L":  (L_near,  L_med,  L_far),
#             "C":  (C_near,  C_med,  C_far),
#             "R":  (R_near,  R_med,  R_far),
#             "FR": (FR_near, FR_med, FR_far)
#         }
#
#         # Fuzzy conditions ----------------------------------------------
#         front_near = max(FL_near, C_near, FR_near)
#         front_med  = C_med
#         front_far  = C_far
#
#         # "Blocked" = high NEAR membership on that side
#         left_near_high  = max(FL_near, L_near)
#         right_near_high = max(FR_near, R_near)
#
#         left_blocked  = left_near_high
#         right_blocked = right_near_high
#
#         # "Free" = high FAR membership on that side
#         left_far_high  = max(FL_far, L_far)
#         right_far_high = max(FR_far, R_far)
#
#         log["conditions"] = {
#             "front_near": front_near,
#             "front_med": front_med,
#             "front_far": front_far,
#             "left_blocked": left_blocked,
#             "right_blocked": right_blocked,
#             "left_near_high": left_near_high,
#             "right_near_high": right_near_high,
#             "left_far_high": left_far_high,
#             "right_far_high": right_far_high,
#         }
#
#         # Mamdani rules -------------------------------------------------
#         aggregated = np.zeros_like(self.x_steer)
#         rule_strengths = {}
#
#         # RULE 1: Front near → hard turn
#         if front_near > 0.0:
#             # Safer side = lower NEAR membership
#             if left_near_high < right_near_high:
#                 # left is safer -> hard LEFT (positive)
#                 rule1 = np.fmin(front_near, self.turn_hard_left)
#             elif right_near_high < left_near_high:
#                 # right is safer -> hard RIGHT (negative)
#                 rule1 = np.fmin(front_near, self.turn_hard_right)
#             else:
#                 # tie: use FAR as tiebreaker
#                 if left_far_high >= right_far_high:
#                     rule1 = np.fmin(front_near, self.turn_hard_left)
#                 else:
#                     rule1 = np.fmin(front_near, self.turn_hard_right)
#
#             aggregated = np.fmax(aggregated, rule1)
#             rule_strengths["rule1_front_near"] = np.max(rule1)
#         else:
#             rule_strengths["rule1_front_near"] = 0.0
#
#         # RULE 2: Left blocked → soft right
#         rule2 = np.fmin(left_blocked, self.turn_soft_right)
#         aggregated = np.fmax(aggregated, rule2)
#         rule_strengths["rule2_left_blocked"] = np.max(rule2)
#
#         # RULE 3: Right blocked → soft left
#         rule3 = np.fmin(right_blocked, self.turn_soft_left)
#         aggregated = np.fmax(aggregated, rule3)
#         rule_strengths["rule3_right_blocked"] = np.max(rule3)
#
#         # RULE 4: Front medium → soft turn towards freer side
#         if front_med > 0.0:
#             if left_far_high >= right_far_high:
#                 rule4 = np.fmin(front_med, self.turn_soft_left)
#             else:
#                 rule4 = np.fmin(front_med, self.turn_soft_right)
#             aggregated = np.fmax(aggregated, rule4)
#             rule_strengths["rule4_front_med"] = np.max(rule4)
#         else:
#             rule_strengths["rule4_front_med"] = 0.0
#
#         # RULE 5: Front far → go straight
#         rule5 = np.fmin(front_far, self.go_straight)
#         aggregated = np.fmax(aggregated, rule5)
#         rule_strengths["rule5_front_far"] = np.max(rule5)
#
#         log["rules"] = rule_strengths
#
#         # Defuzzification -----------------------------------------------
#         if np.sum(aggregated) == 0.0:
#             correction = 0.0
#         else:
#             correction = fuzz.defuzz(self.x_steer, aggregated, 'centroid')
#
#         log["correction"] = correction
#
#         # Debug plot
#         if debug:
#             self.fuzz_output(
#                 self.x_steer,
#                 [
#                     self.turn_hard_right,
#                     self.turn_soft_right,
#                     self.go_straight,
#                     self.turn_soft_left,
#                     self.turn_hard_left
#                 ],
#                 correction,
#                 fuzz.interp_membership(self.x_steer, aggregated, correction),
#                 aggregated
#             )
#
#         return correction, log
#
#
#     def model(self):
#         self.plot_distance_mfs()
#         self.plot_output_mfs()
