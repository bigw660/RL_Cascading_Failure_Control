# ===========================
#   Power Flow Environment
# ===========================
import numpy as np
import matlab.engine
eng = matlab.engine.start_matlab()

# get power system data
case_file = 'case6_mariana'


def mlarray_to_pylist(x):
    # convert Matlab array into a python list
    a = np.array(x)
    if a.size == 1:
        a = [x]
    else:
        a = a.transpose()
        a = list(a[0])
    return a


class PowerSystem:
    def __init__(self):
        self.ctrl_gens_idx = [1, 2]
        # output is a list of list containing 7 sub lists.
        output = eng.ps_data(case_file, nargout=7)
        # generator output power (info)
        self.p_gen = mlarray_to_pylist(output[0])
        # demand real power (info)
        self.p_dem = mlarray_to_pylist(output[1])
        # demand reactive power (info)
        self.q_dem = mlarray_to_pylist(output[2])
        # reference voltage (single number for all buses)
        self.v_ref = mlarray_to_pylist(output[3])
        # lower bound of voltage (single number for all buses)
        self.v_min = mlarray_to_pylist(output[4])
        # upper bound of voltage (single number for all buses)
        self.v_max = mlarray_to_pylist(output[5])
        # upper bound of the line current for each line
        self.i_max = mlarray_to_pylist(output[6])
        # (indicating the index of the line we are going to take out, 0 means no line out,
        # index of line is [1, 11])
        self.lineout = 0
        # number of the generators
        self.n_gens = len(self.p_gen)
        # number of the buses
        self.n_buses = len(self.p_dem)
        # number of lines (for 6-bus system, it's 11 lines)
        self.n_lines = len(self.i_max)
        # current state
        self.i_cs = None
        # create two variable to store the indexes of safe and unsafe lines

    # def reset(self):
    #     output = eng.ps_reset(case_file, nargout=6)
    #     self.p_gen = mlarray_to_pylist(output[0])
    #     self.p_dem = mlarray_to_pylist(output[1])
    #     self.q_dem = mlarray_to_pylist(output[2])
    #     self.lineout = mlarray_to_pylist(output[3])
    #     v_cs = mlarray_to_pylist(output[4])
    #     self.i_cs = np.array(mlarray_to_pylist(output[5]))
    #     # print(self.i_cs.shape)
    #     self.p_dem = matlab.double(self.p_dem)
    #     self.q_dem = matlab.double(self.q_dem)
    #     self.lineout = matlab.double(self.lineout)
    #     self.safe_init = []
    #     self.unsafe_init = []
    #     for i in range(0, self.n_lines):
    #         if self.i_cs[i] < self.i_max[i]:
    #             self.safe_init.append(i)
    #         else:
    #             self.unsafe_init.append(i)
    #
    #     # return v_cs, i_cs (for future)
    #     print('########################################')
    #     print('Environment Reset !!!')
    #     print('########################################\n')
    #     print("******** The Initial Threshold ********")
    #     print(self.i_max, '\n')
    #     print("******** The Initial State ********")
    #     print(self.i_cs, '\n')
    #     print("******** The Indexes of Initial Unsafe Lines ********")
    #     print(self.unsafe_init, '\n')
    #     print("******** The Number of Initial Unsafe Lines ********")
    #     print(len(self.unsafe_init), '\n')
    #     return self.i_cs

    def reset_offline(self, episode):
        output = eng.ps_reset_offline(episode, nargout=6)
        self.p_gen = mlarray_to_pylist(output[0])
        self.p_dem = mlarray_to_pylist(output[1])
        self.q_dem = mlarray_to_pylist(output[2])
        self.lineout = mlarray_to_pylist(output[3])
        v_cs = mlarray_to_pylist(output[4])
        self.i_cs = np.array(mlarray_to_pylist(output[5]))
        # print(self.i_cs.shape)
        self.p_dem = matlab.double(self.p_dem)
        self.q_dem = matlab.double(self.q_dem)
        self.lineout = matlab.double(self.lineout)
        self.safe_init = []
        self.unsafe_init = []
        for i in range(0, self.n_lines):
            if self.i_cs[i] < self.i_max[i]:
                self.safe_init.append(i)
            else:
                self.unsafe_init.append(i)

        # return v_cs, i_cs (for future)
        print('########################################')
        print('Environment Reset !!!')
        print('########################################\n')
        print("******** The Initial Threshold ********")
        print(self.i_max, '\n')
        print("******** The Initial State ********")
        print(self.i_cs, '\n')
        print("******** The Initial Generator Output ********")
        print(self.p_gen, '\n')
        # print("******** The Indexes of Initial Unsafe Lines ********")
        # print(self.unsafe_init, '\n')
        print("******** The Number of Initial Unsafe Lines ********")
        print(len(self.unsafe_init), '\n')
        return self.i_cs

    # env.step() --> step the environment by one time-step. Returns
    # observation: Observations of the environment
    # reward: If your action was beneficial or not
    # terminal: Indicates if we have successfully mitigated line current and bus voltage violations

    def step(self, action):
        self.safe_before = []
        self.unsafe_before = []
        self.safe_after = []
        self.unsafe_after = []
        p_gen_range = [[0.375, 1.5], [0.45, 1.8]]
        reward = 0
        terminal = True
        early_stop = False

        # The range of generator
        for i in range(0, self.n_lines):
            if self.i_cs[i] < self.i_max[i]:
                self.safe_before.append(i)
            else:
                self.unsafe_before.append(i)

        i = 0
        for k in self.ctrl_gens_idx:
            self.p_gen[k] = action[i] + self.p_gen[k]
            i = i + 1
            if self.p_gen[k] < p_gen_range[k-1][0] or self.p_gen[k] > p_gen_range[k-1][1]:
                reward = reward - 5
                early_stop = True
        if early_stop:
            print("******** State Before Action ********")
            print(self.i_cs, '\n')
            print("******** Unsafe Lines Before Action ********")
            print(self.unsafe_before, '\n')
            print("******** Action Taken ********")
            print(action, '\n')
            print("******** Generator Output ********")
            print(self.p_gen, '\n')
            return self.i_cs, reward, terminal, early_stop

        self.p_gen = matlab.double(self.p_gen)
        output = eng.ps_solve(case_file, self.p_gen, self.p_dem, self.q_dem, self.lineout, nargout=3)
        self.p_gen = mlarray_to_pylist(output[0])
        # v_ns = mlarray_to_pylist(output[1])
        i_ns = np.asarray(mlarray_to_pylist(output[2]))

        # reward function
        counter = 0
        for i in range(0, self.n_lines):
            if i_ns[i] > self.i_max[i]:
                terminal = False
                counter = counter + 1
                self.unsafe_after.append(i)
                if self.i_cs[i] < self.i_max[i]:
                    reward = reward - (i_ns[i] - self.i_max[i]) / self.i_max[i]
            else:
                self.safe_after.append(i)
                if self.i_cs[i] > self.i_max[i]:
                    reward = reward + (self.i_max[i] - i_ns[i]) / self.i_max[i]
        if terminal:
            reward = reward + 10

        print("******** State Before Action ********")
        print(self.i_cs, '\n')
        print("******** Unsafe Lines Before Action ********")
        print(self.unsafe_before, '\n')
        print("******** Action Taken ********")
        print(action, '\n')
        print("******** Generator Output ********")
        print(self.p_gen, '\n')
        print("******** Unsafe Lines After Action & Teminal ********")
        print(self.unsafe_after, '\t', 'Terminal? ', terminal, '\n')
        print("******** State After Action ********")
        print(i_ns, '\n')
        self.i_cs = i_ns
        return self.i_cs, reward, terminal, early_stop

