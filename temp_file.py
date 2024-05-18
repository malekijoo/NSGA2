import numpy as np
from data import *


class ThisProblem:

    def __init__(self, pop):
        self._X = pop.X
        self.len_X = len(self._X)
        self.objectives  = {}
        self.constraints = {}
        self.c_max = 3
        self.p_max = 4
        self.r_max = 5

    def f1_objective(self):
        one, two, three, four, five, six, seven, eight = [np.empty(shape=self.len_X, dtype=object) for _ in range(8)]
        z1 = np.empty(shape=self.len_X, dtype=object)

        ''' Obj 1: Cost Min '''
        for k in range(self.len_X):
            one[k]     = np.sum(CPp.values * self._X[k].var['PBp'])
            two[k]     = np.sum(CFr.values * self._X[k].var['RAi'])
            three[k]   = np.sum(CEc.values * self._X[k].var['CYc'])
            four[k]    = np.sum(TCFir.T.values * self._X[k].var['XAir'] * (1+dr))
            five[k]    = np.sum(Tcs_rp.T.values * self._X[k].var['XZrp'])
            six[k]     = np.sum(Tct_rd.T.values * self._X[k].var['XWrd'])
            seven[k]   = np.sum(Tcc_cr.T.values * self._X[k].var['XTci'])
            eight[k]   = np.sum(Tcd_ri.T.values * (self._X[k].var['XAir'] * dr))

        z1 = one + two  + three + four  + five + six + seven + eight

        return z1

    def f2_objective(self):
        one, two, three, four, five = [np.empty(shape=self.len_X, dtype=object) for _ in range(5)]
        z2 = np.empty(shape=self.len_X, dtype=object)
        ''' Obj 2: CO2 Min '''
        for k in range(self.len_X):
            one[k]   = np.sum(COt.values * np.sum(DCIci.values * self._X[k].var['NTCIci']))
            two[k]   = np.sum(COt.values * np.sum(DIRir_I2R.values * self._X[k].var['NTIRtir']))
            three[k] = np.sum(COt.values * np.sum(DRPrp.values * self._X[k].var['NTRPrp']))
            four[k]  = np.sum(COt.values * np.sum(DRDrd.values * self._X[k].var['NTRDrd']))
            five[k]  = np.sum(COt.values * np.sum(DIRir_R2I.values * self._X[k].var['NTRItri']))

        z2 = one + two + three + four + five

        return z2

    def f3_objective(self):
        one, two, three, four, five, six = [np.empty(shape=self.len_X, dtype=object) for _ in range(6)]
        z3 = np.empty(shape=self.len_X, dtype=object)

        ''' Obj 3: Social factor Max'''
        for k in range(self.len_X):
            one[k]   = np.sum(FJRr.values * self._X[k].var['RAi'])
            two[k]   = np.sum(FJPp.values * self._X[k].var['PBp'])
            three[k] = np.sum(FJCc.values * self._X[k].var['CYc'])
            four[k]  = np.sum(self._X[k].var['VJRr'] * self._X[k].var['RAi'])
            five[k]  = np.sum(self._X[k].var['VJPp'] * self._X[k].var['PBp'])
            six[k]   = np.sum(self._X[k].var['VJCc'] * self._X[k].var['CYc'])

        z3 = -(one + two + three + four + five + six)

        return z3

    def _constraints(self):

        one, two, three, four, five = [np.empty(shape=self.len_X, dtype=object) for _ in range(5)]
        six, seven, eight, nine, ten = [np.empty(shape=self.len_X, dtype=object) for _ in range(5)]
        eleven, twelve, thirteen, fourteen  = [np.empty(shape=self.len_X, dtype=object) for _ in range(4)]
        fifteen, sixteen, seventeen, eighteen = [np.empty(shape=self.len_X, dtype=object) for _ in range(4)]
        equality, inequality = [np.empty(shape=self.len_X, dtype=object) for _ in range(2)]

        for k in range(self.len_X):

            one[k]       = np.sum(self._X[k].var['XAir']) - np.sum(self._X[k].var['XTci'])
            two[k]       = np.sum(self._X[k].var['XZrp']) - ((1 - dw) * np.sum(self._X[k].var['XAir']))
            three[k]     = np.sum(self._X[k].var['XWrd']) - (dw * np.sum(self._X[k].var['XAir']))
            four[k]      = np.sum(self._X[k].var['XTci']) - wi
            five[k]      = ((1 + dr) * np.sum(self._X[k].var['XAir'])) - (self._X[k].var['RAi'] * MFr.values)
            six[k]       = np.sum(self._X[k].var['XZrp']) - (self._X[k].var['PBp'] * MSp.values)
            seven[k]     = np.sum(self._X[k].var['XAir']) - (self._X[k].var['CYc'] * MCc.values)

            eight[k]     = self._X[k].var['XWrd'] - np.sum(MFt.values[0][0] * self._X[k].var['NTRDrd'] +
                                                  MFt.values[0][1] * self._X[k].var['NTRDrd'] +
                                                  MFt.values[0][2] * self._X[k].var['NTRDrd'] +
                                                  MFt.values[0][3] * self._X[k].var['NTRDrd'])

            nine[k]      = self._X[k].var['XAir'] * (1 + dr) - np.sum(MFt.values[0][0] * self._X[k].var['NTIRtir'] +
                                                            MFt.values[0][1] * self._X[k].var['NTIRtir'] +
                                                            MFt.values[0][2] * self._X[k].var['NTIRtir'] +
                                                            MFt.values[0][3] * self._X[k].var['NTIRtir'])

            ten[k]       = self._X[k].var['XZrp'] - np.sum(MFt.values[0][0] * self._X[k].var['NTRPrp'] +
                                                  MFt.values[0][1] * self._X[k].var['NTRPrp'] +
                                                  MFt.values[0][2] * self._X[k].var['NTRPrp'] +
                                                  MFt.values[0][3] * self._X[k].var['NTRPrp'])


            eleven[k]    = (self._X[k].var['XAir'] * dr) - np.sum(MFt.values[0][0] * self._X[k].var['NTRItri'] +
                                                         MFt.values[0][1] * self._X[k].var['NTRItri'] +
                                                         MFt.values[0][2] * self._X[k].var['NTRItri'] +
                                                         MFt.values[0][3] * self._X[k].var['NTRItri'])

            twelve[k]    = self._X[k].var['XTci'] - np.sum(MFt.values[0][0] * self._X[k].var['NTCIci'] +
                                                  MFt.values[0][1] * self._X[k].var['NTCIci'] +
                                                  MFt.values[0][2] * self._X[k].var['NTCIci'] +
                                                  MFt.values[0][3] * self._X[k].var['NTCIci'])

            thirteen[k]  = np.sum(self._X[k].var['PBp']) - self.p_max
            fourteen[k]  = np.sum(self._X[k].var['RAi']) - self.r_max
            fifteen[k]   = np.sum(self._X[k].var['CYc']) - self.c_max

            sixteen[k]   = WRr.values - self._X[k].var['VJRr']
            seventeen[k] = WPp.values - self._X[k].var['VJPp']
            eighteen[k]  = WCc.values - self._X[k].var['VJCc']


        equality = [one, two, three, four]

        inequality = [
            five, six, seven, eight, nine, ten,
            eleven, twelve, thirteen, fourteen,
            fifteen, sixteen, seventeen, eighteen
        ]

        return equality, inequality


    def make(self):

        f1, f2, f3 = self.f1_objective(), self.f2_objective(), self.f3_objective()
        self.objectives = np.column_stack([f1, f2, f3])
        self.constraints['eq'], self.constraints['ineq'] = self._constraints()



if __name__ == '__main__':

    from population import Population

    pop = Population(pop_size=3)
    pop.make()

    problem = ThisProblem(pop)
    problem.make()

    # for key, value in problem.objectives.items():
    #     if 'f' in key:
    #         print(key, value)
    #
    # print(problem.constraints)


    # print(problem)
    # print(problem.objectives['eq'])



class Population:

    def __init__(self):

        self.id_lists = [0]
        self._X = []

    def __repr__(self):
        return "<Population: size->{}>".format(len(self._X))

    def __len__(self):
        return len(self._X)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    def __getitem__(self, key):
        return self._X[key]

    def __setitem__(self, value, key):
        self._X[key] = value

    def generate(self, pop_size):
        for i in range(pop_size):
            _temp = Individual()
            self.X.append(deepcopy(_temp))

    def append(self, new_indv):
        self.X.append(new_indv)

    def remove(self, indx):
        self.X.pop(indx-1)

    @staticmethod
    def init_random(arr, lb, ub):

        new_arr = np.zeros_like(arr, dtype=arr.dtype)
        arr_type = arr.dtype
        if len(arr.shape) == 2:
            for i, j in np.ndindex(arr.shape):
                if arr_type == np.int64:
                    new_arr[i, j] = np.random.randint(lb, high=ub + 1)

                elif arr_type == np.float64:
                    new_arr[i, j] = np.random.uniform(low=lb, high=ub + 1)
        elif len(arr.shape) == 1:
            for i in np.ndindex(arr.shape):
                if arr_type == np.int64:
                    new_arr[i] = np.random.randint(lb, high=ub + 1)

                elif arr_type == np.float64:
                    new_arr[i] = np.random.uniform(low=lb, high=ub + 1)

        return new_arr


    # def empty_disc(self, empty_list=None):
    #      if empty_list:
    #         for _id in empty_list:
    #             for i in range(self.pop_size):
    #                 if self._X[i].id == _id:
    #                     self._disc[i] = {
    #                                     'position': None,
    #                                     'cost': None,
    #                                     'rank': None,
    #                                     'crowding_distance': None,
    #                                   }
    #      else:
    #         self._disc = [{
    #             'position': None,
    #             'cost': None,
    #             'rank': None,
    #             'crowding_distance': None,
    #         } for _ in range(self.pop_size)]



    def init(self):

        pop_size = len(pop)

        for i in range(pop_size):
            item = self._X[i].var
            ub   = self._X[i].ub
            lb   = self._X[i].lb

            for key, value in item.items():
                new_arr = __class__.init_random(value, lb[key], ub[key])
                self._X[i].var[key] = deepcopy(new_arr)

    def __next__(self):
        pass

    pop = Population()
    print('pop initialized ', pop.X)

    pop.generate(pop_size=30)
    print(pop.X)
    print(pop.X[0].var)
    pop.init()
    print(pop.X[0].var)
    print(pop.X[0].disc)
    pop.X[0].disc['cost'] = 0
    print(pop.X[0].disc)
    pop.append(new_indv=Individual())
    print(pop)
    pop.remove(31)
    print(pop)