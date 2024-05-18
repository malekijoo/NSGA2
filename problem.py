import numpy as np
from data import *
from population import population, init_pop
import pandas as pd

class ThisProblem:

    def __init__(self, n_obj, n_ie, n_e):

        self.c_max = 3
        self.p_max = 4
        self.r_max = 5
        self.n_obj, self.n_ie, self.n_e = n_obj, n_ie, n_e

    def f1_objective(self, individual):

        # one, two, three, four, five, six, seven, eight = [np.empty(shape=len(pop), dtype=object) for _ in range(8)]
        # z1 = np.empty(shape=len(pop), dtype=object)

        ''' Obj 1: Cost Min '''
        # for k in range(len(pop)):
        one     = np.sum(CPp.values * individual.X['PBp'])
        two     = np.sum(CFr.values * individual.X['RAi'])
        three   = np.sum(CEc.values * individual.X['CYc'])
        four    = np.sum(TCFir.T.values * individual.X['XAir'] * (1+dr))
        five    = np.sum(Tcs_rp.T.values * individual.X['XZrp'])
        six     = np.sum(Tct_rd.T.values * individual.X['XWrd'])
        seven   = np.sum(Tcc_cr.T.values * individual.X['XTci'])
        eight   = np.sum(Tcd_ri.T.values * (individual.X['XAir'] * dr))

        z1 = one + two  + three + four  + five + six + seven + eight

        return z1

    def f2_objective(self, individual):
        # one, two, three, four, five = [np.empty(shape=len(pop), dtype=object) for _ in range(5)]
        # z2 = np.empty(shape=len(pop), dtype=object)
        ''' Obj 2: CO2 Min '''

        one   = np.sum(COt.values * np.sum(DCIci.values * individual.X['NTCIci']))
        two   = np.sum(COt.values * np.sum(DIRir_I2R.values * individual.X['NTIRtir']))
        three = np.sum(COt.values * np.sum(DRPrp.values * individual.X['NTRPrp']))
        four  = np.sum(COt.values * np.sum(DRDrd.values * individual.X['NTRDrd']))
        five  = np.sum(COt.values * np.sum(DIRir_R2I.values * individual.X['NTRItri']))

        z2 = one + two + three + four + five

        return z2

    def f3_objective(self, individual):
        # one, two, three, four, five, six = [np.empty(shape=len(pop), dtype=object) for _ in range(6)]
        # z3 = np.empty(shape=len(pop), dtype=object)

        ''' Obj 3: Social factor Max'''

        one   = np.sum(FJRr.values * individual.X['RAi'])
        two   = np.sum(FJPp.values * individual.X['PBp'])
        three = np.sum(FJCc.values * individual.X['CYc'])
        four  = np.sum(individual.X['VJRr'] * individual.X['RAi'])
        five  = np.sum(individual.X['VJPp'] * individual.X['PBp'])
        six   = np.sum(individual.X['VJCc'] * individual.X['CYc'])

        z3 = -(one + two + three + four + five + six)

        return z3


    def c_one(self, individual):
        const = np.sum(individual.X['XAir'], axis=1) - np.sum(individual.X['XTci'], axis=0)
        # print('const one', const)
        return (const==0)

    def c_two(self, individual):
        const = np.sum(individual.X['XZrp'], axis=1) - ((1 - dw) * np.sum(individual.X['XAir'], axis=0))
        # print('const two ', const)
        return (const==0)

    def c_three(self, individual):
        const =  np.sum(individual.X['XWrd'], axis=1) - (dw * np.sum(individual.X['XAir'], axis=0))
        # print('const three ', const)
        return (const==0)

    def c_four(self, individual):
        const = np.sum(individual.X['XTci'], axis=1) - wi
        # print('const four ', const)
        return (const==0)

    def c_five(self, individual):
        const = ((1 + dr) * np.sum(individual.X['XAir'], axis=0)) - (individual.X['RAi'] * MFr.values)
        # print('const five ', const)
        return (const[0]<=0)

    def c_six(self, individual):
        const = np.sum(individual.X['XZrp'], axis=0) - (individual.X['PBp'] * MSp.values)
        # print('const six ', const)
        return (const[0]<=0)

    def c_seven(self, individual):
        const = np.sum(individual.X['XTci'], axis=1) - (individual.X['CYc'] * MCc.values)
        # print('const seven ', const)
        return (const[0]<=0)

    def c_eight(self, individual):
        const = individual.X['XWrd'] - (MFt.values[0][0] * individual.X['NTRDrd'] +
                                        MFt.values[0][1] * individual.X['NTRDrd'] +
                                        MFt.values[0][2] * individual.X['NTRDrd'] +
                                        MFt.values[0][3] * individual.X['NTRDrd'])
        # print('const 8 ', const)
        return (const<=0)

    def c_nine(self, individual):
        const = individual.X['XAir'] * (1 + dr) - (MFt.values[0][0] * individual.X['NTIRtir'] +
                                                            MFt.values[0][1] * individual.X['NTIRtir'] +
                                                            MFt.values[0][2] * individual.X['NTIRtir'] +
                                                            MFt.values[0][3] * individual.X['NTIRtir'])
        # print('const 9 ', const)
        return (const<=0)

    def c_ten(self, individual):
        const = individual.X['XZrp'] - (MFt.values[0][0] * individual.X['NTRPrp'] +
                                                MFt.values[0][1] * individual.X['NTRPrp'] +
                                                MFt.values[0][2] * individual.X['NTRPrp'] +
                                                MFt.values[0][3] * individual.X['NTRPrp'])
        # print('const 10 ', const)
        return (const<=0)

    def c_eleven(self, individual):
        const = (individual.X['XAir'] * dr) - (MFt.values[0][0] * individual.X['NTRItri'] +
                                                        MFt.values[0][1] * individual.X['NTRItri'] +
                                                        MFt.values[0][2] * individual.X['NTRItri'] +
                                                        MFt.values[0][3] * individual.X['NTRItri'])
        # print('const 11 ', const)
        return (const<=0)

    def c_twelve(self, individual):
        const = individual.X['XTci'] - (MFt.values[0][0] * individual.X['NTCIci'] +
                                                 MFt.values[0][1] * individual.X['NTCIci'] +
                                                 MFt.values[0][2] * individual.X['NTCIci'] +
                                                 MFt.values[0][3] * individual.X['NTCIci'])
        # print('const 12 ', const)
        return (const<=0)

    def c_thirteen(self, individual):
        const = np.sum(individual.X['PBp']) - self.p_max
        # print('const 13 ', const)
        return (np.array([const])<=0)

    def c_fourteen(self, individual):
        const = np.sum(individual.X['RAi']) - self.r_max
        # print('const 14 ', const)
        return (np.array([const])<=0)

    def c_fifteen(self, individual):
        const = np.sum(individual.X['CYc']) - self.c_max
        # print('const 15 ', const)
        return (np.array([const])<=0)

    def c_sixteen(self, individual):
        const = WRr.values - individual.X['VJRr']
        # print('const 16 ', const)
        return (const[0]<=0)

    def c_seventeen(self, individual):
        const = WPp.values - individual.X['VJPp']
        # print('const 17 ', const)
        return (const[0]<=0)

    def c_eighteen(self, individual):
        const = WCc.values - individual.X['VJCc']
        # print('const 18 ', const)
        return (const[0]<=0)


    def _constraints(self, individual):

        one       = self.c_one(individual)
        two       = self.c_two(individual)
        three     = self.c_three(individual)
        four      = self.c_four(individual)

        five      = self.c_five(individual)
        six       = self.c_six(individual)
        seven     = self.c_seven(individual)

        eight     = self.c_eight(individual)
        nine      = self.c_nine(individual)
        ten       = self.c_ten(individual)
        eleven    = self.c_eleven(individual)
        twelve    = self.c_twelve(individual)

        thirteen  = self.c_thirteen(individual)
        fourteen  = self.c_fourteen(individual)
        fifteen   = self.c_fifteen(individual)

        sixteen   = self.c_sixteen(individual)
        seventeen = self.c_seventeen(individual)
        eighteen  = self.c_eighteen(individual)




        constraints = np.concatenate([
            one, two, three, four,
            five, six, seven, eight.flatten(), nine.flatten(), ten.flatten(),
            eleven.flatten(), twelve.flatten(), thirteen, fourteen,
            fifteen, sixteen, seventeen, eighteen
        ])

        return np.count_nonzero(~constraints)

    def calculate_obj_const(self, pop):

        for i in range(len(pop)):
            f1 = self.f1_objective(pop[i])
            f2 = self.f2_objective(pop[i])
            f3 = self.f3_objective(pop[i])
            pop[i].objectives = np.array([f1, f2, f3])
            pop[i].constraint_violation_count = self._constraints(pop[i])



if __name__ == "__main__":
    pop = population(1)
    pop = init_pop(pop)
    thisproblem = ThisProblem(n_obj=3, n_ie=13, n_e=4)
    thisproblem.calculate_obj_const(pop)
