import numpy as np
from copy import deepcopy


class Individual:

    def __init__(self):
        # super.__init__()

        self.objectives = None
        # self.eq_const = None
        self.constraint_violation_count = None

        self._X, self._lb, self._ub = {}, {}, {}


        self.rank = None
        self.crowding_distance = None
        self.position = None
        self.F = None

        self._X, self._lb, self._ub = Individual.creat_variables()


    def __repr__(self):
        return f"< Individual Rank:{self.rank}>"

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, key):
        return self._X[key]

    def __setitem__(self, value, key):
        self._X[key] = deepcopy(value)

    def __getitem__(self, key):
        return self._X[key]

    @property
    def lb(self):
        return self._lb

    @property
    def ub(self):
        return self._ub


    @staticmethod
    def creat_variables():

        X, lb, ub = {}, {}, {}

        # # R=5  ( Total = 5 ) RAi
        X['RAi']  = np.zeros(shape=(5,), dtype=int)
        lb['RAi'], ub['RAi'] = 0, 1

        # # P=4  ( Total = 4 ) PBp
        X['PBp'] = np.zeros(shape=(4,), dtype=int)
        lb['PBp'], ub['PBp'] = 0, 1

        # # C=3  ( Total = 3 ) CYc
        X['CYc'] = np.zeros(shape=(3,), dtype=int)
        lb['CYc'], ub['CYc'] = 0, 1

        # # C=3 , I=5 ( Total = 15 ) NTCIci
        X['NTCIci'] = np.zeros(shape=(3, 5), dtype=int)
        lb['NTCIci'], ub['NTCIci'] = 0, 10 ** 7

        # # I=5 , R=5 ( Total = 25 ) NTIRtir
        X['NTIRtir'] = np.zeros(shape=(5, 5), dtype=int)
        lb['NTIRtir'], ub['NTIRtir'] = 0, 10 ** 7

        # # R=5 , P=4 ( Total = 20 )
        X['NTRPrp'] = np.zeros(shape=(5, 4), dtype=int)
        lb['NTRPrp'], ub['NTRPrp'] = 0, 10 ** 7

        # # # R=5 , D=3 ( Total = 15 ) NTRDrd
        X['NTRDrd'] = np.zeros(shape=(5, 3), dtype=int)
        lb['NTRDrd'], ub['NTRDrd'] = 0, 10 ** 7

        # # R=5 , I=5 ( Total = 25 ) NTRItri
        X['NTRItri'] = np.zeros(shape=(5, 5), dtype=int)
        lb['NTRItri'], ub['NTRItri'] = 0, 10 ** 7

        # # # R=5  ( Total = 5 )
        X['VJRr']  = np.zeros(shape=(5,), dtype=int)
        lb['VJRr'], ub['VJRr'] = 0, 10 ** 7

        # # # P=4  ( Total = 4 )
        X['VJPp']  = np.zeros(shape=(4,), dtype=int)
        lb['VJPp'], ub['VJPp'] = 0, 10 ** 7

        # # # C=3  ( Total = 3 )
        X['VJCc']  = np.zeros(shape=(3,), dtype=int)
        lb['VJCc'], ub['VJCc'] = 0, 10 ** 7

        # # # I=5 , R=5 ( Total = 25 ) XAir
        X['XAir']  = np.zeros(shape=(5, 5), dtype=float)
        lb['XAir'], ub['XAir'] = 0, 10 ** 7

        # #
        # # R=5 , P=4 ( Total = 20 ) XZrp
        X['XZrp']  = np.zeros(shape=(5, 4), dtype=float)
        lb['XZrp'], ub['XZrp'] = 0, 10 ** 7
        # #
        # # R=5 , D=3 ( Total = 15 )
        X['XWrd']  = np.zeros(shape=(5, 3), dtype=float)
        lb['XWrd'], ub['XWrd'] = 0, 10 ** 7

        # # # C=3 , I=5 ( Total = 15 )
        X['XTci']  = np.zeros(shape=(3, 5), dtype=float)
        lb['XTci'], ub['XTci'] = 0, 10 ** 7

        return X, lb, ub


