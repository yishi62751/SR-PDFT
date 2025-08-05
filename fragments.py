"""
08.05.2025
Yi Shi

This code was originally developed by Dr. Yuming Shi.
It serves as an outer layer for performing Kohn-Sham DFT calculations on individual Partition-DFT fragments sequentially.

Before calling the functions in this module, ensure the following:
(1) Fragment Definition: The fragments must be explicitly defined in your script.
(2) Ensemble Treatment: Fragments must be treated as ensembles, with their components and weights properly specified.
(3) The partition potential (Vp) must be incorporated into the PySCF SCF solver.

(1) The fragments are unambiguously in your script.
(2) The fragments are treated as ensembles, with ensemble components and weights specified.
(3) The partition potential Vp has been added to the SCF solver in pyscf.

The implementation involves:
(1) Adding a Vp property to the pyscf.scf.hf object.
(2) Modifying pyscf.scf.hf.get_hcore() to include Vp in the core Hamiltonian.

Code snippet:

class SCF(lib.StreamObject):
    ....
    Vp = 0.
    ....
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol) + self.Vp

Note that: All calculations here are spin-unrestricted to allow for ensemble representation.
"""

from pyscf import scf, dft
import numpy as np

class FragmentDFT:
    def __init__(self, mol, xc):
        self.mol = mol
        self.solver = self.dftsolver = dft.UKS(mol)
        self.dftsolver.xc = xc
        # one could also specify the dft grid point 
        
        self.S = self.mol.intor('int1e_ovlp')
        self.E = None # total energy WITHOUT vp contribution
        self.D = None # on AO, (Da, Db)
        # ---------------------------------------
        # The following are unnecessary attributes. You might want to trigger them as you will.
        # self.V = self.mol.intor('int1e_nuc')
        # self.T = self.mol.intor('int1e_kin')
        # self.eri = self.mol.intor('int2e') # 4-rank, large.
        # self.Exc = None
        # self.EH = None
        # self.Eext = None
        
    def get_rdm1(self):
        Da, Db = self.dftsolver.make_rdm1()
        
        # checker, could be deleted.
        assert np.isclose(np.sum(Da * self.S), self.mol.nelec[0], atol=1e-3), (np.sum(Da * self.S), self.mol.nelec[0])
        assert np.isclose(np.sum(Db * self.S), self.mol.nelec[1], atol=1e-3), (np.sum(Db * self.S), self.mol.nelec[1])
        
        self.D = np.array((Da, Db))
        return Da, Db

    def kernel(self, *args, Vp=None, dm0=None, **kwargs):
        """
        SCF runner.
        dm0: initial guess. If None, use the dm of the last iteration.
        """
        if dm0 is None:
            dm0 = self.D
        
        if Vp is not None:
            self.dftsolver.Vp = Vp
            self.dftsolver.kernel(*args, dm0=dm0, **kwargs)
            Da, Db = self.get_rdm1()
            self.E = self.dftsolver.e_tot - np.sum((Da+Db) * Vp)
        else:
            self.dftsolver.Vp = 0.
            self.dftsolver.kernel(*args, dm0=dm0, **kwargs)
            Da, Db = self.get_rdm1()
            self.E = self.dftsolver.e_tot
        
        return self.E

class ens():
    """This is an extra layer of ensemble for spin. with spin-restricted Vp."""
    def recursive_sum(self, t):
        total = 0
        for item in t:
            if isinstance(item, (list, tuple)):
                assert len(item) == 2, item
                total += sum(item)
            else:
                total += item
        return total
    
    def __init__(self, fragments, omega):
        """
        fragments: a list of fragments calculated as defined above
        omega: a list ensemble weights.
            The current rule is that if the element is a number, it is the weight of the corresponding fragment in self.fragments.
            If the element is itself a list of two numbers, then the spin-flipped result is automatically generated without calculation for those two numbers.
            All the numbers must sum to 1.
            E.g. (spin-up, spin-down)
            fragments = [|(4,5)>, |(5,5)>]
            omega = [(0.4, 0.4), 0.2]
            real ensemble:
            |ens> = 0.4 * |(4,5)> + 0.4 * |(5,4)> + 0.2 * |(5,5)>.
        """
        assert np.isclose(self.recursive_sum(omega), 1), self.recursive_sum(omega)
        assert len(fragments) == len(omega)
        self.fragments = fragments
        self.omega = omega

    def scf(self, Vp, *args, **keywords):
        for frag in self.fragments:
            frag.kernel(Vp=Vp, *args, **keywords)
        return
    
    def get_D(self):
        Da = 0.
        Db = 0.
        for w, frag in zip(self.omega, self.fragments):
            Da_, Db_ = frag.get_rdm1()
            if isinstance(w, (list, tuple)):
                wa, wb = w
                Da += Da_ * wa
                Db += Db_ * wa
                # flip the dm
                Da += Db_ * wb
                Db += Da_ * wb
            else:
                Da += Da_ * w
                Db += Db_ * w
        return Da, Db
    
    def get_E(self):
        E = 0.
        for w, frag in zip(self.omega, self.fragments):
            if isinstance(w, (list, tuple)):
                wa, wb = w
                E += frag.E * (wa+wb)
            else:
                E += frag.E * w
        return E

def get_Exc(ks, mol, D):
    veff = ks.get_veff(mol, D)
    Exc = veff.exc.real
    return Exc
