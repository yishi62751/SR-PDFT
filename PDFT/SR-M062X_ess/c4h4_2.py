import numpy as np
import pickle
import scipy
from pyscf import gto, scf, dft, lib
from pyscf.lib import chkfile

class FragmentDFT:
    def __init__(self, mol, xc):
        self.mol = mol
        self.solver = self.dftsolver = dft.UKS(mol) # does not necessarily have to be UKS, right?
        self.dftsolver.xc = xc
        # one could also specify the dft grid point 
        
        self.S = self.mol.intor('int1e_ovlp')
        self.E = None # total energy WITHOUT vp contribution
        self.D = None # on AO, (Da, Db)
        # ---------------------------------------
        # The following are unnecessary attributes. You might want to trigger them when necessary.
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
        dm0: initial guess. If None, use the dm of last iteration.
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
    """this is an extra layer of ensemble for spin. with spin-restricted Vp."""
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
        fragments: a list of fragment calculate defined above
        omega: a list ensemble weights.
            The current rule is, if the element is a number, it is the weight of the correspoinding fragment in self.fragments.
            If the element is itself a list of two numbers, then the a spin fliped result is automatically generated without calculation for those two numbers.
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

lib.num_threads = 32
basis  = 'ccpvqz'
ghostbasis = 'sto3g'
maxiter = 1000
reg = 0.
steps = np.linspace(1., 1., maxiter)

# Promolecular calculation
mol_geo = gto.M(atom =f"""
C         -0.67400        0.00180       -0.77570
C         -0.67320       -0.00670        0.77630
C          0.67400       -0.00180        0.77570
C          0.67320        0.00670       -0.77630
H         -1.45330        0.00300       -1.53370
H         -1.45180       -0.01370        1.53500
H          1.45330       -0.00300        1.53370
H          1.45180        0.01370       -1.53500 """, unit='ANG', basis=basis)

mol_dft = dft.UKS(mol_geo)
mol_dft.xc = 'm062x'
dm = mol_dft.from_chk('c4h4_2.chk')
mol_dft.kernel(dm)
Da, Db = mol_dft.make_rdm1()
D = np.array((Da, Db))
phi = mol_dft._numint.eval_ao(mol_geo, mol_dft.grids.coords, deriv=0)
x,y,z = mol_dft.grids.coords.T
w   = mol_dft.grids.weights
na = dft.numint.eval_rho(mol_geo, phi, Da)
nb = dft.numint.eval_rho(mol_geo, phi, Db)
ntot = np.array((na, nb))
nin = na + nb
spin_pop, Ms = mol_dft.mulliken_spin_pop(mol_geo, D)
M = abs(Ms[0] + Ms[4])
print(f"M={M:.6f}.")

# frag info
CH1_geo = gto.M(atom=[
        ['C', (-0.67400, 0.00180, -0.77570)],
        ['ghost-C', (-0.67320, -0.00670, 0.77630)],
        ['ghost-C', (0.67400, -0.00180, 0.77570)],
        ['ghost-C', (0.67320, 0.00670, -0.77630)],
        ['H', (-1.45330, 0.00300, -1.53370)],
        ['ghost-H', (-1.45180, -0.01370, 1.53500)],
        ['ghost-H', (1.45330, -0.00300, 1.53370)],
        ['ghost-H', (1.45180, 0.01370, -1.53500)],
],
unit='ANG', basis={'C': basis, 'ghost-C': ghostbasis, 'H': basis, 'ghost-H': ghostbasis}, spin=3) 
CH1dft = FragmentDFT(CH1_geo, 'm062x')
CH1dft.kernel()
phiCH1 = dft.numint.eval_ao(CH1_geo, mol_dft.grids.coords, deriv=0)

CH2_geo = gto.M(atom=[
        ['ghost-C', (-0.67400, 0.00180, -0.77570)],
        ['C', (-0.67320, -0.00670, 0.77630)],
        ['ghost-C', (0.67400, -0.00180, 0.77570)],
        ['ghost-C', (0.67320, 0.00670, -0.77630)],
        ['ghost-H', (-1.45330, 0.00300, -1.53370)],
        ['H', (-1.45180, -0.01370, 1.53500)],
        ['ghost-H', (1.45330, -0.00300, 1.53370)],
        ['ghost-H', (1.45180, 0.01370, -1.53500)],
],
unit='ANG', basis={'C': basis, 'ghost-C': ghostbasis, 'H': basis, 'ghost-H': ghostbasis}, spin=3)  
CH2dft = FragmentDFT(CH2_geo, 'm062x')
CH2dft.kernel()
phiCH2 = dft.numint.eval_ao(CH2_geo, mol_dft.grids.coords, deriv=0)
    
CH3_geo = gto.M(atom=[
        ['ghost-C', (-0.67400, 0.00180, -0.77570)],
        ['ghost-C', (-0.67320, -0.00670, 0.77630)],
        ['C', (0.67400, -0.00180, 0.77570)],
        ['ghost-C', (0.67320, 0.00670, -0.77630)],
        ['ghost-H', (-1.45330, 0.00300, -1.53370)],
        ['ghost-H', (-1.45180, -0.01370, 1.53500)],
        ['H', (1.45330, -0.00300, 1.53370)],
        ['ghost-H', (1.45180, 0.01370, -1.53500)],
],
unit='ANG', basis={'C': basis, 'ghost-C': ghostbasis, 'H': basis, 'ghost-H': ghostbasis}, spin=3) 
CH3dft = FragmentDFT(CH3_geo, 'm062x')
CH3dft.kernel()
phiCH3 = dft.numint.eval_ao(CH3_geo, mol_dft.grids.coords, deriv=0)

CH4_geo = gto.M(atom=[
        ['ghost-C', (-0.67400, 0.00180, -0.77570)],
        ['ghost-C', (-0.67320, -0.00670, 0.77630)],
        ['ghost-C', (0.67400, -0.00180, 0.77570)],
        ['C', (0.67320, 0.00670, -0.77630)],
        ['ghost-H', (-1.45330, 0.00300, -1.53370)],
        ['ghost-H', (-1.45180, -0.01370, 1.53500)],
        ['ghost-H', (1.45330, -0.00300, 1.53370)],
        ['H', (1.45180, 0.01370, -1.53500)],
],
unit='ANG', basis={'C': basis, 'ghost-C': ghostbasis, 'H': basis, 'ghost-H': ghostbasis}, spin=3) 
CH4dft = FragmentDFT(CH4_geo, 'm062x')
CH4dft.kernel()
phiCH4 = dft.numint.eval_ao(CH4_geo, mol_dft.grids.coords, deriv=0)

CH1 = ens([CH1dft, ],  (((3+M)/6, (3-M)/6),))
CH2 = ens([CH2dft, ],  (((3-M)/6, (3+M)/6),))
CH3 = ens([CH3dft, ],  (((3+M)/6, (3-M)/6),))
CH4 = ens([CH4dft, ],  (((3-M)/6, (3+M)/6),))

# First iteration
CH1.scf(None)
CH2.scf(None)
CH3.scf(None)
CH4.scf(None)
DaCH1, DbCH1 = CH1.get_D()
DaCH2, DbCH2 = CH2.get_D()
DaCH3, DbCH3 = CH3.get_D()
DaCH4, DbCH4 = CH4.get_D()
naCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DaCH1)
nbCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DbCH1)
naCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DaCH2)
nbCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DbCH2)
naCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DaCH3)
nbCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DbCH3)
naCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DaCH4)
nbCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DbCH4)
nfa = naCH1 + naCH2 + naCH3 + naCH4
nfb = nbCH1 + nbCH2 + nbCH3 + nbCH4
NaCH1 = np.sum(naCH1*w)
NaCH2 = np.sum(naCH2*w)
NaCH3 = np.sum(naCH3*w)
NaCH4 = np.sum(naCH4*w)
NbCH1 = np.sum(nbCH1*w)
NbCH2 = np.sum(nbCH2*w)
NbCH3 = np.sum(nbCH3*w)
NbCH4 = np.sum(nbCH4*w)
print("NaCH1:", NaCH1)
print("NbCH1:", NbCH1)
print("NaCH2:", NaCH2)
print("NbCH2:", NbCH2)
print("NaCH3:", NaCH3)
print("NbCH3:", NbCH3)
print("NaCH4:", NaCH4)
print("NbCH4:", NbCH4)
vp = 0.
Vp = np.zeros_like(DaCH1)

CH1.scf(Vp)
CH2.scf(Vp)
CH3.scf(Vp)
CH4.scf(Vp)
DaCH1, DbCH1 = CH1.get_D()
DaCH2, DbCH2 = CH2.get_D()
DaCH3, DbCH3 = CH3.get_D()
DaCH4, DbCH4 = CH4.get_D()
ECH1 = CH1.get_E()
ECH2 = CH2.get_E()
ECH3 = CH3.get_E()
ECH4 = CH4.get_E()
Ef = ECH1 + ECH2 + ECH3 + ECH4
naCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DaCH1)
nbCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DbCH1)
naCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DaCH2)
nbCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DbCH2)
naCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DaCH3)
nbCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DbCH3)
naCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DaCH4)
nbCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DbCH4)
nfa = naCH1 + naCH2 + naCH3 + naCH4
nfb = nbCH1 + nbCH2 + nbCH3 + nbCH4
nf = nfa + nfb
assert np.isclose(np.sum(nf * w), 28., atol=1e-4), np.sum(nf * w)
print(f"Ef={Ef:.5f} L1={np.sum(np.abs(nin-nf)*w):.4f} ECH1={CH1dft.dftsolver.e_tot:.5f} ECH2={CH2dft.dftsolver.e_tot:.5f} ECH3={CH3dft.dftsolver.e_tot:.5f} ECH4={CH4dft.dftsolver.e_tot:.5f}.")

# Partition-DFT iterations
L = np.sum(np.abs(nin-nf)*w)
phiCH1 = dft.numint.eval_ao(CH1_geo, mol_dft.grids.coords, deriv=0)
phiCH2 = dft.numint.eval_ao(CH2_geo, mol_dft.grids.coords, deriv=0)
phiCH3 = dft.numint.eval_ao(CH3_geo, mol_dft.grids.coords, deriv=0)
phiCH4 = dft.numint.eval_ao(CH4_geo, mol_dft.grids.coords, deriv=0)
for itera, thisstep in enumerate(steps):
    vp += thisstep * L * (nf - nin)# - reg * vp
    VpCH1 = np.einsum("p,pu,pv->uv", w*vp, phiCH1, phiCH1, optimize=True)
    VpCH1 = 0.5 * (VpCH1 + VpCH1.T)
    VpCH2 = np.einsum("p,pu,pv->uv", w*vp, phiCH2, phiCH2, optimize=True)
    VpCH2 = 0.5 * (VpCH2 + VpCH2.T)
    VpCH3 = np.einsum("p,pu,pv->uv", w*vp, phiCH3, phiCH3, optimize=True)
    VpCH3 = 0.5 * (VpCH3 + VpCH3.T)
    VpCH4 = np.einsum("p,pu,pv->uv", w*vp, phiCH4, phiCH4, optimize=True)
    VpCH4 = 0.5 * (VpCH4 + VpCH4.T)
    CH1.scf(VpCH1)
    CH2.scf(VpCH2)
    CH3.scf(VpCH3)
    CH4.scf(VpCH4)
    DaCH1, DbCH1 = CH1.get_D()
    DaCH2, DbCH2 = CH2.get_D()
    DaCH3, DbCH3 = CH3.get_D()
    DaCH4, DbCH4 = CH4.get_D()
    ECH1 = CH1.get_E()
    ECH2 = CH2.get_E()
    ECH3 = CH3.get_E()
    ECH4 = CH4.get_E()
    Ef = ECH1 + ECH2 + ECH3 + ECH4
    naCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DaCH1)
    nbCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DbCH1)
    naCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DaCH2)
    nbCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DbCH2)
    naCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DaCH3)
    nbCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DbCH3)
    naCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DaCH4)
    nbCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DbCH4)
    nfa = naCH1 + naCH2 + naCH3 + naCH4
    nfb = nbCH1 + nbCH2 + nbCH3 + nbCH4
    nf = nfa + nfb
    L = np.sum(np.abs(nin-nf)*w)
    print(f"itera={itera:3d} step={thisstep*L:.2f} Ef={Ef:.5f} L1={L:.4f}.")

print(f"final Ef={Ef:.5f} L1={L:.4f} ECH1={CH1dft.dftsolver.e_tot:.5f} ECH2={CH2dft.dftsolver.e_tot:.5f} ECH3={CH3dft.dftsolver.e_tot:.5f} ECH4={CH4dft.dftsolver.e_tot:.5f}")

# Compute Exc
phiCH1 = dft.numint.eval_ao(CH1_geo, mol_dft.grids.coords, deriv=1)
phiCH2 = dft.numint.eval_ao(CH2_geo, mol_dft.grids.coords, deriv=1)
phiCH3 = dft.numint.eval_ao(CH3_geo, mol_dft.grids.coords, deriv=1)
phiCH4 = dft.numint.eval_ao(CH4_geo, mol_dft.grids.coords, deriv=1)
naCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DaCH1, xctype='hyb')
nbCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DbCH1, xctype='hyb')
naCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DaCH2, xctype='hyb')
nbCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DbCH2, xctype='hyb')
naCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DaCH3, xctype='hyb')
nbCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DbCH3, xctype='hyb')
naCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DaCH4, xctype='hyb')
nbCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DbCH4, xctype='hyb')
nfa = naCH1 + naCH2 + naCH3 + naCH4
nfb = nbCH1 + nbCH2 + nbCH3 + nbCH4
ntot = np.array((nfa, nfb))
exc, vxc, fxc, kxc = dft.libxc.eval_xc('m062x', ntot, spin=1, relativity=0, deriv=1, omega=None, verbose=None)
Exctot = np.sum((nfa[0]+nfb[0])*w * exc)
print(f"M={M:.2f} Total XC energy={Exctot:.5f}.")

# M_sr
M_sr = 0.
CH1dft = FragmentDFT(CH1_geo, 'm062x')
CH1dft.kernel()
phiCH1 = dft.numint.eval_ao(CH1_geo, mol_dft.grids.coords, deriv=0)
CH2dft = FragmentDFT(CH2_geo, 'm062x')
CH2dft.kernel()
phiCH2 = dft.numint.eval_ao(CH2_geo, mol_dft.grids.coords, deriv=0)
CH3dft = FragmentDFT(CH3_geo, 'm062x')
CH3dft.kernel()
phiCH3 = dft.numint.eval_ao(CH3_geo, mol_dft.grids.coords, deriv=0)
CH4dft = FragmentDFT(CH4_geo, 'm062x')
CH4dft.kernel()
phiCH4 = dft.numint.eval_ao(CH4_geo, mol_dft.grids.coords, deriv=0)
CH1 = ens([CH1dft, ],  (((3+M_sr)/6, (3-M_sr)/6),))
CH2 = ens([CH2dft, ],  (((3-M_sr)/6, (3+M_sr)/6),))
CH3 = ens([CH3dft, ],  (((3+M_sr)/6, (3-M_sr)/6),))
CH4 = ens([CH4dft, ],  (((3-M_sr)/6, (3+M_sr)/6),))

# First iteration
CH1.scf(None)
CH2.scf(None)
CH3.scf(None)
CH4.scf(None)
DaCH1, DbCH1 = CH1.get_D()
DaCH2, DbCH2 = CH2.get_D()
DaCH3, DbCH3 = CH3.get_D()
DaCH4, DbCH4 = CH4.get_D()
naCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DaCH1)
nbCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DbCH1)
naCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DaCH2)
nbCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DbCH2)
naCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DaCH3)
nbCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DbCH3)
naCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DaCH4)
nbCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DbCH4)
nfa = naCH1 + naCH2 + naCH3 + naCH4
nfb = nbCH1 + nbCH2 + nbCH3 + nbCH4
NaCH1 = np.sum(naCH1*w)
NaCH2 = np.sum(naCH2*w)
NaCH3 = np.sum(naCH3*w)
NaCH4 = np.sum(naCH4*w)
NbCH1 = np.sum(nbCH1*w)
NbCH2 = np.sum(nbCH2*w)
NbCH3 = np.sum(nbCH3*w)
NbCH4 = np.sum(nbCH4*w)
print("NaCH1:", NaCH1)
print("NbCH1:", NbCH1)
print("NaCH2:", NaCH2)
print("NbCH2:", NbCH2)
print("NaCH3:", NaCH3)
print("NbCH3:", NbCH3)
print("NaCH4:", NaCH4)
print("NbCH4:", NbCH4)
vp = 0.
Vp = np.zeros_like(DaCH1)

CH1.scf(Vp)
CH2.scf(Vp)
CH3.scf(Vp)
CH4.scf(Vp)
DaCH1, DbCH1 = CH1.get_D()
DaCH2, DbCH2 = CH2.get_D()
DaCH3, DbCH3 = CH3.get_D()
DaCH4, DbCH4 = CH4.get_D()
ECH1 = CH1.get_E()
ECH2 = CH2.get_E()
ECH3 = CH3.get_E()
ECH4 = CH4.get_E()
Ef = ECH1 + ECH2 + ECH3 + ECH4
naCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DaCH1)
nbCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DbCH1)
naCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DaCH2)
nbCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DbCH2)
naCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DaCH3)
nbCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DbCH3)
naCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DaCH4)
nbCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DbCH4)
nfa = naCH1 + naCH2 + naCH3 + naCH4
nfb = nbCH1 + nbCH2 + nbCH3 + nbCH4
nf = nfa + nfb
assert np.isclose(np.sum(nf * w), 28., atol=1e-4), np.sum(nf * w)

# Partition-DFT iterations
L = np.sum(np.abs(nin-nf)*w)
phiCH1 = dft.numint.eval_ao(CH1_geo, mol_dft.grids.coords, deriv=0)
phiCH2 = dft.numint.eval_ao(CH2_geo, mol_dft.grids.coords, deriv=0)
phiCH3 = dft.numint.eval_ao(CH3_geo, mol_dft.grids.coords, deriv=0)
phiCH4 = dft.numint.eval_ao(CH4_geo, mol_dft.grids.coords, deriv=0)
for itera, thisstep in enumerate(steps):
    vp += thisstep * L * (nf - nin)# - reg * vp
    VpCH1 = np.einsum("p,pu,pv->uv", w*vp, phiCH1, phiCH1, optimize=True)
    VpCH1 = 0.5 * (VpCH1 + VpCH1.T)
    VpCH2 = np.einsum("p,pu,pv->uv", w*vp, phiCH2, phiCH2, optimize=True)
    VpCH2 = 0.5 * (VpCH2 + VpCH2.T)
    VpCH3 = np.einsum("p,pu,pv->uv", w*vp, phiCH3, phiCH3, optimize=True)
    VpCH3 = 0.5 * (VpCH3 + VpCH3.T)
    VpCH4 = np.einsum("p,pu,pv->uv", w*vp, phiCH4, phiCH4, optimize=True)
    VpCH4 = 0.5 * (VpCH4 + VpCH4.T)
    CH1.scf(VpCH1)
    CH2.scf(VpCH2)
    CH3.scf(VpCH3)
    CH4.scf(VpCH4)
    DaCH1, DbCH1 = CH1.get_D()
    DaCH2, DbCH2 = CH2.get_D()
    DaCH3, DbCH3 = CH3.get_D()
    DaCH4, DbCH4 = CH4.get_D()
    ECH1 = CH1.get_E()
    ECH2 = CH2.get_E()
    ECH3 = CH3.get_E()
    ECH4 = CH4.get_E()
    Ef = ECH1 + ECH2 + ECH3 + ECH4
    naCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DaCH1)
    nbCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DbCH1)
    naCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DaCH2)
    nbCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DbCH2)
    naCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DaCH3)
    nbCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DbCH3)
    naCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DaCH4)
    nbCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DbCH4)
    nfa = naCH1 + naCH2 + naCH3 + naCH4
    nfb = nbCH1 + nbCH2 + nbCH3 + nbCH4
    nf = nfa + nfb
    L = np.sum(np.abs(nin-nf)*w)
    print(f"itera={itera:3d} step={thisstep*L:.2f} Ef={Ef:.5f} L1={L:.4f}.")

print(f"final Ef={Ef:.5f} L1={L:.4f} ECH1={CH1dft.dftsolver.e_tot:.5f} ECH2={CH2dft.dftsolver.e_tot:.5f} ECH3={CH3dft.dftsolver.e_tot:.5f} ECH4={CH4dft.dftsolver.e_tot:.5f}")


# Compute Excnad
phiCH1 = dft.numint.eval_ao(CH1_geo, mol_dft.grids.coords, deriv=1)
phiCH2 = dft.numint.eval_ao(CH2_geo, mol_dft.grids.coords, deriv=1)
phiCH3 = dft.numint.eval_ao(CH3_geo, mol_dft.grids.coords, deriv=1)
phiCH4 = dft.numint.eval_ao(CH4_geo, mol_dft.grids.coords, deriv=1)
naCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DaCH1, xctype='hyb')
nbCH1 = dft.numint.eval_rho(mol_geo, phiCH1, DbCH1, xctype='hyb')
naCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DaCH2, xctype='hyb')
nbCH2 = dft.numint.eval_rho(mol_geo, phiCH2, DbCH2, xctype='hyb')
naCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DaCH3, xctype='hyb')
nbCH3 = dft.numint.eval_rho(mol_geo, phiCH3, DbCH3, xctype='hyb')
naCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DaCH4, xctype='hyb')
nbCH4 = dft.numint.eval_rho(mol_geo, phiCH4, DbCH4, xctype='hyb')
n1 = np.array((naCH1, nbCH1))
n2 = np.array((naCH2, nbCH2))
n3 = np.array((naCH3, nbCH3))
n4 = np.array((naCH4, nbCH4))
nfa = naCH1 + naCH2 + naCH3 + naCH4
nfb = nbCH1 + nbCH2 + nbCH3 + nbCH4
ntot = np.array((nfa, nfb))
exc, vxc, fxc, kxc = dft.libxc.eval_xc('m062x', ntot, spin=1, relativity=0, deriv=1, omega=None, verbose=None)
Exctot_sr = np.sum((nfa[0]+nfb[0])*w * exc)
exc1, vxc1, fxc1, kxc1 = dft.libxc.eval_xc('m062x', n1, spin=1, relativity=0, deriv=1, omega=None, verbose=None)
ExcCH1 = np.sum((naCH1[0]+nbCH1[0])*w * exc1)
exc2, vxc2, fxc2, kxc2 = dft.libxc.eval_xc('m062x', n2, spin=1, relativity=0, deriv=1, omega=None, verbose=None)
ExcCH2 = np.sum((naCH2[0]+nbCH2[0])*w * exc2)
exc3, vxc3, fxc3, kxc3 = dft.libxc.eval_xc('m062x', n3, spin=1, relativity=0, deriv=1, omega=None, verbose=None)
ExcCH3 = np.sum((naCH3[0]+nbCH3[0])*w * exc3)
exc4, vxc4, fxc4, kxc4 = dft.libxc.eval_xc('m062x', n4, spin=1, relativity=0, deriv=1, omega=None, verbose=None)
ExcCH4 = np.sum((naCH4[0]+nbCH4[0])*w * exc4)
Efxc = ExcCH1 + ExcCH2 + ExcCH3 + ExcCH4
Excnad = Exctot - Efxc
print(f"M_sr={M_sr:.2f} Total XC energy(SR)={Exctot_sr:.5f} Total XC energy={Exctot:.5f} Fragment XC energy={Efxc:.5f} Non-additive XC energy={Excnad:.5f} SR-difference={(Exctot_sr-Exctot)*627.51:.5f}kcal/mol.")