import numpy as np
from pyscf import gto, scf, dft, lib
from pyscf.lib import chkfile

lib.num_threads = 32
basis  = 'ccpvqz'

c4h4_1 = gto.M(atom =f"""
C         -0.66970        0.00180       -0.78430
C         -0.66890       -0.00670        0.78500
C          0.66970       -0.00180        0.78430
C          0.66890        0.00670       -0.78500
H         -1.44600        0.00310       -1.54560
H         -1.44450       -0.01370        1.54690
H          1.44600       -0.00310        1.54560
H          1.44450        0.01370       -1.54690 """, unit='ANG', basis=basis)

c4h4_2 = gto.M(atom =f"""
C         -0.67400        0.00180       -0.77570
C         -0.67320       -0.00670        0.77630
C          0.67400       -0.00180        0.77570
C          0.67320        0.00670       -0.77630
H         -1.45330        0.00300       -1.53370
H         -1.45180       -0.01370        1.53500
H          1.45330       -0.00300        1.53370
H          1.45180        0.01370       -1.53500 """, unit='ANG', basis=basis)

c4h4_3 = gto.M(atom =f"""
C         -0.68070        0.00170       -0.76780
C         -0.67990       -0.00670        0.76840
C          0.68070       -0.00170        0.76780
C          0.67990        0.00670       -0.76840
H         -1.45840        0.00300       -1.52700
H         -1.45690       -0.01370        1.52830
H          1.45840       -0.00300        1.52700
H          1.45690        0.01370       -1.52830 """, unit='ANG', basis=basis)

c4h4_4 = gto.M(atom =f"""
C         -0.68740        0.00160       -0.76000
C         -0.68670       -0.00670        0.76060
C          0.68740       -0.00160        0.76000
C          0.68670        0.00670       -0.76060
H         -1.46360        0.00290       -1.52050
H         -1.46210       -0.01360        1.52180
H          1.46360       -0.00290        1.52050
H          1.46210        0.01360       -1.52180 """, unit='ANG', basis=basis)

c4h4_5 = gto.M(atom =f"""
C         -0.69430        0.00160       -0.75220
C         -0.69360       -0.00660        0.75280
C          0.69430       -0.00160        0.75220
C          0.69360        0.00660       -0.75280
H         -1.46890        0.00290       -1.51400
H         -1.46740       -0.01360        1.51540
H          1.46890       -0.00290        1.51400
H          1.46740        0.01360       -1.51540 """, unit='ANG', basis=basis)

c4h4_6 = gto.M(atom =f"""
C         -0.70130        0.00150       -0.74450
C         -0.70050       -0.00660        0.74520
C          0.70130       -0.00150        0.74450
C          0.70050        0.00660       -0.74520
H         -1.47430        0.00280       -1.50780
H         -1.47280       -0.01360        1.50910
H          1.47430       -0.00280        1.50780
H          1.47280        0.01360       -1.50910 """, unit='ANG', basis=basis)

c4h4_7 = gto.M(atom =f"""
C         -0.70830        0.00140       -0.73700
C         -0.70760       -0.00660        0.73760
C          0.70830       -0.00140        0.73700
C          0.70760        0.00660       -0.73760
H         -1.47980        0.00280       -1.50160
H         -1.47840       -0.01360        1.50300
H          1.47980       -0.00280        1.50160
H          1.47840        0.01360       -1.50300 """, unit='ANG', basis=basis, symmetry=False)

c4h4_8 = gto.M(atom =f"""
C         -0.71550        0.00140       -0.72950
C         -0.71480       -0.00660        0.73010
C          0.71550       -0.00140        0.72950
C          0.71480        0.00660       -0.73010
H         -1.48540        0.00270       -1.49560
H         -1.48400       -0.01360        1.49700
H          1.48540       -0.00270        1.49560
H          1.48400        0.01360       -1.49700 """, unit='ANG', basis=basis)

c4h4_9 = gto.M(atom =f"""
C         -0.72280        0.00130       -0.72210
C         -0.72210       -0.00660        0.72280
C          0.72280       -0.00130        0.72210
C          0.72210        0.00660       -0.72280
H         -1.49120        0.00270       -1.48980
H         -1.48980       -0.01360        1.49120
H          1.49120       -0.00270        1.48980
H          1.48980        0.01360       -1.49120 """, unit='ANG', basis=basis)

c4h4_1_m062x = dft.UKS(c4h4_1)
c4h4_1_m062x.xc = 'm062x'
c4h4_1_m062x.chkfile = 'c4h4_1.chk'
dma, dmb = c4h4_1_m062x.get_init_guess()
dmb[:2, :2] = 0 
dm1 = (dma, dmb)
c4h4_1_m062x.kernel(dm1)

c4h4_9_m062x = dft.UKS(c4h4_9)
c4h4_9_m062x.xc = 'm062x'
c4h4_9_m062x.chkfile = 'c4h4_9.chk'
c4h4_9_m062x.kernel()
mo9 = c4h4_9_m062x.stability()[0]
dm9 = c4h4_9_m062x.make_rdm1(mo9, c4h4_9_m062x.mo_occ)
c4h4_9_m062x.kernel(dm9)
mo9 = c4h4_9_m062x.stability()[0]
dm9 = c4h4_9_m062x.make_rdm1(mo9, c4h4_9_m062x.mo_occ)
c4h4_9_m062x.kernel(dm9)

c4h4_8_m062x = dft.UKS(c4h4_8)
c4h4_8_m062x.xc = 'm062x'
c4h4_8_m062x.chkfile = 'c4h4_8.chk'
dm8 = c4h4_8_m062x.from_chk('c4h4_9.chk')
c4h4_8_m062x.kernel(dm8)

c4h4_7_m062x = dft.UKS(c4h4_7)
c4h4_7_m062x.xc = 'm062x'
c4h4_7_m062x.chkfile = 'c4h4_7.chk'
dm7 = c4h4_7_m062x.from_chk('c4h4_8.chk')
c4h4_7_m062x.kernel(dm7)

c4h4_6_m062x = dft.UKS(c4h4_6)
c4h4_6_m062x.xc = 'm062x'
c4h4_6_m062x.chkfile = 'c4h4_6.chk'
dm6 = c4h4_6_m062x.from_chk('c4h4_7.chk')
c4h4_6_m062x.kernel(dm6)

c4h4_5_m062x = dft.UKS(c4h4_5)
c4h4_5_m062x.xc = 'm062x'
c4h4_5_m062x.chkfile = 'c4h4_5.chk'
dm5 = c4h4_5_m062x.from_chk('c4h4_6.chk')
c4h4_5_m062x.kernel(dm5)

c4h4_4_m062x = dft.UKS(c4h4_4)
c4h4_4_m062x.xc = 'm062x'
c4h4_4_m062x.chkfile = 'c4h4_4.chk'
dm4 = c4h4_4_m062x.from_chk('c4h4_5.chk')
c4h4_4_m062x.kernel(dm4)

c4h4_3_m062x = dft.UKS(c4h4_3)
c4h4_3_m062x.xc = 'm062x'
c4h4_3_m062x.chkfile = 'c4h4_3.chk'
dm3 = c4h4_3_m062x.from_chk('c4h4_4.chk')
c4h4_3_m062x.kernel(dm3)

c4h4_2_m062x = dft.UKS(c4h4_2)
c4h4_2_m062x.xc = 'm062x'
c4h4_2_m062x.chkfile = 'c4h4_2.chk'
dm2 = c4h4_2_m062x.from_chk('c4h4_3.chk')
c4h4_2_m062x.kernel(dm2)

spin_pop_1, Ms_1 = c4h4_1_m062x.mulliken_spin_pop(c4h4_1, np.array((c4h4_1_m062x.make_rdm1())))
spin_pop_2, Ms_2 = c4h4_2_m062x.mulliken_spin_pop(c4h4_2, np.array((c4h4_2_m062x.make_rdm1())))
spin_pop_3, Ms_3 = c4h4_3_m062x.mulliken_spin_pop(c4h4_3, np.array((c4h4_3_m062x.make_rdm1())))
spin_pop_4, Ms_4 = c4h4_4_m062x.mulliken_spin_pop(c4h4_4, np.array((c4h4_4_m062x.make_rdm1())))
spin_pop_5, Ms_5 = c4h4_5_m062x.mulliken_spin_pop(c4h4_5, np.array((c4h4_5_m062x.make_rdm1())))
spin_pop_6, Ms_6 = c4h4_6_m062x.mulliken_spin_pop(c4h4_6, np.array((c4h4_6_m062x.make_rdm1())))
spin_pop_7, Ms_7 = c4h4_7_m062x.mulliken_spin_pop(c4h4_7, np.array((c4h4_7_m062x.make_rdm1())))
spin_pop_8, Ms_8 = c4h4_8_m062x.mulliken_spin_pop(c4h4_8, np.array((c4h4_8_m062x.make_rdm1())))
spin_pop_9, Ms_9 = c4h4_9_m062x.mulliken_spin_pop(c4h4_9, np.array((c4h4_9_m062x.make_rdm1())))

print(f"M1={Ms_1[0]+Ms_1[4]:.6f} \nM2={Ms_2[0]+Ms_2[4]:.6f} \nM3={Ms_3[0]+Ms_3[4]:.6f} \nM4={Ms_4[0]+Ms_4[4]:.6f} \nM5={Ms_5[0]+Ms_5[4]:.6f} \nM6={Ms_6[0]+Ms_6[4]:.6f} \nM7={Ms_7[0]+Ms_7[4]:.6f} \nM8={Ms_8[0]+Ms_8[4]:.6f} \nM9={Ms_9[0]+Ms_9[4]:.6f}")

print(f"dE={(c4h4_2_m062x.e_tot-c4h4_1_m062x.e_tot)*627.51:.3f} \n{(c4h4_3_m062x.e_tot-c4h4_1_m062x.e_tot)*627.51:.3f} \n{(c4h4_4_m062x.e_tot-c4h4_1_m062x.e_tot)*627.51:.3f} \n{(c4h4_5_m062x.e_tot-c4h4_1_m062x.e_tot)*627.51:.3f} \n{(c4h4_6_m062x.e_tot-c4h4_1_m062x.e_tot)*627.51:.3f} \n{(c4h4_7_m062x.e_tot-c4h4_1_m062x.e_tot)*627.51:.3f} \n{(c4h4_8_m062x.e_tot-c4h4_1_m062x.e_tot)*627.51:.3f} \n{(c4h4_9_m062x.e_tot-c4h4_1_m062x.e_tot)*627.51:.3f}")