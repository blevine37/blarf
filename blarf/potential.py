import numpy as np
import math

class potential():
    def __init__(self):
        self.forcefield = ""
        self.numdims = 0
        self.energy = 0.0
        self.forces = np.zeros(self.get_numdims())

    def eval_pes(self,r):
        exec("self.eval_pes_" + self.forcefield + "(r)")

    def set_forcefield(self,ff):
        self.forcefield = ff

    def get_forcefield(self,ff):
        return ff

    def set_numdims(self,ndims):
        self.numdims = ndims
        self.forces = np.zeros(ndims)
    
    def get_numdims(self):
        return self.numdims

    def set_energy(self,en):
        self.energy = en

    def get_energy(self):
        return self.energy

    def set_forces(self,f):
        self.forces = f.copy()

    def get_forces(self):
        return self.forces.copy()

###########################################
# chain forcefield
###########################################

    def set_chain_length(self,n):
        self.chain_length = n
        self.set_numdims(3*n)

    def get_chain_length(self):
        return self.chain_length

    def set_chain_bond_r(self,r):
        self.chain_bond_r = r

    def get_chain_bond_r(self):
        return self.chain_bond_r

    def set_chain_bond_k(self,k):
        self.chain_bond_k =  k

    def get_chain_bond_k(self):
        return self.chain_bond_k

    def set_chain_lj_eps(self,eps):
        self.chain_lj_eps = eps

    def get_chain_lj_eps(self):
        return self.chain_lj_eps
    
    def set_chain_lj_sigma(self,sigma):
        self.chain_lj_sigma = sigma

    def get_chain_lj_sigma(self):
        return self.chain_lj_sigma
    
    def init_chain(self,n,r,k,sigma,eps):
        self.set_forcefield("chain")
        self.set_chain_length(n)
        self.set_chain_bond_r(r)
        self.set_chain_bond_k(k)
        self.set_chain_lj_sigma(sigma)
        self.set_chain_lj_eps(eps)

    def eval_pes_chain(self,r):
        e = 0.0
        f = np.zeros(self.get_numdims())

        # compute bonding terms
        for iatom in range(self.get_chain_length()-1):
            ix = 3*iatom
            iy = 3*iatom + 1
            iz = 3*iatom + 2
            jx = 3*iatom + 3
            jy = 3*iatom + 4
            jz = 3*iatom + 5
            
            x_ab = r[ix] - r[jx]
            y_ab = r[iy] - r[jy]
            z_ab = r[iz] - r[jz]
            
            r_ab = math.sqrt(x_ab*x_ab + y_ab*y_ab + z_ab*z_ab)
            disp_ab = r_ab - self.get_chain_bond_r()
            
            e += self.get_chain_bond_k() * disp_ab * disp_ab
            
            ftmp = 2.0 * self.get_chain_bond_k() * disp_ab
            
            x_ab_norm = x_ab / r_ab
            y_ab_norm = y_ab / r_ab
            z_ab_norm = z_ab / r_ab
            
            f[ix] -= x_ab_norm * ftmp
            f[iy] -= y_ab_norm * ftmp
            f[iz] -= z_ab_norm * ftmp
            f[jx] += x_ab_norm * ftmp
            f[jy] += y_ab_norm * ftmp
            f[jz] += z_ab_norm * ftmp

            # compute lj terms
            for jatom in range(iatom + 2, self.get_chain_length()):
                jx = 3*jatom 
                jy = 3*jatom + 1
                jz = 3*jatom + 2

                x_ab = r[ix] - r[jx]
                y_ab = r[iy] - r[jy]
                z_ab = r[iz] - r[jz]
                
                r_ab = math.sqrt(x_ab*x_ab + y_ab*y_ab + z_ab*z_ab)

                one_d_r = 1.0 / r_ab

                sigma_d_r = self.get_chain_lj_sigma() * one_d_r

                tmp3 = sigma_d_r * sigma_d_r * sigma_d_r
                tmp6 = tmp3 * tmp3
                tmp12 = tmp6 * tmp6

                e += 4.0 * self.get_chain_lj_eps() * (tmp12 - tmp6)
                
                ftmp = 4.0 * self.get_chain_lj_eps()
                ftmp *= 6.0 * tmp6 * one_d_r - 12.0 * tmp12 * one_d_r
                
                x_ab_norm = x_ab / r_ab
                y_ab_norm = y_ab / r_ab
                z_ab_norm = z_ab / r_ab
                
                f[ix] -= x_ab_norm * ftmp
                f[iy] -= y_ab_norm * ftmp
                f[iz] -= z_ab_norm * ftmp
                f[jx] += x_ab_norm * ftmp
                f[jy] += y_ab_norm * ftmp
                f[jz] += z_ab_norm * ftmp

        self.set_energy(e)
        self.set_forces(f)
        
#################################
# ethylene-like forcefield
#################################

    def set_ethylene_bond_k(self,k):
        self.ethylene_bond_k = k

    def get_ethylene_bond_k(self):
        return self.ethylene_bond_k
    
    def set_ethylene_angle_k(self,k):
        self.ethylene_angle_k = k

    def get_ethylene_angle_k(self):
        return self.ethylene_angle_k
    
    def set_ethylene_e_twist(self,e):
        self.ethylene_e_twist = e

    def get_ethylene_e_twist(self):
        return self.ethylene_e_twist
    
    def init_ethylene(self,k_bond,k_angle,e_twist):
        self.set_numdims(18)
        self.set_forcefield("ethylene")
        self.set_ethylene_bond_k(k_bond)
        self.set_ethylene_angle_k(k_angle)
        self.set_ethylene_e_twist(e_twist)

    def eval_pes_ethylene(self,r):
        e = 0.0
        f = np.zeros(self.get_numdims())

        iatom = [ 0, 0, 0, 1, 1]
        jatom = [ 1, 2, 3, 4, 5]
        
        for ibond in range(5):
            # compute bonding terms
            ix = 3*iatom[ibond]
            iy = 3*iatom[ibond] + 1
            iz = 3*iatom[ibond] + 2
            jx = 3*jatom[ibond]
            jy = 3*jatom[ibond] + 1
            jz = 3*jatom[ibond] + 2
            
            x_ab = r[ix] - r[jx]
            y_ab = r[iy] - r[jy]
            z_ab = r[iz] - r[jz]
            
            r_ab = math.sqrt(x_ab*x_ab + y_ab*y_ab + z_ab*z_ab)
            disp_ab = r_ab - 1.0
            
            e += self.get_ethylene_bond_k() * disp_ab * disp_ab
            
            ftmp = 2.0 * self.get_ethylene_bond_k() * disp_ab
        
            x_ab_norm = x_ab / r_ab
            y_ab_norm = y_ab / r_ab
            z_ab_norm = z_ab / r_ab
            
            f[ix] -= x_ab_norm * ftmp
            f[iy] -= y_ab_norm * ftmp
            f[iz] -= z_ab_norm * ftmp
            f[jx] += x_ab_norm * ftmp
            f[jy] += y_ab_norm * ftmp
            f[jz] += z_ab_norm * ftmp
            
            self.set_energy(e)
            self.set_forces(f)

        iatom = [ 0, 0, 1, 1, 2, 4]
        jatom = [ 1, 1, 0, 0, 0, 1]
        katom = [ 4, 5, 2, 3, 3, 5]
        
        for ibond in range(5):
            # compute bonding terms
            ix = 3*iatom[ibond]
            iy = 3*iatom[ibond] + 1
            iz = 3*iatom[ibond] + 2
            jx = 3*jatom[ibond]
            jy = 3*jatom[ibond] + 1
            jz = 3*jatom[ibond] + 2
            kx = 3*katom[ibond]
            ky = 3*katom[ibond] + 1
            kz = 3*katom[ibond] + 2

            x_ji = r[jx] - r[ix]
            y_ji = r[jy] - r[iy]
            z_ji = r[jz] - r[iz]
            r_ji = math.sqrt(x_ji*x_ji + y_ji*y_ji + z_ji*z_ji)

            x_jk = r[jx] - r[kx]
            y_jk = r[jy] - r[ky]
            z_jk = r[jz] - r[kz]
            r_jk = math.sqrt(x_jk*x_jk + y_jk*y_jk + z_jk*z_jk)

            x_ji_s = x_ji / r_ji
            y_ji_s = y_ji / r_ji
            z_ji_s = z_ji / r_ji
            
            x_jk_s = x_jk / r_jk
            y_jk_s = y_jk / r_jk
            z_jk_s = z_jk / r_jk

            dot = x_ji_s * x_jk_s + y_ji_s * y_jk_s + z_ji_s * z_jk_s
            
            theta = 180 * math.acos(dot) / 3.14159265359

            disp_theta = theta - 120.0

            e += self.get_ethylene_angle_k() * disp_theta * disp_theta

            dedtheta = 2.0 * self.get_ethylene_angle_k() * disp_theta

            dthetaddot = -1.0 / math.sqrt(1.0 - dot * dot)

            ddotdx_ji_s = x_jk
            ddotdy_ji_s = y_jk
            ddotdz_ji_s = z_jk
            ddotdx_jk_s = x_ji
            ddotdy_jk_s = y_ji
            ddotdz_jk_s = z_ji

            dr_jidx_ji = x_ji / r_ji
            dr_jidy_ji = y_ji / r_ji
            dr_jidz_ji = z_ji / r_ji
            dr_jkdx_jk = x_jk / r_jk
            dr_jkdy_jk = y_jk / r_jk
            dr_jkdz_jk = z_jk / r_jk

            dx_ji_sdx_ji = ( x_ji * dr_jidx_ji - r_ji ) / (r_ji * r_ji)
            dy_ji_sdy_ji = ( y_ji * dr_jidy_ji - r_ji ) / (r_ji * r_ji)
            dz_ji_sdz_ji = ( z_ji * dr_jidz_ji - r_ji ) / (r_ji * r_ji)
            dx_jk_sdx_jk = ( x_jk * dr_jkdx_jk - r_jk ) / (r_jk * r_jk)
            dy_jk_sdy_jk = ( y_jk * dr_jkdy_jk - r_jk ) / (r_jk * r_jk)
            dz_jk_sdz_jk = ( z_jk * dr_jkdz_jk - r_jk ) / (r_jk * r_jk)

            dedx_ji = dedtheta * dthetaddot * ddotdx_ji_s * dx_ji_sdx_ji
            dedy_ji = dedtheta * dthetaddot * ddotdy_ji_s * dy_ji_sdy_ji
            dedz_ji = dedtheta * dthetaddot * ddotdz_ji_s * dz_ji_sdz_ji
            dedx_jk = dedtheta * dthetaddot * ddotdx_jk_s * dx_jk_sdx_jk
            dedy_jk = dedtheta * dthetaddot * ddotdy_jk_s * dy_jk_sdy_jk
            dedz_jk = dedtheta * dthetaddot * ddotdz_jk_s * dz_jk_sdz_jk

            f[ix] -= dedx_ji
            f[iy] -= dedy_ji
            f[iz] -= dedz_ji
            f[kx] -= dedx_jk
            f[ky] -= dedy_jk
            f[kz] -= dedz_jk
            f[jx] += dedx_ji + dedx_jk
            f[jy] += dedy_ji + dedy_jk
            f[jz] += dedz_ji + dedz_jk
            
        iatom = [ 2, 3, 2, 3 ]
        jatom = [ 0, 0, 0, 0 ]
        katom = [ 1, 1, 1, 1 ]
        latom = [ 4, 4, 5, 5 ]
        
        for ibond in range(4):
            # compute bonding terms
            ix = 3*iatom[ibond]
            iy = 3*iatom[ibond] + 1
            iz = 3*iatom[ibond] + 2
            jx = 3*jatom[ibond]
            jy = 3*jatom[ibond] + 1
            jz = 3*jatom[ibond] + 2
            kx = 3*katom[ibond]
            ky = 3*katom[ibond] + 1
            kz = 3*katom[ibond] + 2
            lx = 3*latom[ibond]
            ly = 3*latom[ibond] + 1
            lz = 3*latom[ibond] + 2

            x_ji = r[jx] - r[ix]
            y_ji = r[jy] - r[iy]
            z_ji = r[jz] - r[iz]
            r_ji = math.sqrt(x_ji*x_ji + y_ji*y_ji + z_ji*z_ji)

            x_jk = r[jx] - r[kx]
            y_jk = r[jy] - r[ky]
            z_jk = r[jz] - r[kz]
            r_jk = math.sqrt(x_jk*x_jk + y_jk*y_jk + z_jk*z_jk)

            x_kl = r[kx] - r[lx]
            y_kl = r[ky] - r[ly]
            z_kl = r[kz] - r[lz]
            r_kl = math.sqrt(x_kl*x_kl + y_kl*y_kl + z_kl*z_kl)

            xcross_jijk = y_ji * z_jk - z_ji * y_jk 
            ycross_jijk = z_ji * x_jk - x_ji * z_jk
            zcross_jijk = x_ji * y_jk - y_ji * x_jk
            rcross_jijk = math.sqrt(xcross_jijk * xcross_jijk + ycross_jijk * ycross_jijk + zcross_jijk * zcross_jijk)
            xcross_jijk_s = xcross_jijk / rcross_jijk
            ycross_jijk_s = ycross_jijk / rcross_jijk
            zcross_jijk_s = zcross_jijk / rcross_jijk

            xcross_kjkl = -1.0 * y_jk * z_kl
            ycross_kjkl = -1.0 * z_jk * x_kl
            zcross_kjkl = -1.0 * x_jk * y_kl
            rcross_kjkl = math.sqrt(xcross_kjkl * xcross_kjkl + ycross_kjkl * ycross_kjkl + zcross_kjkl * zcross_kjkl)
            xcross_kjkl_s = xcross_kjkl / rcross_kjkl
            ycross_kjkl_s = ycross_kjkl / rcross_kjkl
            zcross_kjkl_s = zcross_kjkl / rcross_kjkl

            dot = xcross_jijk_s * xcross_kjkl_s + ycross_jijk_s * ycross_kjkl_s + zcross_jijk_s * zcross_kjkl_s

            e += -0.25 * self.get_ethylene_e_twist() * dot * dot

            deddot = -0.5 * self.get_ethylene_e_twist() * dot

            ddotdxc_jijk_s = xcross_kjkl_s
            ddotdyc_jijk_s = ycross_kjkl_s
            ddotdzc_jijk_s = zcross_kjkl_s
            ddotdxc_kjkl_s = xcross_jijk_s
            ddotdyc_kjkl_s = ycross_jijk_s
            ddotdzc_kjkl_s = zcross_jijk_s

            drc_jijkdxc_jijk = xcross_jijk / rcross_jijk
            drc_jijkdyc_jijk = ycross_jijk / rcross_jijk
            drc_jijkdzc_jijk = zcross_jijk / rcross_jijk
            drc_kjkldxc_kjkl = xcross_kjkl / rcross_kjkl
            drc_kjkldyc_kjkl = ycross_kjkl / rcross_kjkl
            drc_kjkldzc_kjkl = zcross_kjkl / rcross_kjkl
            
            dxc_jijk_sdxc_jijk = xcross_jijk * drc_jijkdxc_jijk - rcross_jijk
            dyc_jijk_sdyc_jijk = ycross_jijk * drc_jijkdyc_jijk - rcross_jijk
            dzc_jijk_sdzc_jijk = zcross_jijk * drc_jijkdzc_jijk - rcross_jijk
            dxc_kjkl_sdxc_kjkl = xcross_kjkl * drc_kjkldxc_kjkl - rcross_kjkl
            dyc_kjkl_sdyc_kjkl = ycross_kjkl * drc_kjkldyc_kjkl - rcross_kjkl
            dzc_kjkl_sdzc_kjkl = zcross_kjkl * drc_kjkldzc_kjkl - rcross_kjkl

            dxc_jijkdy_ji = z_jk
            dxc_jijkdz_jk = y_ji
            dyc_jijkdz_ji = x_jk
            dyc_jijkdx_jk = z_ji
            dzc_jijkdx_ji = y_jk
            dzc_jijkdy_jk = x_ji

            dxc_jijkdz_ji = -1.0 * y_jk
            dxc_jijkdy_jk = -1.0 * z_ji
            dyc_jijkdx_ji = -1.0 * z_jk
            dyc_jijkdz_jk = -1.0 * x_ji
            dzc_jijkdy_ji = -1.0 * x_jk
            dzc_jijkdx_jk = -1.0 * y_ji

            dxc_kjkldy_jk = -1.0 * z_kl
            dxc_kjkldz_kl = -1.0 * y_jk
            dyc_kjkldz_jk = -1.0 * x_kl
            dyc_kjkldx_kl = -1.0 * z_jk
            dzc_kjkldx_jk = -1.0 * y_kl
            dzc_kjkldy_kl = -1.0 * x_jk

            dxc_kjkldz_jk = y_kl
            dxc_kjkldy_kl = z_jk
            dyc_kjkldx_jk = z_kl
            dyc_kjkldz_kl = x_jk
            dzc_kjkldy_jk = x_kl
            dzc_kjkldx_kl = y_jk

            dedx_ji  = deddot * ddotdzc_jijk_s * dzc_jijk_sdzc_jijk * dzc_jijkdx_ji
            dedx_ji += deddot * ddotdyc_jijk_s * dyc_jijk_sdyc_jijk * dyc_jijkdx_ji
            dedx_jk  = deddot * ddotdyc_jijk_s * dyc_jijk_sdyc_jijk * dyc_jijkdx_jk
            dedx_jk += deddot * ddotdzc_jijk_s * dzc_jijk_sdzc_jijk * dzc_jijkdx_jk
            dedy_ji  = deddot * ddotdxc_jijk_s * dxc_jijk_sdxc_jijk * dxc_jijkdy_ji
            dedy_ji += deddot * ddotdzc_jijk_s * dzc_jijk_sdzc_jijk * dzc_jijkdy_ji
            dedy_jk  = deddot * ddotdzc_jijk_s * dzc_jijk_sdzc_jijk * dzc_jijkdy_jk
            dedy_jk += deddot * ddotdxc_jijk_s * dxc_jijk_sdxc_jijk * dxc_jijkdy_jk
            dedz_ji  = deddot * ddotdyc_jijk_s * dyc_jijk_sdyc_jijk * dyc_jijkdz_ji
            dedz_ji += deddot * ddotdxc_jijk_s * dxc_jijk_sdxc_jijk * dxc_jijkdz_ji
            dedz_jk  = deddot * ddotdxc_jijk_s * dxc_jijk_sdxc_jijk * dxc_jijkdz_jk
            dedz_jk += deddot * ddotdyc_jijk_s * dyc_jijk_sdyc_jijk * dyc_jijkdz_jk

            dedx_jk += deddot * ddotdzc_kjkl_s * dzc_kjkl_sdzc_kjkl * dzc_kjkldx_jk
            dedx_jk += deddot * ddotdyc_kjkl_s * dyc_kjkl_sdyc_kjkl * dyc_kjkldx_jk
            dedx_kl  = deddot * ddotdyc_kjkl_s * dyc_kjkl_sdyc_kjkl * dyc_kjkldx_kl
            dedx_kl += deddot * ddotdzc_kjkl_s * dzc_kjkl_sdzc_kjkl * dzc_kjkldx_kl
            dedy_jk += deddot * ddotdxc_kjkl_s * dxc_kjkl_sdxc_kjkl * dxc_kjkldy_jk
            dedy_jk += deddot * ddotdzc_kjkl_s * dzc_kjkl_sdzc_kjkl * dzc_kjkldy_jk
            dedy_kl  = deddot * ddotdzc_kjkl_s * dzc_kjkl_sdzc_kjkl * dzc_kjkldy_kl
            dedy_kl += deddot * ddotdxc_kjkl_s * dxc_kjkl_sdxc_kjkl * dxc_kjkldy_kl
            dedz_jk += deddot * ddotdyc_kjkl_s * dyc_kjkl_sdyc_kjkl * dyc_kjkldz_jk
            dedz_jk += deddot * ddotdxc_kjkl_s * dxc_kjkl_sdxc_kjkl * dxc_kjkldz_jk
            dedz_kl  = deddot * ddotdxc_kjkl_s * dxc_kjkl_sdxc_kjkl * dxc_kjkldz_kl
            dedz_kl += deddot * ddotdyc_kjkl_s * dyc_kjkl_sdyc_kjkl * dyc_kjkldz_kl

            f[ix] += dedx_ji
            f[iy] += dedy_ji
            f[iz] += dedz_ji
            f[jx] -= dedx_ji + dedx_jk
            f[jy] -= dedy_ji + dedy_jk
            f[jz] -= dedz_ji + dedz_jk
            f[kx] += dedx_jk - dedx_kl
            f[ky] += dedy_jk - dedy_kl
            f[kz] += dedz_jk - dedz_kl
            f[lx] += dedx_kl
            f[ly] += dedy_kl
            f[lz] += dedz_kl
            
            
