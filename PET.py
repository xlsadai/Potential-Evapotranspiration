# coding=utf-8
import numpy as np
import os
import pandas as pd


class refPET():
    def __init__(self, j, z, phi, Tmax, Tmin, Tmean, RH, n, u10, Ta=None, Tb=None, P=None):
        """
        基于Penman-Monteith公式计算潜在蒸散发PET
        :param j: 日序，取值范围为1到365或366，1月1日取日序为1
        :param z: 站点海拔(m)
        :param phi: 纬度(rad)
        :param Tmax: 日最高温度(℃)
        :param Tmin: 日最低温度(℃)
        :param Tmean: 日平均温度(℃)
        :param RH: 相对湿度
        :param n: 实际日照时数SSD (h)
        :param u10: 10米高处风速(m/s)
        :param Ta: 前一日平均温度(℃)
        :param Tb: 后一日平均温度(℃)
        """
        self.j = j
        self.z = z
        self.phi = phi
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.Tmean = Tmean
        self.RH = RH
        self.P = P
        self.n = n
        self.u10 = u10
        self.Ta = Ta if Ta is not None else Tmean
        self.Tb = Tb if Tb is not None else Tmean

        """
        系数库
        a_s: 太阳辐射截距
        b_s: 太阳辐射系数
        gsc: 太阳常数
        alpha: 参考作物反照率(一般选用绿色草地0.23)
        sigma: 斯蒂芬·玻尔兹曼常数(MK·K^-4·m^-2·d^-1)
        """
        self.a_s = 0.25
        self.b_s = 0.50
        self.gsc = 0.0820
        self.alpha = 0.23
        self.sigma = 4.903e-9

        """
        中间变量
        alphaP: Δ, 饱和水汽压曲线斜率(kPa/℃)
        Rn: 地表净辐射(MJ/(m·d))
        G: 土壤热通量(MJ/(m^2·d))
        gama: γ, 干湿表常数(kPa/℃)
        u2: 2米高处风速(m/s)
        es: 饱和水汽压(kPa)
        ea: 实际水汽压(kPa)
        """
        self.alphaP = None
        self.Rn = None
        self.G = None
        self.gama = None
        self.u2 = None
        self.es = None
        self.ea = None

        """单站点 日潜在蒸散发 PET"""
        self.PET = None

    def set_as(self, a_s):
        self.a_s = a_s

    def set_bs(self, b_s):
        self.b_s = b_s

    def set_gsc(self, gsc):
        self.gsc = gsc

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_sigma(self, sigma):
        self.sigma = sigma

    def calculate_PET(self):
        """计算潜在蒸散发PET(mm/d)"""
        self.PET = (0.408 * self.alphaP * (self.Rn - self.G) + self.gama * 900 / (self.Tmean + 273) * self.u2 * (
                self.es - self.ea)) / (self.alphaP + self.gama * (1 + 0.34 * self.u2))
        return self.PET

    def calculate_alphaP(self):
        """饱和水汽压曲线斜率alphaP (kPa/℃)"""
        self.alphaP = 4098 * (0.6108 * np.exp(17.27 * self.Tmean / (self.Tmean + 237.3))) / (self.Tmean + 237.3) ** 2
        return self.alphaP

    def calculate_ea(self):
        """计算饱和水汽压es (kPa)"""

        def fe(T):
            e = 0.6108 * np.exp(17.27 * T / (T + 237.3))
            return e

        self.es = (fe(self.Tmax) + fe(self.Tmin)) / 2
        """计算实际水汽压ea (kPa)"""
        self.ea = self.RH * self.es
        return self.ea

    def calculate_G(self):
        """计算土壤热通量G"""
        self.G = 0.07 * (self.Tb - self.Ta)
        return self.G

    def calculate_Rn(self):
        """计算太阳磁偏角delta"""
        delta = 0.408 * np.sin(2 * np.pi / 365 * self.j - 1.39)
        """计算日出时角ws"""
        ws = np.arccos(-np.tan(self.phi) * np.tan(delta))
        """计算日地平均距离dr"""
        dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * self.j)
        """计算日地球外辐射"""
        Ra = 24 * 60 / np.pi * self.gsc * dr * (
                ws * np.sin(self.phi) * np.sin(delta) + np.cos(self.phi) * np.cos(delta) * np.sin(ws))
        """计算最大可能日照时数N"""
        n0 = 24 / np.pi * ws
        """计算太阳辐射Rs"""
        Rs = (self.a_s + self.b_s * self.n / n0) * Ra
        """计算短波辐射Rns"""
        Rns = (1 - self.alpha) * Rs
        """计算Rso"""
        Rso = (0.75 + 2e-5 * self.z) * Ra
        """计算支出的净长波辐射Rnl"""
        if self.ea is None:
            self.calculate_ea()
        Rnl = self.sigma * ((self.Tmax + 272.15) ** 4 + (self.Tmin + 272.15) ** 4) / 2 * (
                0.34 - 0.14 * (self.ea ** 0.5)) * (1.35 * Rs / Rso - 0.35)

        """计算净辐射Rn"""
        self.Rn = Rns - Rnl
        if self.Rn < 0:
            print(self.Rn)
        return self.Rn

    def calculate_gama(self):
        """计算平均气压"""
        if self.P is None:
            self.P = 101.3 * ((293 - 0.0065 * self.z) / 293) ** 5.26
        """计算干湿表常数γ"""
        self.gama = 0.665e-3 * self.P
        return self.gama

    def calculate_u2(self):
        """计算2米处风速(m/s)"""
        self.u2 = self.u10 * 4.87 / np.log(67.8 * 10 - 5.42)
        return self.u2

    def refPET_main(self):
        """计算中间变量"""
        self.calculate_alphaP()
        self.calculate_G()
        self.calculate_ea()
        self.calculate_Rn()
        self.calculate_u2()
        self.calculate_gama()

        """计算潜在蒸散发"""
        self.calculate_PET()
        if self.PET < 0:
            print(self.PET)
        return self.PET


def get_j(year, month, day):
    """
    计算日序，即第1到365或366天，1月1日取日序为1
    :param year: 年份
    :param month: 月份
    :param day: 日
    :return:
    """

    def isleapyear(year):
        """判断是否为闰年"""
        if (year % 100 != 0) & (year % 4 == 0):
            return True
        elif year % 400 == 0:
            return True
        else:
            return False

    def day_num_in_month(year, month):
        """返回每月天数"""
        if month == 2:
            if isleapyear(year):
                return 29
            else:
                return 28
        elif (month == 1) or (month == 3) or (month == 5) or (month == 7) or (month == 8) or (
                month == 10) or (month == 12):
            return 31
        elif (month == 4) or (month == 6) or (month == 9) or (month == 11):
            return 30

    j = 0
    if month != 1:
        for i in range(1, month):
            j += day_num_in_month(year, i)
    j += day
    return j


if __name__ == "__main__":
    j = "日序"
    z = "海拔(m)"
    phi = "纬度(rad)"
    Tmax = "日最高气温(℃)"
    Tmin = "日最低气温(℃)"
    Tmean = "日平均气温(℃)"
    RH = "相对湿度"
    n = "日照时数(h)"
    u10 = "10米高处风速(m/s)"
    Ta = "前一日平均气温(℃)"
    Tb = "后一日平均气温(℃)"
    P = "平均气压(kPa)"
    obj = refPET(j, z, phi, Tmax, Tmin, Tmean, RH, n, u10, Ta, Tb, P)
    PET = obj.refPET_main()
