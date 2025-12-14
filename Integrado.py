"""
SISTEMA INTEGRADO GAIA DR3 + ANALISADOR GRAVITACIONAL 3D + CALCULADOR DE COLUNAS
Interface com abas para consulta Gaia, cálculo de propriedades físicas e análise de aglomerados
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
from datetime import datetime
import warnings
from scipy.spatial import KDTree
import networkx as nx
import plotly.graph_objects as go
import requests
from io import StringIO
import urllib.parse
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES FÍSICAS
# ============================================================

G = 6.67430e-11  # Constante gravitacional (m³/kg/s²)
PC_TO_M = 3.086e16  # 1 parsec em metros
MSUN_TO_KG = 1.98847e30  # 1 massa solar em kg


# ============================================================
# CLASSE CALCULADOR DE COLUNAS (do script dados.txt)
# ============================================================

class CalculadorColunas:
    """Classe principal para cálculo de colunas"""

    def __init__(self, df):
        self.df = df.copy()
        self.constantes = {
            'G': 6.67430e-11,  # m³ kg⁻¹ s⁻²
            'M_sun': 1.989e30,  # kg
            'L_sun': 3.828e26,  # W
            'R_sun': 6.957e8,  # m
            'pc_to_m': 3.08567758128e16,
            'km_to_pc_year': 0.001022,  # km/s to pc/year
            'year_to_sec': 365.25 * 24 * 3600
        }

    def calcular_distancia(self, snr_minimo=5.0, aplicar_lutz_kelker=True):
        """
        Calcula distância a partir da paralaxe com tratamento de erros

        Retorna:
        - distance_pc: Distância em parsecs
        - distance_error_pc: Erro na distância
        - snr_parallax: Razão sinal-ruído da paralaxe
        """
        self.df = self.df.copy()

        # Verificar se temos dados de paralaxe
        if 'parallax' not in self.df.columns:
            raise ValueError("Coluna 'parallax' não encontrada")

        # 1. Filtrar valores válidos
        mask_valido = (
                (self.df['parallax'].notna()) &
                (self.df['parallax'] > 0)
        )

        if 'parallax_error' in self.df.columns:
            mask_valido = mask_valido & (
                    (self.df['parallax_error'].notna()) &
                    (self.df['parallax_error'] > 0)
            )

        # 2. Calcular distância básica (1/parallax)
        self.df.loc[mask_valido, 'distance_pc'] = 1000.0 / self.df.loc[mask_valido, 'parallax']

        # 3. Calcular erro se disponível
        if 'parallax_error' in self.df.columns:
            # Propagação de erros: σ_d = (1000/ϖ²) * σ_ϖ
            self.df.loc[mask_valido, 'distance_error_pc'] = (
                                                                    1000.0 / (self.df.loc[mask_valido, 'parallax'] ** 2)
                                                            ) * self.df.loc[mask_valido, 'parallax_error']

        # 4. Calcular SNR da paralaxe
        if 'parallax_error' in self.df.columns:
            self.df.loc[mask_valido, 'snr_parallax'] = (
                    self.df.loc[mask_valido, 'parallax'] /
                    self.df.loc[mask_valido, 'parallax_error']
            )

        # 5. Aplicar correção de Lutz-Kelker (opcional)
        if aplicar_lutz_kelker and 'parallax_error' in self.df.columns:
            mask_baixo_snr = (
                    mask_valido &
                    (self.df['snr_parallax'] < 10) &
                    (self.df['snr_parallax'] > 0)
            )

            if mask_baixo_snr.any():
                # Correção: d_corrigido = d * (1 + (σ_ϖ/ϖ)²)
                self.df.loc[mask_baixo_snr, 'distance_pc_corrected'] = (
                        self.df.loc[mask_baixo_snr, 'distance_pc'] *
                        (1 + (self.df.loc[mask_baixo_snr, 'parallax_error'] /
                              self.df.loc[mask_baixo_snr, 'parallax']) ** 2)
                )

                # Usar valor corrigido para baixo SNR
                self.df.loc[mask_baixo_snr, 'distance_pc'] = self.df.loc[mask_baixo_snr, 'distance_pc_corrected']
                self.df = self.df.drop(columns=['distance_pc_corrected'])

        # 6. Aplicar filtro de qualidade baseado no SNR
        if 'snr_parallax' in self.df.columns:
            mask_alta_qualidade = (
                    mask_valido &
                    (self.df['snr_parallax'] >= snr_minimo)
            )

            # Marcar estrelas de baixa qualidade
            self.df['parallax_quality'] = 'HIGH'
            self.df.loc[~mask_alta_qualidade & mask_valido, 'parallax_quality'] = 'LOW'
            self.df.loc[~mask_valido, 'parallax_quality'] = 'INVALID'

        # 7. Log de estatísticas
        n_total = len(self.df)
        n_valid = mask_valido.sum()

        if 'snr_parallax' in self.df.columns:
            n_high_quality = (self.df['snr_parallax'] >= snr_minimo).sum()
            print(
                f"Distância calculada: {n_valid}/{n_total} válidas, {n_high_quality} alta qualidade (SNR ≥ {snr_minimo})")
        else:
            print(f"Distância calculada: {n_valid}/{n_total} válidas")

        return self.df

    def calcular_massa(self, metodo="Luminosidade"):
        """
        Calcula massa estelar usando diferentes métodos

        Métodos disponíveis:
        1. "Luminosidade": M ∝ L^(1/3.5) (requer luminosidade_Lsun)
        2. "Teff-Logg": Usa temperatura efetiva e gravidade superficial
        3. "Cor-Magnitude": Usa relação cor-magnitude empírica
        4. "Padrão": Valor fixo de 1.0 M☉
        """
        self.df = self.df.copy()

        # Inicializar coluna de massa
        self.df['massa_estimada_Msun'] = np.nan
        metodo_usado = []

        # Método 1: Via luminosidade (M ∝ L^(1/3.5))
        if metodo == "Luminosidade" and 'luminosidade_Lsun' in self.df.columns:
            mask = (
                    (self.df['luminosidade_Lsun'].notna()) &
                    (self.df['luminosidade_Lsun'] > 0)
            )
            if mask.any():
                # Relação massa-luminosidade: L ∝ M^3.5
                self.df.loc[mask, 'massa_estimada_Msun'] = (
                        self.df.loc[mask, 'luminosidade_Lsun'] ** (1 / 3.5)
                )
                metodo_usado.append(f"Luminosidade: {mask.sum()} estrelas")

        # Método 2: Via Teff e logg (mais preciso)
        elif metodo == "Teff-Logg":
            if ('teff_gspphot' in self.df.columns and
                    'logg_gspphot' in self.df.columns):

                mask = (
                        (self.df['teff_gspphot'].notna()) &
                        (self.df['teff_gspphot'] > 0) &
                        (self.df['logg_gspphot'].notna())
                )

                if mask.any():
                    # Calcular raio relativo ao Sol
                    # Primeiro precisamos da luminosidade
                    if 'luminosidade_Lsun' not in self.df.columns:
                        # Estimar luminosidade se não disponível
                        self.df = self.calcular_luminosidade()

                    # R/R☉ = sqrt(L/L☉) / (T/T☉)^2
                    T_sun = 5772  # K
                    self.df.loc[mask, 'raio_estimado_Rsun'] = np.sqrt(
                        self.df.loc[mask, 'luminosidade_Lsun']
                    ) / (self.df.loc[mask, 'teff_gspphot'] / T_sun) ** 2

                    # M/M☉ = (g/g☉) * (R/R☉)^2
                    # g☉ = 274 m/s² = 10^(logg☉) onde logg☉ = 4.44
                    g_sun = 10 ** 4.44  # cm/s² em unidades logg

                    self.df.loc[mask, 'massa_estimada_Msun'] = (
                                                                       (10 ** self.df.loc[mask, 'logg_gspphot']) / g_sun
                                                               ) * (self.df.loc[mask, 'raio_estimado_Rsun'] ** 2)

                    metodo_usado.append(f"Teff-Logg: {mask.sum()} estrelas")

        # Método 3: Via relação cor-magnitude empírica
        elif metodo == "Cor-Magnitude":
            if ('phot_g_mean_mag' in self.df.columns and
                    'distance_pc' in self.df.columns):

                mask = (
                        (self.df['phot_g_mean_mag'].notna()) &
                        (self.df['distance_pc'].notna()) &
                        (self.df['distance_pc'] > 0)
                )

                if mask.any():
                    # Calcular magnitude absoluta
                    M_g = self.df.loc[mask, 'phot_g_mean_mag'] - 5 * np.log10(
                        self.df.loc[mask, 'distance_pc']
                    ) + 5

                    # Converter para luminosidade
                    # M_bol☉ = 4.74, M_G☉ ≈ 4.67 (para filtro G do Gaia)
                    M_G_sun = 4.67
                    L_Lsun = 10 ** (0.4 * (M_G_sun - M_g))

                    # Massa a partir da luminosidade
                    self.df.loc[mask, 'massa_estimada_Msun'] = L_Lsun ** (1 / 3.5)

                    metodo_usado.append(f"Cor-Magnitude: {mask.sum()} estrelas")

        # Método 4: Valor padrão
        if metodo == "Padrão" or self.df['massa_estimada_Msun'].isna().all():
            mask = self.df['massa_estimada_Msun'].isna()
            if mask.any():
                self.df.loc[mask, 'massa_estimada_Msun'] = 1.0
                metodo_usado.append(f"Padrão: {mask.sum()} estrelas")

        # 3. Aplicar limites físicos
        # Massas estelares típicas: 0.08 M☉ (anã marrom) a 100 M☉ (estrelas massivas)
        self.df['massa_estimada_Msun'] = self.df['massa_estimada_Msun'].clip(0.08, 100.0)

        # 4. Preencher valores NaN restantes
        nan_count = self.df['massa_estimada_Msun'].isna().sum()
        if nan_count > 0:
            self.df['massa_estimada_Msun'] = self.df['massa_estimada_Msun'].fillna(1.0)
            metodo_usado.append(f"Preenchido: {nan_count} NaN")

        # 5. Log
        if metodo_usado:
            print(f"Massa calculada via: {', '.join(metodo_usado)}")
            print(f"Range de massas: {self.df['massa_estimada_Msun'].min():.3f} - "
                  f"{self.df['massa_estimada_Msun'].max():.3f} M☉")

        return self.df

    def calcular_luminosidade(self):
        """
        Calcula luminosidade estelar a partir da magnitude e distância

        Fórmula: L/L☉ = 10^(0.4 * (M_bol☉ - M))
        Para Gaia G band: M_G☉ ≈ 4.67
        """
        self.df = self.df.copy()

        # Verificar requisitos mínimos
        if 'phot_g_mean_mag' not in self.df.columns:
            print("Aviso: Coluna 'phot_g_mean_mag' não encontrada")
            return self.df

        if 'distance_pc' not in self.df.columns:
            print("Aviso: Calculando distância primeiro...")
            self.df = self.calcular_distancia()

        # Filtrar valores válidos
        mask = (
                (self.df['phot_g_mean_mag'].notna()) &
                (self.df['distance_pc'].notna()) &
                (self.df['distance_pc'] > 0)
        )

        if not mask.any():
            print("Aviso: Não há dados válidos para cálculo de luminosidade")
            self.df['luminosidade_Lsun'] = np.nan
            return self.df

        # 1. Calcular magnitude absoluta G
        M_G = self.df.loc[mask, 'phot_g_mean_mag'] - 5 * np.log10(
            self.df.loc[mask, 'distance_pc']
        ) + 5

        # 2. Converter para luminosidade
        # Magnitude bolométrica do Sol: M_bol☉ = 4.74
        # Para filtro G do Gaia, correção bolométrica aproximada
        M_G_sun = 4.67  # Magnitude absoluta do Sol no filtro G

        L_Lsun = 10 ** (0.4 * (M_G_sun - M_G))

        self.df.loc[mask, 'luminosidade_Lsun'] = L_Lsun
        self.df.loc[mask, 'magnitude_absoluta_G'] = M_G

        # 3. Se disponível, usar BP e RP para correção bolométrica melhor
        if ('phot_bp_mean_mag' in self.df.columns and
                'phot_rp_mean_mag' in self.df.columns):
            # Calcular cor BP-RP
            bp_rp = self.df.loc[mask, 'phot_bp_mean_mag'] - self.df.loc[mask, 'phot_rp_mean_mag']

            # Correção bolométrica empírica para Gaia (aproximada)
            # Baseada em relações de Andrae et al. 2018
            BC_G = -0.1 * bp_rp  # Aproximação simples

            # Magnitude bolométrica: M_bol = M_G + BC_G
            M_bol = M_G + BC_G
            M_bol_sun = 4.74  # Magnitude bolométrica do Sol

            L_Lsun_corrigida = 10 ** (0.4 * (M_bol_sun - M_bol))

            self.df.loc[mask, 'luminosidade_Lsun_corrigida'] = L_Lsun_corrigida
            self.df.loc[mask, 'correcao_bolometrica_G'] = BC_G

        # 4. Estatísticas
        n_calculado = mask.sum()
        print(f"Luminosidade calculada para {n_calculado} estrelas")
        if 'luminosidade_Lsun' in self.df.columns:
            valores = self.df['luminosidade_Lsun'].dropna()
            if len(valores) > 0:
                print(f"Range: {valores.min():.3e} - {valores.max():.3e} L☉")

        return self.df

    def calcular_snr_parallax(self):
        """Calcula SNR da paralaxe se disponível"""
        self.df = self.df.copy()

        if 'parallax' in self.df.columns and 'parallax_error' in self.df.columns:
            mask = (
                    (self.df['parallax'].notna()) &
                    (self.df['parallax_error'].notna()) &
                    (self.df['parallax_error'] > 0)
            )

            self.df.loc[mask, 'snr_parallax'] = (
                    self.df.loc[mask, 'parallax'] /
                    self.df.loc[mask, 'parallax_error']
            )

            # Classificação de qualidade
            self.df['parallax_quality_class'] = 'UNKNOWN'
            self.df.loc[mask & (self.df['snr_parallax'] >= 10), 'parallax_quality_class'] = 'EXCELLENT'
            self.df.loc[mask & (self.df['snr_parallax'] >= 5) & (
                    self.df['snr_parallax'] < 10), 'parallax_quality_class'] = 'GOOD'
            self.df.loc[mask & (self.df['snr_parallax'] >= 2) & (
                    self.df['snr_parallax'] < 5), 'parallax_quality_class'] = 'FAIR'
            self.df.loc[mask & (self.df['snr_parallax'] < 2), 'parallax_quality_class'] = 'POOR'

        return self.df

    def calcular_velocidades(self):
        """
        Calcula velocidades espaciais a partir de movimentos próprios e velocidade radial

        Retorna:
        - velocity_tangential_km_s: Velocidade tangencial
        - velocity_3d_km_s: Velocidade espacial total 3D
        - velocity_ra_km_s, velocity_dec_km_s: Componentes
        """
        self.df = self.df.copy()

        # Verificar requisitos mínimos
        if not ('pmra' in self.df.columns and 'pmdec' in self.df.columns):
            print("Aviso: Movimentos próprios não disponíveis")
            return self.df

        if 'distance_pc' not in self.df.columns:
            print("Aviso: Calculando distância primeiro...")
            self.df = self.calcular_distancia()

        # Máscara para dados válidos
        mask = (
                (self.df['pmra'].notna()) &
                (self.df['pmdec'].notna()) &
                (self.df['distance_pc'].notna()) &
                (self.df['distance_pc'] > 0)
        )

        if not mask.any():
            print("Aviso: Não há dados válidos para cálculo de velocidades")
            return self.df

        # Constantes de conversão
        # 1 mas/yr = 4.74047 km/s a 1 pc
        k = 4.74047  # km/s * pc / (mas/yr)

        # 1. Velocidade tangencial total
        # vt [km/s] = k * d [pc] * μ [mas/yr]
        # onde μ = sqrt(pmra^2 + pmdec^2)

        # Converter pmra e pmdec para movimento próprio total
        pm_total = np.sqrt(
            self.df.loc[mask, 'pmra'] ** 2 +
            self.df.loc[mask, 'pmdec'] ** 2
        )

        self.df.loc[mask, 'velocity_tangential_km_s'] = (
                k * self.df.loc[mask, 'distance_pc'] * pm_total / 1000  # dividido por 1000 se pm em mas/yr
        )

        # 2. Componentes de velocidade
        # v_ra = k * d * pmra * cos(dec)
        # v_dec = k * d * pmdec

        # Converter Dec para radianos para correção
        dec_rad = np.radians(self.df.loc[mask, 'dec'])

        self.df.loc[mask, 'velocity_ra_km_s'] = (
                k * self.df.loc[mask, 'distance_pc'] *
                self.df.loc[mask, 'pmra'] * np.cos(dec_rad) / 1000
        )

        self.df.loc[mask, 'velocity_dec_km_s'] = (
                k * self.df.loc[mask, 'distance_pc'] *
                self.df.loc[mask, 'pmdec'] / 1000
        )

        # 3. Velocidade 3D total (se velocidade radial disponível)
        if 'radial_velocity' in self.df.columns:
            mask_rv = mask & self.df['radial_velocity'].notna()

            if mask_rv.any():
                self.df.loc[mask_rv, 'velocity_3d_km_s'] = np.sqrt(
                    self.df.loc[mask_rv, 'velocity_tangential_km_s'] ** 2 +
                    self.df.loc[mask_rv, 'radial_velocity'] ** 2
                )

        # 4. Calcular erros se disponíveis
        if ('pmra_error' in self.df.columns and
                'pmdec_error' in self.df.columns and
                'distance_error_pc' in self.df.columns):
            # Propagação de erros para velocidade tangencial (aproximada)
            # σ_vt ≈ k * sqrt((d*σ_μ)^2 + (μ*σ_d)^2)

            # Erro no movimento próprio
            sigma_mu = np.sqrt(
                (self.df.loc[mask, 'pmra'] * self.df.loc[mask, 'pmra_error']) ** 2 +
                (self.df.loc[mask, 'pmdec'] * self.df.loc[mask, 'pmdec_error']) ** 2
            ) / pm_total

            self.df.loc[mask, 'velocity_tangential_error_km_s'] = k * np.sqrt(
                (self.df.loc[mask, 'distance_pc'] * sigma_mu) ** 2 +
                (pm_total * self.df.loc[mask, 'distance_error_pc']) ** 2
            ) / 1000

        # 5. Log
        n_calculado = mask.sum()
        print(f"Velocidades calculadas para {n_calculado} estrelas")

        if 'velocity_tangential_km_s' in self.df.columns:
            valores = self.df.loc[mask, 'velocity_tangential_km_s'].dropna()
            if len(valores) > 0:
                print(f"Velocidade tangencial: {valores.mean():.1f} ± {valores.std():.1f} km/s")

        return self.df

    def calcular_indices_cor(self):
        """Calcula índices de cor e relações fotométricas"""
        self.df = self.df.copy()

        # 1. Cores básicas (já podem existir)
        if 'phot_bp_mean_mag' in self.df.columns and 'phot_rp_mean_mag' in self.df.columns:
            mask = (
                    (self.df['phot_bp_mean_mag'].notna()) &
                    (self.df['phot_rp_mean_mag'].notna())
            )

            # BP - RP (se não existir)
            if 'bp_rp' not in self.df.columns:
                self.df.loc[mask, 'bp_rp'] = (
                        self.df.loc[mask, 'phot_bp_mean_mag'] -
                        self.df.loc[mask, 'phot_rp_mean_mag']
                )

        # 2. BP - G
        if 'phot_bp_mean_mag' in self.df.columns and 'phot_g_mean_mag' in self.df.columns:
            mask = (
                    (self.df['phot_bp_mean_mag'].notna()) &
                    (self.df['phot_g_mean_mag'].notna())
            )
            self.df.loc[mask, 'bp_g'] = (
                    self.df.loc[mask, 'phot_bp_mean_mag'] -
                    self.df.loc[mask, 'phot_g_mean_mag']
            )

        # 3. G - RP
        if 'phot_g_mean_mag' in self.df.columns and 'phot_rp_mean_mag' in self.df.columns:
            mask = (
                    (self.df['phot_g_mean_mag'].notna()) &
                    (self.df['phot_rp_mean_mag'].notna())
            )
            self.df.loc[mask, 'g_rp'] = (
                    self.df.loc[mask, 'phot_g_mean_mag'] -
                    self.df.loc[mask, 'phot_rp_mean_mag']
            )

        # 4. Temperatura efetiva a partir da cor (se não disponível)
        if 'teff_gspphot' not in self.df.columns and 'bp_rp' in self.df.columns:
            # Relação empírica aproximada: Teff em função de BP-RP
            # Baseada em relações para estrelas da sequência principal

            def bp_rp_to_teff(bp_rp):
                """Converte BP-RP para temperatura efetiva (aproximada)"""
                # Relação polinomial aproximada
                # Valores típicos: BP-RP = -0.5 (azul) a 3.0 (vermelho)

                # Para evitar valores extremos
                bp_rp_clipped = np.clip(bp_rp, -0.5, 3.5)

                # Polinômio aproximado (log Teff em função de BP-RP)
                # Coeficientes empíricos
                log_teff = (
                        3.95 - 0.25 * bp_rp_clipped +
                        0.05 * bp_rp_clipped ** 2 -
                        0.01 * bp_rp_clipped ** 3
                )

                teff = 10 ** log_teff

                # Limitar faixa razoável
                teff = np.clip(teff, 2500, 50000)

                return teff

            mask = self.df['bp_rp'].notna()
            if mask.any():
                self.df.loc[mask, 'teff_from_color'] = bp_rp_to_teff(
                    self.df.loc[mask, 'bp_rp']
                )

        # 5. Excesso de cor (estimativa de avermelhamento)
        if ('bp_rp' in self.df.columns and
                'teff_gspphot' in self.df.columns):

            # Cor intrínseca esperada para a temperatura
            # Usando relação aproximada
            def teff_to_bp_rp_intrinsic(teff):
                """Cor intrínseca BP-RP para dada temperatura"""
                log_teff = np.log10(teff)

                # Relação inversa da anterior
                bp_rp_intrinsic = (
                        15.8 - 4.0 * log_teff +
                        0.4 * log_teff ** 2
                )

                return np.clip(bp_rp_intrinsic, -0.5, 3.0)

            mask = (
                    (self.df['teff_gspphot'].notna()) &
                    (self.df['bp_rp'].notna())
            )

            if mask.any():
                self.df.loc[mask, 'bp_rp_intrinsic'] = teff_to_bp_rp_intrinsic(
                    self.df.loc[mask, 'teff_gspphot']
                )

                # Excesso de cor E(BP-RP) = (BP-RP)observado - (BP-RP)intrínseco
                self.df.loc[mask, 'color_excess_bp_rp'] = (
                        self.df.loc[mask, 'bp_rp'] -
                        self.df.loc[mask, 'bp_rp_intrinsic']
                )

                # Excesso de cor tipicamente positivo (avermelhamento)
                self.df.loc[mask, 'color_excess_bp_rp'] = self.df.loc[mask, 'color_excess_bp_rp'].clip(0, 2)

        # 6. Magnitude absoluta em diferentes bandas
        if 'distance_pc' in self.df.columns:
            # Magnitude absoluta G
            if 'phot_g_mean_mag' in self.df.columns:
                mask = (
                        (self.df['phot_g_mean_mag'].notna()) &
                        (self.df['distance_pc'].notna()) &
                        (self.df['distance_pc'] > 0)
                )
                self.df.loc[mask, 'abs_mag_G'] = (
                        self.df.loc[mask, 'phot_g_mean_mag'] -
                        5 * np.log10(self.df.loc[mask, 'distance_pc']) + 5
                )

            # Magnitude absoluta BP
            if 'phot_bp_mean_mag' in self.df.columns:
                mask = (
                        (self.df['phot_bp_mean_mag'].notna()) &
                        (self.df['distance_pc'].notna()) &
                        (self.df['distance_pc'] > 0)
                )
                self.df.loc[mask, 'abs_mag_BP'] = (
                        self.df.loc[mask, 'phot_bp_mean_mag'] -
                        5 * np.log10(self.df.loc[mask, 'distance_pc']) + 5
                )

            # Magnitude absoluta RP
            if 'phot_rp_mean_mag' in self.df.columns:
                mask = (
                        (self.df['phot_rp_mean_mag'].notna()) &
                        (self.df['distance_pc'].notna()) &
                        (self.df['distance_pc'] > 0)
                )
                self.df.loc[mask, 'abs_mag_RP'] = (
                        self.df.loc[mask, 'phot_rp_mean_mag'] -
                        5 * np.log10(self.df.loc[mask, 'distance_pc']) + 5
                )

        print("Índices de cor calculados")

        return self.df


# ============================================================
# FUNÇÕES AUXILIARES COMUNS
# ============================================================

def calcular_tamanho_magnitude(mag, min_mag=-2, max_mag=24):
    """Calcula tamanho não-linear baseado na magnitude"""
    if pd.isna(mag):
        return 3.0

    mag = np.clip(mag, min_mag, max_mag)
    tamanho = 24 * np.exp(-mag / 5)
    tamanho = np.clip(tamanho, 0.5, 18)
    return tamanho


def calcular_massa_estrela(df, idx):
    """Tenta obter massa de uma estrela de diferentes colunas"""
    if 'massa_estimada_Msun' in df.columns and not pd.isna(df.iloc[idx]['massa_estimada_Msun']):
        return df.iloc[idx]['massa_estimada_Msun']
    elif 'mass' in df.columns and not pd.isna(df.iloc[idx]['mass']):
        return df.iloc[idx]['mass']
    elif 'phot_g_mean_mag' in df.columns and not pd.isna(df.iloc[idx]['phot_g_mean_mag']):
        # Estimativa grosseira baseada na magnitude
        mag = df.iloc[idx]['phot_g_mean_mag']
        return 10 ** (-mag / 10 + 1)  # Aproximação
    else:
        return 1.0  # Massa solar padrão


def ra_dec_dist_to_cartesian(ra_deg, dec_deg, dist_pc):
    """
    Converte coordenadas esféricas (RA, Dec, Distância) para coordenadas cartesianas
    """
    # Converter para radianos
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    # Converter distância para metros
    dist_m = dist_pc * PC_TO_M

    # Coordenadas cartesianas
    x = dist_m * np.cos(dec_rad) * np.cos(ra_rad)
    y = dist_m * np.cos(dec_rad) * np.sin(ra_rad)
    z = dist_m * np.sin(dec_rad)

    return x, y, z


def cartesian_to_ra_dec_dist(x, y, z):
    """
    Converte coordenadas cartesianas para RA, Dec, Distância
    """
    dist_m = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dist_pc = dist_m / PC_TO_M

    # Calcular Dec
    if dist_m > 0:
        dec_rad = np.arcsin(z / dist_m)
    else:
        dec_rad = 0.0

    # Calcular RA
    if x != 0 or y != 0:
        ra_rad = np.arctan2(y, x)
    else:
        ra_rad = 0.0

    # Converter para graus e garantir RA entre 0-360
    ra_deg = np.degrees(ra_rad) % 360
    dec_deg = np.degrees(dec_rad)

    return ra_deg, dec_deg, dist_pc


# ============================================================
# CLASSE DE ANÁLISE GRAVITACIONAL
# ============================================================

class AnaliseGravitacional:
    """Analisa conexões gravitacionais entre estrelas com correções 3D"""

    def __init__(self, df_estrelas):
        self.df = df_estrelas
        self.resultados = None
        self.grafo = None
        self.coords_cartesianas = None  # Armazenar coordenadas cartesianas

    def calcular_conexoes_gravitacionais(self, raio_max_pc=2.0, forca_min_N=1e20):
        """
        Calcula todas as conexões gravitacionais significativas
        dentro de um raio máximo usando coordenadas cartesianas
        """
        # Converter para coordenadas cartesianas em METROS
        self.coords_cartesianas = np.column_stack(
            ra_dec_dist_to_cartesian(
                self.df['ra'].values,
                self.df['dec'].values,
                self.df['distance_pc'].values
            )
        )

        # Obter massas em kg
        massas_Msun = np.array([calcular_massa_estrela(self.df, i)
                                for i in range(len(self.df))])
        massas_kg = massas_Msun * MSUN_TO_KG

        # KDTree para vizinhos próximos
        raio_max_m = raio_max_pc * PC_TO_M
        tree = KDTree(self.coords_cartesianas)

        # Encontrar todos os pares dentro do raio
        pares = list(tree.query_pairs(raio_max_m))

        # Calcular forças e armazenar resultados
        conexoes = []

        for i, j in pares:
            # Vetor distância
            r_vec = self.coords_cartesianas[j] - self.coords_cartesianas[i]
            r = np.linalg.norm(r_vec)

            # Força gravitacional (N)
            F = G * (massas_kg[i] * massas_kg[j]) / (r ** 2)

            if F >= forca_min_N:
                # Calcular distância em parsecs para exibição
                dist_pc = r / PC_TO_M

                conexoes.append({
                    'estrela1': int(i),
                    'estrela2': int(j),
                    'estrela1_id': self.df.iloc[i]['source_id'] if 'source_id' in self.df.columns else i,
                    'estrela2_id': self.df.iloc[j]['source_id'] if 'source_id' in self.df.columns else j,
                    'distancia_pc': dist_pc,
                    'forca_N': F,
                    'forca_log10': np.log10(F),
                    'massa1_Msun': massas_Msun[i],
                    'massa2_Msun': massas_Msun[j],
                    'massa_total_Msun': massas_Msun[i] + massas_Msun[j]
                })

        self.resultados = pd.DataFrame(conexoes)
        return self.resultados

    def construir_grafo_gravitacional(self):
        """Constrói grafo das conexões gravitacionais"""
        if self.resultados is None or len(self.resultados) == 0:
            return None

        self.grafo = nx.Graph()

        # Adicionar nós (estrelas)
        for idx in range(len(self.df)):
            massa = calcular_massa_estrela(self.df, idx)
            self.grafo.add_node(idx, massa_Msun=massa)

        # Adicionar arestas (conexões gravitacionais)
        for _, row in self.resultados.iterrows():
            self.grafo.add_edge(
                row['estrela1'],
                row['estrela2'],
                forca_N=row['forca_N'],
                distancia_pc=row['distancia_pc'],
                peso=row['forca_log10']  # Peso para visualização
            )

        return self.grafo

    def identificar_grupos_gravitacionais(self, min_estrelas=3):
        """Identifica grupos conectados no grafo gravitacional"""
        if self.grafo is None:
            self.construir_grafo_gravitacional()

        if self.grafo is None:
            return []

        # Encontrar componentes conectados
        grupos = list(nx.connected_components(self.grafo))

        # Filtrar por tamanho mínimo e calcular estatísticas
        grupos_info = []

        for i, grupo in enumerate(grupos):
            if len(grupo) >= min_estrelas:
                estrelas = list(grupo)

                # Calcular estatísticas do grupo
                subgrafo = self.grafo.subgraph(estrelas)

                # Massa total
                massas = [self.grafo.nodes[n]['massa_Msun'] for n in estrelas]
                massa_total = sum(massas)

                # Força total (soma das forças das conexões)
                forca_total = sum(data['forca_N'] for _, _, data in subgrafo.edges(data=True))

                # Distâncias médias
                distancias = [data['distancia_pc'] for _, _, data in subgrafo.edges(data=True)]
                distancia_media = np.mean(distancias) if distancias else 0

                # Centro do grupo em coordenadas originais
                ra = self.df.iloc[estrelas]['ra'].values
                dec = self.df.iloc[estrelas]['dec'].values
                dist = self.df.iloc[estrelas]['distance_pc'].values

                grupos_info.append({
                    'id': i + 1,
                    'estrelas': estrelas,
                    'n_estrelas': len(estrelas),
                    'massa_total_Msun': massa_total,
                    'forca_total_N': forca_total,
                    'distancia_media_pc': distancia_media,
                    'centro_ra': np.mean(ra),
                    'centro_dec': np.mean(dec),
                    'centro_dist_pc': np.mean(dist),
                    'ra_std': np.std(ra),
                    'dec_std': np.std(dec),
                    'dist_std': np.std(dist),
                    'densidade_gravitacional': forca_total / (distancia_media ** 2) if distancia_media > 0 else 0
                })

        # Ordenar por força total (grupos mais fortemente ligados primeiro)
        grupos_info.sort(key=lambda x: x['forca_total_N'], reverse=True)

        return grupos_info


# ============================================================
# CLASSE CALCULADORA DE EIXOS SIMPLIFICADA
# ============================================================

class CalculadoraEixos:
    """Calculadora simplificada para dimensões de eixos da caixa de estrelas"""

    @staticmethod
    def calcular_dimensoes(ra_center, dec_center, delta_ra, delta_dec, dist_min, dist_max):
        """
        Calcula as dimensões físicas da caixa de estrelas 3D

        Retorna:
            dict: Dicionário com todas as dimensões calculadas
        """
        # Validar valores
        if dist_min >= dist_max:
            raise ValueError("Distância mínima deve ser menor que distância máxima!")

        # Calcular dimensões
        dec_rad = np.radians(dec_center)
        cos_dec = np.cos(dec_rad)

        # Converter ΔRA e ΔDec de graus para radianos
        delta_ra_rad = np.radians(delta_ra)
        delta_dec_rad = np.radians(delta_dec)

        # Calcular dimensões no limite inferior
        dx_min = dist_min * delta_ra_rad * cos_dec
        dy_min = dist_min * delta_dec_rad
        dz = dist_max - dist_min

        # Calcular dimensões no limite superior
        dx_max = dist_max * delta_ra_rad * cos_dec
        dy_max = dist_max * delta_dec_rad

        # Calcular dimensões médias
        dist_avg = (dist_min + dist_max) / 2
        dx_avg = dist_avg * delta_ra_rad * cos_dec
        dy_avg = dist_avg * delta_dec_rad

        # Calcular volume aproximado
        volume = dx_avg * dy_avg * dz

        # Preparar resultados
        resultados = {
            'dz': dz,
            'dx_min': dx_min,
            'dy_min': dy_min,
            'dx_max': dx_max,
            'dy_max': dy_max,
            'dx_avg': dx_avg,
            'dy_avg': dy_avg,
            'volume': volume,
            'cos_dec': cos_dec,
            'ra_center': ra_center,
            'dec_center': dec_center,
            'delta_ra': delta_ra,
            'delta_dec': delta_dec,
            'dist_min': dist_min,
            'dist_max': dist_max
        }

        return resultados

    @staticmethod
    def formatar_resultados(resultados):
        """Formata os resultados para exibição em texto"""
        return f"""
        DIMENSÕES DA CAIXA DE ESTRELAS
        ===============================

        Parâmetros de entrada:
        • Centro: RA={resultados['ra_center']:.2f}°, Dec={resultados['dec_center']:.2f}°
        • Extensão angular: ΔRA={resultados['delta_ra']:.2f}°, ΔDec={resultados['delta_dec']:.2f}°
        • Distância: {resultados['dist_min']:.1f} a {resultados['dist_max']:.1f} pc

        Dimensões físicas:
        • Profundidade (Δz): {resultados['dz']:.1f} pc

        No limite inferior ({resultados['dist_min']:.0f} pc):
        • Largura (Δx): {resultados['dx_min']:.1f} pc
        • Altura (Δy): {resultados['dy_min']:.1f} pc

        No limite superior ({resultados['dist_max']:.0f} pc):
        • Largura (Δx): {resultados['dx_max']:.1f} pc
        • Altura (Δy): {resultados['dy_max']:.1f} pc

        Valores médias:
        • Largura média (Δx): {resultados['dx_avg']:.1f} pc
        • Altura média (Δy): {resultados['dy_avg']:.1f} pc

        Proporções (x:y:z):
        • Inferior: 1 : {resultados['dy_min'] / resultados['dx_min']:.2f} : {resultados['dz'] / resultados['dx_min']:.2f}
        • Superior: 1 : {resultados['dy_max'] / resultados['dx_max']:.2f} : {resultados['dz'] / resultados['dx_max']:.2f}
        • Média: 1 : {resultados['dy_avg'] / resultados['dx_avg']:.2f} : {resultados['dz'] / resultados['dx_avg']:.2f}

        Fator cos(Dec): {resultados['cos_dec']:.4f}
        Volume aproximado: {resultados['volume']:.2e} pc³
        """


# ============================================================
# CLASSE PRINCIPAL COM GUI COM ABAS
# ============================================================

class SistemaIntegradoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema Integrado Gaia DR3 + Calculador de Colunas + Analisador Gravitacional 3D")
        self.root.geometry("1400x900")

        # Dados compartilhados entre abas
        self.df_estrelas = None
        self.df_filtrado = None
        self.analise_grav = None
        self.grupos_gravitacionais = None

        # Variáveis para Gaia
        self.server_choice = tk.StringVar(value="astron")
        self.gaia_data = None
        self.query_thread = None
        self.job_id = None
        self.job_status = None

        # Variáveis para calculador de colunas
        self.calc_distance_var = tk.BooleanVar(value=True)
        self.calc_mass_var = tk.BooleanVar(value=True)
        self.calc_lum_var = tk.BooleanVar(value=True)
        self.calc_snr_var = tk.BooleanVar(value=True)
        self.calc_velocity_var = tk.BooleanVar(value=True)
        self.calc_color_var = tk.BooleanVar(value=True)
        self.snr_minimo = tk.StringVar(value="5")
        self.lutz_var = tk.BooleanVar(value=True)
        self.mass_method = tk.StringVar(value="Luminosidade")

        # Variáveis para visualização 3D
        self.aspect_x = tk.DoubleVar(value=1.0)
        self.aspect_y = tk.DoubleVar(value=1.0)
        self.aspect_z = tk.DoubleVar(value=1.0)
        self.rotacao_ativa = tk.BooleanVar(value=True)
        self.velocidade_rotacao = tk.DoubleVar(value=1.0)
        self.eixo_rotacao = tk.StringVar(value="auto")
        self.duracao_rotacao = tk.IntVar(value=30)

        # Variáveis para calculadora de eixos
        self.ra_center = tk.DoubleVar(value=180.0)
        self.dec_center = tk.DoubleVar(value=0.0)
        self.delta_ra = tk.DoubleVar(value=4.0)
        self.delta_dec = tk.DoubleVar(value=4.0)
        self.dist_min_calc = tk.DoubleVar(value=900.0)
        self.dist_max_calc = tk.DoubleVar(value=1200.0)

        self.setup_gui()

    def setup_gui(self):
        # Notebook principal com abas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Aba 1: Consulta Gaia
        self.gaia_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.gaia_frame, text="🔭 Consulta Gaia DR3")
        self.setup_gaia_tab()

        # Aba 2: Calculador de Colunas (NOVA ABA)
        self.calc_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.calc_frame, text="🧮 Calculador de Colunas")
        self.setup_calculador_tab()

        # Aba 3: Calculadora de Eixos
        self.eixos_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.eixos_frame, text="📐 Calculadora de Eixos")
        self.setup_eixos_tab()

        # Aba 4: Análise Gravitacional
        self.analise_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analise_frame, text="🪐 Análise Gravitacional")
        self.setup_analise_tab()

        # Aba 5: Visualização 3D
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="🌌 Visualização 3D")
        self.setup_viz_tab()

        # Aba 6: Logs e Resultados
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="📊 Logs e Resultados")
        self.setup_logs_tab()

        # Barra de status
        self.status = tk.StringVar(value="Sistema Integrado Gaia + Calculador de Colunas + Gravitacional - Pronto")
        status_bar = ttk.Label(self.root, textvariable=self.status,
                               relief=tk.SUNKEN, font=('Arial', 8), padding=2)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ============================================================
    # ABA 1: CONSULTA GAIA
    # ============================================================
    def setup_gaia_tab(self):
        # Frame principal
        main_frame = ttk.Frame(self.gaia_frame, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Painel esquerdo - Controles
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

        # Seção Servidor
        server_frame = ttk.LabelFrame(left_panel, text="Configurações do Servidor", padding="5")
        server_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Radiobutton(server_frame, text="ESA (gea.esac.esa.int)",
                        variable=self.server_choice, value="esa").pack(anchor=tk.W)
        ttk.Radiobutton(server_frame, text="ASTRON (gaia.aip.de)",
                        variable=self.server_choice, value="astron").pack(anchor=tk.W)

        ttk.Button(server_frame, text="Testar Conexão",
                   command=self.test_connection).pack(pady=5)

        # Seção Coordenadas
        coords_frame = ttk.LabelFrame(left_panel, text="Coordenadas", padding="5")
        coords_frame.pack(fill=tk.X, pady=(0, 10))

        # RA
        ra_frame = ttk.Frame(coords_frame)
        ra_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ra_frame, text="RA (°):", width=12).pack(side=tk.LEFT)
        self.ra_min = ttk.Entry(ra_frame, width=8)
        self.ra_min.insert(0, "166.0")
        self.ra_min.pack(side=tk.LEFT, padx=2)
        ttk.Label(ra_frame, text="a").pack(side=tk.LEFT, padx=2)
        self.ra_max = ttk.Entry(ra_frame, width=8)
        self.ra_max.insert(0, "170.0")
        self.ra_max.pack(side=tk.LEFT, padx=2)

        # DEC
        dec_frame = ttk.Frame(coords_frame)
        dec_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dec_frame, text="DEC (°):", width=12).pack(side=tk.LEFT)
        self.dec_min = ttk.Entry(dec_frame, width=8)
        self.dec_min.insert(0, "-62.0")
        self.dec_min.pack(side=tk.LEFT, padx=2)
        ttk.Label(dec_frame, text="a").pack(side=tk.LEFT, padx=2)
        self.dec_max = ttk.Entry(dec_frame, width=8)
        self.dec_max.insert(0, "-59.0")
        self.dec_max.pack(side=tk.LEFT, padx=2)

        # Distância
        dist_frame = ttk.Frame(coords_frame)
        dist_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dist_frame, text="Distância (pc):", width=12).pack(side=tk.LEFT)
        self.dist_min = ttk.Entry(dist_frame, width=8)
        self.dist_min.insert(0, "2000.0")
        self.dist_min.pack(side=tk.LEFT, padx=2)
        ttk.Label(dist_frame, text="a").pack(side=tk.LEFT, padx=2)
        self.dist_max = ttk.Entry(dist_frame, width=8)
        self.dist_max.insert(0, "3200.0")
        self.dist_max.pack(side=tk.LEFT, padx=2)

        # Seção Filtros
        filters_frame = ttk.LabelFrame(left_panel, text="Filtros", padding="5")
        filters_frame.pack(fill=tk.X, pady=(0, 10))

        # Magnitude
        mag_frame = ttk.Frame(filters_frame)
        mag_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mag_frame, text="Mag G Max:", width=12).pack(side=tk.LEFT)
        self.mag_limit = ttk.Entry(mag_frame, width=10)
        self.mag_limit.insert(0, "18.0")
        self.mag_limit.pack(side=tk.LEFT)

        # SNR Paralaxe
        snr_frame = ttk.Frame(filters_frame)
        snr_frame.pack(fill=tk.X, pady=2)
        ttk.Label(snr_frame, text="SNR Paralaxe Min:", width=12).pack(side=tk.LEFT)
        self.parallax_snr = ttk.Entry(snr_frame, width=10)
        self.parallax_snr.insert(0, "5.0")
        self.parallax_snr.pack(side=tk.LEFT)

        # Seção Configurações
        config_frame = ttk.LabelFrame(left_panel, text="Configurações", padding="5")
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # Max Registros
        max_frame = ttk.Frame(config_frame)
        max_frame.pack(fill=tk.X, pady=2)
        ttk.Label(max_frame, text="Máx. Registros:", width=12).pack(side=tk.LEFT)
        self.max_records = ttk.Entry(max_frame, width=10)
        self.max_records.insert(0, "10000")
        self.max_records.pack(side=tk.LEFT)

        # Timeout
        timeout_frame = ttk.Frame(config_frame)
        timeout_frame.pack(fill=tk.X, pady=2)
        ttk.Label(timeout_frame, text="Timeout (s):", width=12).pack(side=tk.LEFT)
        self.timeout_val = ttk.Entry(timeout_frame, width=10)
        self.timeout_val.insert(0, "120")
        self.timeout_val.pack(side=tk.LEFT)

        # Status Job
        self.gaia_status_label = ttk.Label(left_panel, text="Pronto para executar")
        self.gaia_status_label.pack(pady=5)

        self.gaia_job_id_label = ttk.Label(left_panel, text="Job ID: Nenhum", foreground="blue")
        self.gaia_job_id_label.pack(pady=5)

        # Botões - GARANTIR QUE EXISTAM
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill=tk.X, pady=10)

        # Botão Executar Query - SEMPRE presente
        self.execute_btn = ttk.Button(button_frame, text="Executar Query", command=self.start_gaia_query)
        self.execute_btn.pack(fill=tk.X, pady=2)

        # Botão Verificar Status - SEMPRE presente (inicialmente desabilitado)
        self.check_btn = ttk.Button(button_frame, text="Verificar Status",
                                    command=self.check_gaia_job_status, state=tk.DISABLED)
        self.check_btn.pack(fill=tk.X, pady=2)

        # Botão Salvar CSV - SEMPRE presente (inicialmente desabilitado)
        self.save_btn = ttk.Button(button_frame, text="Salvar CSV",
                                   command=self.save_gaia_data, state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X, pady=2)

        # Botão Limpar Campos - SEMPRE presente
        self.clear_btn = ttk.Button(button_frame, text="Limpar Campos", command=self.clear_gaia_fields)
        self.clear_btn.pack(fill=tk.X, pady=2)

        # Botão Carregar para Calculador - SEMPRE presente (inicialmente desabilitado)
        self.load_calc_btn = ttk.Button(button_frame, text="Carregar para Calculador",
                                        command=self.load_gaia_to_calculador, state=tk.DISABLED)
        self.load_calc_btn.pack(fill=tk.X, pady=2)

        # Progresso
        self.gaia_progress = ttk.Progressbar(left_panel, mode='indeterminate')
        self.gaia_progress.pack(fill=tk.X, pady=5)

        # Painel direito - Log Gaia
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(right_panel, text="Log Gaia DR3", font=('Arial', 10, 'bold')).pack(pady=(0, 5))

        self.gaia_log_text = scrolledtext.ScrolledText(right_panel, height=30, font=('Consolas', 8))
        self.gaia_log_text.pack(fill=tk.BOTH, expand=True)

    # ============================================================
    # ABA 2: CALCULADOR DE COLUNAS (NOVA)
    # ============================================================
    def setup_calculador_tab(self):
        """Configura a interface do calculador de colunas"""
        # Frame principal
        main_frame = ttk.Frame(self.calc_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame esquerdo - Controles
        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_frame.pack_propagate(False)

        # Frame direito - Log e Visualização
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ========== FRAME ESQUERDO ==========

        # Título
        title_label = ttk.Label(left_frame,
                                text="🧮 CALCULADOR DE COLUNAS",
                                font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 15))

        # Seção 1: Carregamento
        load_frame = ttk.LabelFrame(left_frame, text="📁 Carregamento de Dados", padding="10")
        load_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(load_frame, text="Arquivo CSV:", font=('Arial', 9)).pack(anchor=tk.W, pady=(0, 5))

        self.calc_file_entry = ttk.Entry(load_frame)
        self.calc_file_entry.pack(fill=tk.X, pady=(0, 5))

        btn_frame = ttk.Frame(load_frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="Procurar",
                   command=self.calc_browse_file).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(btn_frame, text="Carregar",
                   command=self.calc_load_data).pack(side=tk.LEFT)

        # Info do arquivo
        self.calc_file_info = ttk.Label(load_frame, text="", font=('Arial', 8))
        self.calc_file_info.pack(fill=tk.X, pady=(5, 0))

        # Seção 2: Parâmetros de cálculo
        param_frame = ttk.LabelFrame(left_frame, text="⚙️ Parâmetros de Cálculo", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 15))

        # SNR mínimo para paralaxe
        ttk.Label(param_frame, text="SNR Mínimo Paralaxe:", font=('Arial', 9)).grid(row=0, column=0, sticky=tk.W,
                                                                                    pady=2)
        self.calc_snr_entry = ttk.Entry(param_frame, width=10)
        self.calc_snr_entry.insert(0, "5")
        self.calc_snr_entry.grid(row=0, column=1, padx=5, pady=2)

        # Aplicar correção Lutz-Kelker
        ttk.Checkbutton(param_frame, text="Aplicar correção Lutz-Kelker",
                        variable=self.lutz_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)

        # Método de cálculo de massa
        ttk.Label(param_frame, text="Método Massa:", font=('Arial', 9)).grid(row=2, column=0, sticky=tk.W, pady=2)
        self.calc_mass_method = ttk.Combobox(param_frame,
                                             values=["Luminosidade", "Teff-Logg", "Cor-Magnitude", "Padrão"],
                                             state="readonly",
                                             width=15,
                                             textvariable=self.mass_method)
        self.calc_mass_method.set("Luminosidade")
        self.calc_mass_method.grid(row=2, column=1, padx=5, pady=2)

        # Seção 3: Colunas para calcular
        calc_frame = ttk.LabelFrame(left_frame, text="📊 Colunas para Calcular", padding="10")
        calc_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Checkbutton(calc_frame, text="Distância (distance_pc)",
                        variable=self.calc_distance_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(calc_frame, text="Massa (massa_estimada_Msun)",
                        variable=self.calc_mass_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(calc_frame, text="Luminosidade (luminosidade_Lsun)",
                        variable=self.calc_lum_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(calc_frame, text="SNR Paralaxe (snr_parallax)",
                        variable=self.calc_snr_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(calc_frame, text="Velocidade Espacial (velocity_3d)",
                        variable=self.calc_velocity_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(calc_frame, text="Índices de Cor (color_indices)",
                        variable=self.calc_color_var).pack(anchor=tk.W, pady=2)

        # Botão Calcular Tudo
        ttk.Button(calc_frame, text="📈 Calcular Todas as Colunas",
                   command=self.calcular_todas_colunas,
                   style='Accent.TButton').pack(fill=tk.X, pady=(10, 0))

        # Seção 4: Ações
        action_frame = ttk.LabelFrame(left_frame, text="💾 Ações", padding="10")
        action_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Button(action_frame, text="Visualizar Dados Calculados",
                   command=self.visualizar_dados_calculados).pack(fill=tk.X, pady=2)

        ttk.Button(action_frame, text="Exportar CSV com '_calc'",
                   command=self.exportar_csv_calculado).pack(fill=tk.X, pady=2)

        ttk.Button(action_frame, text="Carregar para Análise",
                   command=self.carregar_para_analise).pack(fill=tk.X, pady=2)

        ttk.Button(action_frame, text="Gerar Relatório Estatístico",
                   command=self.gerar_relatorio_calculador).pack(fill=tk.X, pady=2)

        # Status
        self.calc_status_var = tk.StringVar(value="🟢 Pronto")
        status_bar = ttk.Label(left_frame, textvariable=self.calc_status_var,
                               relief=tk.SUNKEN, font=('Arial', 8))
        status_bar.pack(fill=tk.X, pady=(10, 0))

        # ========== FRAME DIREITO ==========

        # Notebook com abas
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Aba 1: Log
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="📝 Log")

        self.calc_log_text = scrolledtext.ScrolledText(log_frame, height=30,
                                                       font=('Consolas', 9))
        self.calc_log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Aba 2: Dados
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="📊 Dados")

        # Treeview para mostrar dados
        self.calc_tree_frame = ttk.Frame(data_frame)
        self.calc_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbars
        tree_scroll_y = ttk.Scrollbar(self.calc_tree_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        tree_scroll_x = ttk.Scrollbar(self.calc_tree_frame, orient=tk.HORIZONTAL)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        # Treeview
        self.calc_tree = ttk.Treeview(self.calc_tree_frame,
                                      yscrollcommand=tree_scroll_y.set,
                                      xscrollcommand=tree_scroll_x.set)
        self.calc_tree.pack(fill=tk.BOTH, expand=True)

        tree_scroll_y.config(command=self.calc_tree.yview)
        tree_scroll_x.config(command=self.calc_tree.xview)

        # Aba 3: Estatísticas
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="📈 Estatísticas")

        self.calc_stats_text = scrolledtext.ScrolledText(stats_frame, height=30,
                                                         font=('Consolas', 9))
        self.calc_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Dados do calculador
        self.calc_df = None

    def calc_log(self, message):
        """Adiciona mensagem ao log do calculador"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.calc_log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.calc_log_text.see(tk.END)
        self.root.update_idletasks()

    def calc_browse_file(self):
        """Seleciona arquivo CSV para o calculador"""
        filename = filedialog.askopenfilename(
            title="Selecionar arquivo CSV",
            filetypes=[("CSV files", "*.csv"), ("Todos os arquivos", "*.*")]
        )
        if filename:
            self.calc_file_entry.delete(0, tk.END)
            self.calc_file_entry.insert(0, filename)

    def calc_load_data(self):
        """Carrega os dados do arquivo CSV para o calculador"""
        filename = self.calc_file_entry.get()
        if not filename:
            messagebox.showerror("Erro", "Selecione um arquivo CSV")
            return

        try:
            self.calc_log(f"Carregando arquivo: {filename}")
            self.calc_status_var.set("🟡 Carregando dados...")

            self.calc_df = pd.read_csv(filename)

            # Verificar colunas mínimas
            colunas_minimas = ['source_id', 'ra', 'dec', 'parallax']
            colunas_faltantes = [c for c in colunas_minimas if c not in self.calc_df.columns]

            if colunas_faltantes:
                messagebox.showerror("Erro", f"Colunas obrigatórias faltantes: {colunas_faltantes}")
                return

            self.calc_log(f"Dados carregados: {len(self.calc_df)} estrelas")
            self.calc_log(f"Colunas disponíveis: {', '.join(self.calc_df.columns.tolist())}")

            # Mostrar colunas no treeview
            self._atualizar_calc_treeview()

            # Analisar dados disponíveis
            self._analisar_colunas_disponiveis()

            self.calc_status_var.set(f"🟢 Dados carregados: {len(self.calc_df)} estrelas")
            self.calc_file_info.config(text=f"{len(self.calc_df)} estrelas, {len(self.calc_df.columns)} colunas")

        except Exception as e:
            self.calc_log(f"❌ Erro ao carregar dados: {str(e)}")
            messagebox.showerror("Erro", f"Falha ao carregar: {str(e)}")
            self.calc_status_var.set("🔴 Erro ao carregar")

    def _atualizar_calc_treeview(self):
        """Atualiza o treeview do calculador com os dados"""
        # Limpar treeview existente
        for item in self.calc_tree.get_children():
            self.calc_tree.delete(item)

        if self.calc_df is None:
            return

        # Configurar colunas
        colunas = self.calc_df.columns.tolist()
        self.calc_tree["columns"] = colunas
        self.calc_tree["show"] = "headings"

        # Configurar cabeçalhos
        for col in colunas:
            self.calc_tree.heading(col, text=col)
            self.calc_tree.column(col, width=100, minwidth=50)

        # Adicionar dados (mostrar apenas primeiras 100 linhas para performance)
        n_mostrar = min(100, len(self.calc_df))
        for i in range(n_mostrar):
            valores = [str(self.calc_df.iloc[i][col])[:20] for col in colunas]
            self.calc_tree.insert("", "end", values=valores)

        self.calc_log(f"Mostrando {n_mostrar} de {len(self.calc_df)} linhas no treeview")

    def _analisar_colunas_disponiveis(self):
        """Analisa quais colunas estão disponíveis para cálculo"""
        if self.calc_df is None:
            return

        colunas_disp = set(self.calc_df.columns)

        self.calc_log("\n🔍 ANÁLISE DE COLUNAS DISPONÍVEIS:")
        self.calc_log("-" * 40)

        # Verificar colunas para distância
        if {'parallax', 'parallax_error'}.issubset(colunas_disp):
            self.calc_log("✅ Distância: pode ser calculada com erro")
        elif 'parallax' in colunas_disp:
            self.calc_log("⚠️  Distância: pode ser calculada sem erro")
        else:
            self.calc_log("❌ Distância: impossível calcular (falta parallax)")

        # Verificar colunas para massa
        if 'teff_gspphot' in colunas_disp and 'logg_gspphot' in colunas_disp:
            self.calc_log("✅ Massa: pode ser calculada via Teff-Logg")
        elif 'phot_g_mean_mag' in colunas_disp:
            self.calc_log("✅ Massa: pode ser calculada via magnitude")
        else:
            self.calc_log("⚠️  Massa: será usado valor padrão")

        # Verificar colunas para velocidade
        if {'pmra', 'pmdec', 'radial_velocity'}.issubset(colunas_disp):
            self.calc_log("✅ Velocidade 3D: pode ser calculada")
        elif {'pmra', 'pmdec'}.issubset(colunas_disp):
            self.calc_log("⚠️  Velocidade: apenas velocidade tangencial")
        else:
            self.calc_log("❌ Velocidade: impossível calcular")

        # Verificar cores
        if {'phot_bp_mean_mag', 'phot_rp_mean_mag'}.issubset(colunas_disp):
            self.calc_log("✅ Índices de cor: podem ser calculados")

        self.calc_log("-" * 40)

    def calcular_todas_colunas(self):
        """Calcula todas as colunas selecionadas no calculador"""
        if self.calc_df is None:
            messagebox.showerror("Erro", "Carregue os dados primeiro")
            return

        try:
            self.calc_log("\n" + "=" * 60)
            self.calc_log("🧮 INICIANDO CÁLCULO DE COLUNAS")
            self.calc_log("=" * 60)

            self.calc_status_var.set("🟡 Calculando colunas...")

            # Obter parâmetros
            snr_minimo = float(self.calc_snr_entry.get())
            aplicar_lutz = self.lutz_var.get()
            metodo_massa = self.mass_method.get()

            # Criação do calculador
            calculador = CalculadorColunas(self.calc_df)

            # Distância
            if self.calc_distance_var.get():
                self.calc_log("\n📏 Calculando distâncias...")
                self.calc_df = calculador.calcular_distancia(
                    snr_minimo=snr_minimo,
                    aplicar_lutz_kelker=aplicar_lutz
                )

            # Massa
            if self.calc_mass_var.get():
                self.calc_log("\n⚖️  Calculando massas...")
                self.calc_df = calculador.calcular_massa(metodo=metodo_massa)

            # Luminosidade
            if self.calc_lum_var.get():
                self.calc_log("\n🌟 Calculando luminosidades...")
                self.calc_df = calculador.calcular_luminosidade()

            # SNR
            if self.calc_snr_var.get():
                self.calc_log("\n📊 Calculando SNR...")
                self.calc_df = calculador.calcular_snr_parallax()

            # Velocidade
            if self.calc_velocity_var.get():
                self.calc_log("\n🚀 Calculando velocidades...")
                self.calc_df = calculador.calcular_velocidades()

            # Cores
            if self.calc_color_var.get():
                self.calc_log("\n🎨 Calculando índices de cor...")
                self.calc_df = calculador.calcular_indices_cor()

            # Atualizar treeview
            self._atualizar_calc_treeview()

            # Mostrar estatísticas
            self._mostrar_estatisticas_calculadas()

            self.calc_log("\n✅ Cálculo de colunas concluído!")
            self.calc_status_var.set("🟢 Cálculos concluídos")

            messagebox.showinfo("Concluído", "Cálculo de colunas realizado com sucesso!")

        except Exception as e:
            self.calc_log(f"❌ Erro no cálculo: {str(e)}")
            import traceback
            self.calc_log(f"🔍 Detalhes: {traceback.format_exc()}")
            messagebox.showerror("Erro", f"Falha no cálculo:\n{str(e)}")
            self.calc_status_var.set("🔴 Erro no cálculo")

    def _mostrar_estatisticas_calculadas(self):
        """Mostra estatísticas das colunas calculadas no calculador"""
        if self.calc_df is None:
            return

        colunas_calculadas = []

        for col in ['distance_pc', 'massa_estimada_Msun', 'luminosidade_Lsun',
                    'snr_parallax', 'velocity_3d_km_s']:
            if col in self.calc_df.columns:
                colunas_calculadas.append(col)

        if not colunas_calculadas:
            return

        self.calc_log("\n📈 ESTATÍSTICAS DAS COLUNAS CALCULADAS:")
        self.calc_log("-" * 50)

        for col in colunas_calculadas:
            if col in self.calc_df.columns and self.calc_df[col].notna().any():
                valores = self.calc_df[col].dropna()
                if len(valores) > 0:
                    self.calc_log(f"{col}:")
                    self.calc_log(f"  Min: {valores.min():.3f}")
                    self.calc_log(f"  Max: {valores.max():.3f}")
                    self.calc_log(f"  Média: {valores.mean():.3f}")
                    self.calc_log(f"  Mediana: {valores.median():.3f}")
                    self.calc_log(f"  Não-nulos: {valores.count()} ({valores.count() / len(self.calc_df) * 100:.1f}%)")

        # Mostrar no widget de estatísticas também
        self._atualizar_calc_estatisticas()

    def _atualizar_calc_estatisticas(self):
        """Atualiza o widget de estatísticas do calculador"""
        if self.calc_df is None:
            return

        texto = "📊 ESTATÍSTICAS DO CONJUNTO\n"
        texto += "=" * 50 + "\n\n"
        texto += f"Total de estrelas: {len(self.calc_df)}\n"
        texto += f"Colunas totais: {len(self.calc_df.columns)}\n\n"

        # Listar colunas calculadas
        colunas_originais = [
            'source_id', 'ra', 'dec', 'parallax', 'parallax_error',
            'pmra', 'pmdec', 'radial_velocity', 'pmra_error', 'pmdec_error',
            'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
            'bp_rp', 'bp_g', 'g_rp', 'phot_bp_rp_excess_factor',
            'teff_gspphot', 'logg_gspphot', 'mh_gspphot', 'azero_gspphot'
        ]

        colunas_calculadas = [col for col in self.calc_df.columns if col not in colunas_originais]

        if colunas_calculadas:
            texto += "COLUNAS CALCULADAS:\n"
            texto += "-" * 30 + "\n"
            for col in colunas_calculadas:
                texto += f"• {col}\n"
            texto += "\n"

        # Estatísticas básicas das principais colunas calculadas
        colunas_principais = ['distance_pc', 'massa_estimada_Msun',
                              'luminosidade_Lsun', 'velocity_3d_km_s']

        for col in colunas_principais:
            if col in self.calc_df.columns:
                valores = self.calc_df[col].dropna()
                if len(valores) > 0:
                    texto += f"{col.upper()}:\n"
                    texto += f"  Não-nulos: {valores.count()} ({valores.count() / len(self.calc_df) * 100:.1f}%)\n"
                    texto += f"  Média: {valores.mean():.3f}\n"
                    texto += f"  Desvio: {valores.std():.3f}\n"
                    texto += f"  Range: {valores.min():.3f} - {valores.max():.3f}\n\n"

        self.calc_stats_text.delete(1.0, tk.END)
        self.calc_stats_text.insert(tk.END, texto)

    def visualizar_dados_calculados(self):
        """Mostra preview dos dados calculados no calculador"""
        if self.calc_df is None:
            messagebox.showerror("Erro", "Carregue e calcule os dados primeiro")
            return

        # Criar janela de visualização
        viz_window = tk.Toplevel(self.root)
        viz_window.title("Visualização dos Dados Calculados")
        viz_window.geometry("1000x600")

        # Treeview
        tree_frame = ttk.Frame(viz_window)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Scrollbars
        scroll_y = ttk.Scrollbar(tree_frame)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        scroll_x = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        # Tree
        tree = ttk.Treeview(tree_frame,
                            yscrollcommand=scroll_y.set,
                            xscrollcommand=scroll_x.set)
        tree.pack(fill=tk.BOTH, expand=True)

        scroll_y.config(command=tree.yview)
        scroll_x.config(command=tree.xview)

        # Colunas
        colunas = self.calc_df.columns.tolist()
        tree["columns"] = colunas
        tree["show"] = "headings"

        for col in colunas:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        # Dados (limitado para performance)
        n_mostrar = min(50, len(self.calc_df))
        for i in range(n_mostrar):
            valores = [str(self.calc_df.iloc[i][col])[:30] for col in colunas]
            tree.insert("", "end", values=valores)

        # Label informativo
        info_label = ttk.Label(viz_window,
                               text=f"Mostrando {n_mostrar} de {len(self.calc_df)} linhas",
                               font=('Arial', 9))
        info_label.pack(pady=(0, 10))

    def exportar_csv_calculado(self):
        """Exporta os dados com colunas calculadas para CSV com '_calc' no nome"""
        if self.calc_df is None:
            messagebox.showerror("Erro", "Não há dados para exportar")
            return

        # Obter nome do arquivo original
        original_filename = self.calc_file_entry.get()
        if not original_filename:
            # Usar diálogo padrão
            filename = filedialog.asksaveasfilename(
                title="Salvar dados calculados",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Todos os arquivos", "*.*")]
            )
            if not filename:
                return
        else:
            # Adicionar '_calc' ao nome do arquivo
            import os
            base_name = os.path.splitext(original_filename)[0]
            extension = os.path.splitext(original_filename)[1]
            filename = f"{base_name}_calc{extension}"

            # Confirmar com o usuário
            from tkinter import simpledialog
            filename = filedialog.asksaveasfilename(
                title="Salvar dados calculados",
                defaultextension=".csv",
                initialfile=os.path.basename(filename),
                filetypes=[("CSV files", "*.csv"), ("Todos os arquivos", "*.*")]
            )
            if not filename:
                return

        try:
            self.calc_df.to_csv(filename, index=False)
            self.calc_log(f"✅ Dados exportados: {filename}")
            self.log_general(f"Dados calculados exportados: {filename}")
            messagebox.showinfo("Sucesso", f"Dados calculados salvos em:\n{filename}")
        except Exception as e:
            self.calc_log(f"❌ Erro na exportação: {str(e)}")
            messagebox.showerror("Erro", f"Falha ao salvar:\n{str(e)}")

    def carregar_para_analise(self):
        """Carrega os dados calculados para a aba de análise gravitacional"""
        if self.calc_df is None:
            messagebox.showerror("Erro", "Não há dados calculados para carregar")
            return

        # Verificar se tem coluna distance_pc
        if 'distance_pc' not in self.calc_df.columns:
            messagebox.showerror("Erro", "Os dados não têm coluna 'distance_pc' calculada")
            return

        # Carregar para análise gravitacional
        self.df_estrelas = self.calc_df.copy()
        self.df_filtrado = self.df_estrelas.copy()

        # Atualizar estatísticas na aba de análise
        self.stats_text.delete(1.0, tk.END)
        self.update_stats(f"✅ DADOS CALCULADOS CARREGADOS PARA ANÁLISE")
        self.update_stats(f"Total de estrelas: {len(self.df_estrelas)}")
        self.update_stats(f"RA: {self.df_estrelas['ra'].min():.1f} a {self.df_estrelas['ra'].max():.1f}°")
        self.update_stats(f"Dec: {self.df_estrelas['dec'].min():.1f} a {self.df_estrelas['dec'].max():.1f}°")
        self.update_stats(
            f"Distância: {self.df_estrelas['distance_pc'].min():.1f} a {self.df_estrelas['distance_pc'].max():.1f} pc")

        # Verificar se tem coluna de massa calculada
        if 'massa_estimada_Msun' in self.df_estrelas.columns:
            massas = self.df_estrelas['massa_estimada_Msun'].dropna()
            if len(massas) > 0:
                self.update_stats(f"Massa: {massas.min():.3f} a {massas.max():.3f} M☉")

        self.log_general(f"Dados calculados carregados para análise: {len(self.df_estrelas)} estrelas")
        self.status.set(f"Dados calculados carregados - {len(self.df_estrelas)} estrelas")

        # Ir para aba de análise gravitacional
        self.notebook.select(3)  # Índice 3 = Análise Gravitacional

    def gerar_relatorio_calculador(self):
        """Gera relatório estatístico para o calculador"""
        if self.calc_df is None:
            messagebox.showerror("Erro", "Carregue os dados primeiro")
            return

        try:
            self.calc_log("Gerando relatório estatístico...")

            relatorio = self._criar_relatorio_estatistico_calculador()

            # Mostrar em nova janela
            report_window = tk.Toplevel(self.root)
            report_window.title("Relatório Estatístico - Calculador")
            report_window.geometry("800x600")

            text_widget = scrolledtext.ScrolledText(report_window,
                                                    font=('Consolas', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget.insert(tk.END, relatorio)

            # Botão para salvar
            btn_frame = ttk.Frame(report_window)
            btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

            ttk.Button(btn_frame, text="Salvar Relatório",
                       command=lambda: self._salvar_relatorio_calculador(relatorio)).pack(side=tk.LEFT)

            self.calc_log("✅ Relatório gerado")

        except Exception as e:
            self.calc_log(f"❌ Erro ao gerar relatório: {str(e)}")

    def _criar_relatorio_estatistico_calculador(self):
        """Cria relatório estatístico detalhado para o calculador"""
        from datetime import datetime

        relatorio = "=" * 60 + "\n"
        relatorio += "RELATÓRIO ESTATÍSTICO - DADOS CALCULADOS\n"
        relatorio += "=" * 60 + "\n\n"
        relatorio += f"Data da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        relatorio += f"Total de estrelas: {len(self.calc_df)}\n\n"

        # 1. Informações gerais
        relatorio += "1. INFORMAÇÕES GERAIS\n"
        relatorio += "-" * 40 + "\n"
        relatorio += f"Número de colunas: {len(self.calc_df.columns)}\n"
        if 'parallax' in self.calc_df.columns:
            relatorio += f"Estrelas com parallax válida: {self.calc_df['parallax'].notna().sum()} "
            relatorio += f"({self.calc_df['parallax'].notna().sum() / len(self.calc_df) * 100:.1f}%)\n\n"

        # 2. Colunas calculadas
        colunas_originais = [
            'source_id', 'ra', 'dec', 'parallax', 'parallax_error',
            'pmra', 'pmdec', 'radial_velocity', 'pmra_error', 'pmdec_error',
            'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag',
            'bp_rp', 'bp_g', 'g_rp', 'phot_bp_rp_excess_factor',
            'teff_gspphot', 'logg_gspphot', 'mh_gspphot', 'azero_gspphot'
        ]

        colunas_calculadas = [col for col in self.calc_df.columns if col not in colunas_originais]

        if colunas_calculadas:
            relatorio += "2. COLUNAS CALCULADAS\n"
            relatorio += "-" * 40 + "\n"
            for col in colunas_calculadas:
                na_count = self.calc_df[col].notna().sum()
                relatorio += f"{col}: {na_count} valores ({na_count / len(self.calc_df) * 100:.1f}%)\n"
            relatorio += "\n"

        # 3. Estatísticas detalhadas das principais colunas
        relatorio += "3. ESTATÍSTICAS DETALHADAS\n"
        relatorio += "-" * 40 + "\n"

        colunas_analise = ['distance_pc', 'massa_estimada_Msun',
                           'luminosidade_Lsun', 'velocity_3d_km_s',
                           'teff_gspphot', 'logg_gspphot']

        for col in colunas_analise:
            if col in self.calc_df.columns:
                valores = self.calc_df[col].dropna()
                if len(valores) > 0:
                    relatorio += f"\n{col.upper()}:\n"
                    relatorio += f"  Contagem: {len(valores)} estrelas\n"
                    relatorio += f"  Média: {valores.mean():.3f} ± {valores.std():.3f}\n"
                    relatorio += f"  Mediana: {valores.median():.3f}\n"
                    relatorio += f"  Mínimo: {valores.min():.3f}\n"
                    relatorio += f"  Máximo: {valores.max():.3f}\n"

        relatorio += "\n" + "=" * 60 + "\n"
        relatorio += "FIM DO RELATÓRIO\n"
        relatorio += "=" * 60 + "\n"

        return relatorio

    def _salvar_relatorio_calculador(self, relatorio):
        """Salva o relatório do calculador em arquivo"""
        filename = filedialog.asksaveasfilename(
            title="Salvar relatório",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Todos os arquivos", "*.*")]
        )

        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(relatorio)
            self.calc_log(f"✅ Relatório salvo: {filename}")
            messagebox.showinfo("Sucesso", f"Relatório salvo em:\n{filename}")

    # ============================================================
    # ABA 3: CALCULADORA DE EIXOS
    # ============================================================
    def setup_eixos_tab(self):
        # Frame principal
        main_frame = ttk.Frame(self.eixos_frame, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame de entrada de dados
        input_frame = ttk.LabelFrame(main_frame, text="Parâmetros da Amostra", padding="15")
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        # Campos de entrada
        row = 0

        # Centro RA
        ttk.Label(input_frame, text="Centro RA (graus):").grid(row=row, column=0, sticky=tk.W, pady=5)
        ra_spin = ttk.Spinbox(input_frame, from_=0, to=360, textvariable=self.ra_center, width=15)
        ra_spin.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        row += 1

        # Centro Dec
        ttk.Label(input_frame, text="Centro Dec (graus):").grid(row=row, column=0, sticky=tk.W, pady=5)
        dec_spin = ttk.Spinbox(input_frame, from_=-90, to=90, textvariable=self.dec_center, width=15)
        dec_spin.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        row += 1

        # ΔRA
        ttk.Label(input_frame, text="ΔRA (graus):").grid(row=row, column=0, sticky=tk.W, pady=5)
        delta_ra_spin = ttk.Spinbox(input_frame, from_=0.1, to=360, textvariable=self.delta_ra, width=15)
        delta_ra_spin.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        row += 1

        # ΔDec
        ttk.Label(input_frame, text="ΔDec (graus):").grid(row=row, column=0, sticky=tk.W, pady=5)
        delta_dec_spin = ttk.Spinbox(input_frame, from_=0.1, to=180, textvariable=self.delta_dec, width=15)
        delta_dec_spin.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        row += 1

        # Distância mínima
        ttk.Label(input_frame, text="Distância mínima (pc):").grid(row=row, column=0, sticky=tk.W, pady=5)
        dist_min_spin = ttk.Spinbox(input_frame, from_=1, to=10000, textvariable=self.dist_min_calc, width=15)
        dist_min_spin.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        row += 1

        # Distância máxima
        ttk.Label(input_frame, text="Distância máxima (pc):").grid(row=row, column=0, sticky=tk.W, pady=5)
        dist_max_spin = ttk.Spinbox(input_frame, from_=1, to=10000, textvariable=self.dist_max_calc, width=15)
        dist_max_spin.grid(row=row, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        row += 1

        # Botões
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=row, column=0, columnspan=2, pady=20)

        ttk.Button(button_frame, text="Calcular Dimensões", command=self.calcular_dimensoes_eixos).pack(side=tk.LEFT,
                                                                                                        padx=5)
        ttk.Button(button_frame, text="Copiar para Gaia", command=self.copiar_para_gaia).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Limpar", command=self.limpar_eixos).pack(side=tk.LEFT, padx=5)

        # Frame de resultados
        results_frame = ttk.LabelFrame(main_frame, text="Resultados", padding="15")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(15, 5))

        self.eixos_results_text = tk.Text(results_frame, height=20, font=("Courier", 10))
        self.eixos_results_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Scrollbar para resultados
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.eixos_results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.eixos_results_text.config(yscrollcommand=scrollbar.set)

    def calcular_dimensoes_eixos(self):
        """Calcula as dimensões dos eixos da caixa de estrelas"""
        try:
            # Obter valores
            ra_center = self.ra_center.get()
            dec_center = self.dec_center.get()
            delta_ra = self.delta_ra.get()
            delta_dec = self.delta_dec.get()
            dist_min = self.dist_min_calc.get()
            dist_max = self.dist_max_calc.get()

            # Calcular dimensões usando a classe CalculadoraEixos
            resultados = CalculadoraEixos.calcular_dimensoes(
                ra_center, dec_center, delta_ra, delta_dec, dist_min, dist_max
            )

            # Formatar e exibir resultados
            texto_resultados = CalculadoraEixos.formatar_resultados(resultados)

            self.eixos_results_text.delete(1.0, tk.END)
            self.eixos_results_text.insert(1.0, texto_resultados)

            self.log_general(f"Calculadas dimensões de eixos: RA={ra_center}°, Dec={dec_center}°, "
                             f"ΔRA={delta_ra}°, ΔDec={delta_dec}°, Dist={dist_min}-{dist_max} pc")

            messagebox.showinfo("Sucesso", "Dimensões calculadas com sucesso!")

        except ValueError as e:
            messagebox.showerror("Erro", str(e))
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro: {str(e)}")

    def copiar_para_gaia(self):
        """Copia os valores calculados para os campos da aba Gaia"""
        try:
            # Obter valores atuais
            ra_center = self.ra_center.get()
            dec_center = self.dec_center.get()
            delta_ra = self.delta_ra.get()
            delta_dec = self.delta_dec.get()
            dist_min = self.dist_min_calc.get()
            dist_max = self.dist_max_calc.get()

            # Calcular RA min/max
            ra_min = ra_center - delta_ra / 2
            ra_max = ra_center + delta_ra / 2

            # Garantir que RA fique entre 0-360 graus
            if ra_min < 0:
                ra_min += 360
            if ra_max > 360:
                ra_max -= 360

            # Calcular Dec min/max
            dec_min = dec_center - delta_dec / 2
            dec_max = dec_center + delta_dec / 2

            # Garantir que Dec fique entre -90 e 90 graus
            dec_min = max(-90.0, dec_min)
            dec_max = min(90.0, dec_max)

            # Atualizar campos da aba Gaia
            self.ra_min.delete(0, tk.END)
            self.ra_min.insert(0, f"{ra_min:.2f}")

            self.ra_max.delete(0, tk.END)
            self.ra_max.insert(0, f"{ra_max:.2f}")

            self.dec_min.delete(0, tk.END)
            self.dec_min.insert(0, f"{dec_min:.2f}")

            self.dec_max.delete(0, tk.END)
            self.dec_max.insert(0, f"{dec_max:.2f}")

            self.dist_min.delete(0, tk.END)
            self.dist_min.insert(0, f"{dist_min:.1f}")

            self.dist_max.delete(0, tk.END)
            self.dist_max.insert(0, f"{dist_max:.1f}")

            # Mudar para a aba Gaia
            self.notebook.select(0)  # Índice 0 = Consulta Gaia DR3

            self.log_general(f"Valores copiados para Gaia: RA={ra_min:.2f}-{ra_max:.2f}°, "
                             f"Dec={dec_min:.2f}-{dec_max:.2f}°, Dist={dist_min:.1f}-{dist_max:.1f} pc")

            messagebox.showinfo("Sucesso", "Valores copiados para a aba Gaia DR3!")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao copiar valores: {str(e)}")

    def limpar_eixos(self):
        """Limpa os campos e resultados da calculadora de eixos"""
        self.ra_center.set(180.0)
        self.dec_center.set(0.0)
        self.delta_ra.set(4.0)
        self.delta_dec.set(4.0)
        self.dist_min_calc.set(900.0)
        self.dist_max_calc.set(1200.0)
        self.eixos_results_text.delete(1.0, tk.END)

    # ============================================================
    # ABA 4: ANÁLISE GRAVITACIONAL
    # ============================================================
    def setup_analise_tab(self):
        main_frame = ttk.Frame(self.analise_frame, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Painel esquerdo - Controles
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

        # Seção Carregamento
        load_frame = ttk.LabelFrame(left_panel, text="Carregamento de Dados", padding="5")
        load_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(load_frame, text="Arquivo CSV:").pack(anchor=tk.W)
        file_frame = ttk.Frame(load_frame)
        file_frame.pack(fill=tk.X, pady=2)
        self.file_entry = ttk.Entry(file_frame)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(file_frame, text="Procurar", command=self.browse_file, width=10).pack(side=tk.RIGHT)

        ttk.Button(load_frame, text="Carregar Dados", command=self.load_data).pack(fill=tk.X, pady=5)

        # Filtro Distância
        dist_filter_frame = ttk.LabelFrame(left_panel, text="Filtro de Distância", padding="5")
        dist_filter_frame.pack(fill=tk.X, pady=(0, 10))

        dist_input_frame = ttk.Frame(dist_filter_frame)
        dist_input_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dist_input_frame, text="Min (pc):").pack(side=tk.LEFT)
        self.min_dist = ttk.Entry(dist_input_frame, width=8)
        self.min_dist.insert(0, "0")
        self.min_dist.pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(dist_input_frame, text="Max (pc):").pack(side=tk.LEFT)
        self.max_dist = ttk.Entry(dist_input_frame, width=8)
        self.max_dist.insert(0, "550")
        self.max_dist.pack(side=tk.LEFT)

        self.filtro_ativo = tk.BooleanVar(value=False)
        ttk.Checkbutton(dist_filter_frame, text="Aplicar filtro de distância",
                        variable=self.filtro_ativo).pack(anchor=tk.W, pady=2)

        # Filtro Magnitude
        mag_filter_frame = ttk.LabelFrame(left_panel, text="Filtro de Magnitude", padding="5")
        mag_filter_frame.pack(fill=tk.X, pady=(0, 10))

        mag_input_frame = ttk.Frame(mag_filter_frame)
        mag_input_frame.pack(fill=tk.X, pady=2)
        ttk.Label(mag_input_frame, text="Min:").pack(side=tk.LEFT)
        self.min_mag = ttk.Entry(mag_input_frame, width=8)
        self.min_mag.insert(0, "-2")
        self.min_mag.pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(mag_input_frame, text="Max:").pack(side=tk.LEFT)
        self.max_mag = ttk.Entry(mag_input_frame, width=8)
        self.max_mag.insert(0, "24")
        self.max_mag.pack(side=tk.LEFT)

        self.filtro_mag_ativo = tk.BooleanVar(value=False)
        ttk.Checkbutton(mag_filter_frame, text="Aplicar filtro de magnitude",
                        variable=self.filtro_mag_ativo).pack(anchor=tk.W, pady=2)

        # Parâmetros Gravitacionais
        grav_frame = ttk.LabelFrame(left_panel, text="Parâmetros Gravitacionais", padding="5")
        grav_frame.pack(fill=tk.X, pady=(0, 10))

        grav_params = [
            ("Raio busca (pc):", "raio_grav", "2.0"),
            ("Força mín (N):", "forca_min", "2e17"),
            ("Mín. estrelas:", "min_estrelas_grav", "50")
        ]

        for label, attr, default in grav_params:
            frame = ttk.Frame(grav_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=label, width=15).pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=12)
            entry.insert(0, default)
            entry.pack(side=tk.LEFT)
            setattr(self, attr, entry)

        # Botões Ação
        action_frame = ttk.LabelFrame(left_panel, text="Ações", padding="5")
        action_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(action_frame, text="Analisar Gravidade", command=self.analisar_gravidade).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Exportar Resultados", command=self.exportar_resultados).pack(fill=tk.X, pady=2)

        # Painel direito - Estatísticas
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(right_panel, text="Estatísticas dos Dados", font=('Arial', 10, 'bold')).pack(pady=(0, 5))

        self.stats_text = scrolledtext.ScrolledText(right_panel, height=30, font=('Consolas', 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True)

    # ============================================================
    # ABA 5: VISUALIZAÇÃO 3D
    # ============================================================
    def setup_viz_tab(self):
        main_frame = ttk.Frame(self.viz_frame, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Painel esquerdo - Controles
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))

        # Aspect Ratio
        aspect_frame = ttk.LabelFrame(left_panel, text="Aspect Ratio 3D", padding="5")
        aspect_frame.pack(fill=tk.X, pady=(0, 10))

        # Eixo X
        x_frame = ttk.Frame(aspect_frame)
        x_frame.pack(fill=tk.X, pady=2)
        ttk.Label(x_frame, text="Eixo X:", width=10).pack(side=tk.LEFT)
        scale_x = ttk.Scale(x_frame, from_=0.1, to=10.0, variable=self.aspect_x,
                            orient=tk.HORIZONTAL, length=150)
        scale_x.pack(side=tk.LEFT, padx=5)
        entry_x = ttk.Entry(x_frame, width=8, textvariable=self.aspect_x)
        entry_x.pack(side=tk.LEFT)

        # Eixo Y
        y_frame = ttk.Frame(aspect_frame)
        y_frame.pack(fill=tk.X, pady=2)
        ttk.Label(y_frame, text="Eixo Y:", width=10).pack(side=tk.LEFT)
        scale_y = ttk.Scale(y_frame, from_=0.1, to=10.0, variable=self.aspect_y,
                            orient=tk.HORIZONTAL, length=150)
        scale_y.pack(side=tk.LEFT, padx=5)
        entry_y = ttk.Entry(y_frame, width=8, textvariable=self.aspect_y)
        entry_y.pack(side=tk.LEFT)

        # Eixo Z
        z_frame = ttk.Frame(aspect_frame)
        z_frame.pack(fill=tk.X, pady=2)
        ttk.Label(z_frame, text="Eixo Z:", width=10).pack(side=tk.LEFT)
        scale_z = ttk.Scale(z_frame, from_=0.1, to=10.0, variable=self.aspect_z,
                            orient=tk.HORIZONTAL, length=150)
        scale_z.pack(side=tk.LEFT, padx=5)
        entry_z = ttk.Entry(z_frame, width=8, textvariable=self.aspect_z)
        entry_z.pack(side=tk.LEFT)

        ttk.Button(aspect_frame, text="Resetar Aspect Ratio",
                   command=self.reset_aspect_ratio).pack(pady=5)

        # Rotação Automática
        rotation_frame = ttk.LabelFrame(left_panel, text="Rotação Automática", padding="5")
        rotation_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Checkbutton(rotation_frame, text="Ativar rotação automática",
                        variable=self.rotacao_ativa).pack(anchor=tk.W, pady=2)

        # Velocidade
        speed_frame = ttk.Frame(rotation_frame)
        speed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(speed_frame, text="Velocidade:", width=10).pack(side=tk.LEFT)
        speed_scale = ttk.Scale(speed_frame, from_=0.1, to=5.0, variable=self.velocidade_rotacao,
                                orient=tk.HORIZONTAL, length=150)
        speed_scale.pack(side=tk.LEFT, padx=5)

        # Eixo
        axis_frame = ttk.Frame(rotation_frame)
        axis_frame.pack(fill=tk.X, pady=2)
        ttk.Label(axis_frame, text="Eixo:", width=10).pack(side=tk.LEFT)
        ttk.Radiobutton(axis_frame, text="Auto", variable=self.eixo_rotacao, value="auto").pack(side=tk.LEFT)
        ttk.Radiobutton(axis_frame, text="X", variable=self.eixo_rotacao, value="x").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(axis_frame, text="Y", variable=self.eixo_rotacao, value="y").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(axis_frame, text="Z", variable=self.eixo_rotacao, value="z").pack(side=tk.LEFT, padx=5)

        # Duração
        duration_frame = ttk.Frame(rotation_frame)
        duration_frame.pack(fill=tk.X, pady=2)
        ttk.Label(duration_frame, text="Duração (s):", width=10).pack(side=tk.LEFT)
        duration_spin = ttk.Spinbox(duration_frame, from_=5, to=60, textvariable=self.duracao_rotacao,
                                    width=8)
        duration_spin.pack(side=tk.LEFT)

        # Botões Visualização
        viz_buttons_frame = ttk.Frame(left_panel)
        viz_buttons_frame.pack(fill=tk.X, pady=10)

        ttk.Button(viz_buttons_frame, text="Visualizar 3D", command=self.visualizar_com_ligacoes).pack(fill=tk.X,
                                                                                                       pady=2)

        # Informações
        info_frame = ttk.LabelFrame(left_panel, text="Informações", padding="5")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.viz_info_text = tk.Text(info_frame, height=8, width=40, font=('Arial', 8))
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.viz_info_text.yview)
        self.viz_info_text.configure(yscrollcommand=scrollbar.set)
        self.viz_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Painel direito - Instruções
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(right_panel, text="Instruções de Visualização", font=('Arial', 10, 'bold')).pack(pady=(0, 5))

        instructions = """
🌌 VISUALIZAÇÃO 3D INTERATIVA

1. PRÉ-REQUISITOS:
   • Execute uma consulta Gaia ou carregue um arquivo CSV
   • Execute a análise gravitacional

2. CONTROLES:
   • Use o mouse para rotacionar manualmente
   • Scroll do mouse para zoom
   • Clique e arraste para mover

3. ROTAÇÃO AUTOMÁTICA:
   • Ative/desative com checkbox
   • Ajuste velocidade e eixo
   • Controles Play/Pause na visualização

4. ASPECT RATIO:
   • Ajuste proporções dos eixos
   • Útil para visualizar estruturas alongadas
   • Botão "Resetar" volta para 1:1:1

5. LEGENDA:
   • Estrelas brancas: todas as estrelas
   • Linhas cinza: conexões gravitacionais
   • Cores: grupos gravitacionais identificados

6. DICAS:
   • A visualização abre em navegador separado
   • Pode levar alguns segundos para carregar
   • Use modo tela cheia para melhor visualização
"""
        instructions_text = tk.Text(right_panel, height=30, font=('Arial', 9), wrap=tk.WORD)
        instructions_text.insert(1.0, instructions)
        instructions_text.config(state=tk.DISABLED)
        instructions_text.pack(fill=tk.BOTH, expand=True)

    # ============================================================
    # ABA 6: LOGS E RESULTADOS
    # ============================================================
    def setup_logs_tab(self):
        main_frame = ttk.Frame(self.logs_frame, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Notebook para diferentes logs
        logs_notebook = ttk.Notebook(main_frame)
        logs_notebook.pack(fill=tk.BOTH, expand=True)

        # Log Geral
        log_frame = ttk.Frame(logs_notebook)
        logs_notebook.add(log_frame, text="📝 Log Geral")

        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill=tk.X, padx=5, pady=(5, 0))
        ttk.Button(log_toolbar, text="Limpar Log", command=self.clear_general_log).pack(side=tk.LEFT)

        self.general_log_text = scrolledtext.ScrolledText(log_frame, height=25, font=('Consolas', 9))
        self.general_log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Resultados Gravitacionais
        grav_results_frame = ttk.Frame(logs_notebook)
        logs_notebook.add(grav_results_frame, text="🪐 Resultados Gravitacionais")

        grav_toolbar = ttk.Frame(grav_results_frame)
        grav_toolbar.pack(fill=tk.X, padx=5, pady=(5, 0))
        ttk.Button(grav_toolbar, text="Copiar", command=lambda: self.copy_to_clipboard(self.grav_results_text)).pack(
            side=tk.LEFT)
        ttk.Button(grav_toolbar, text="Exportar", command=self.export_grav_results).pack(side=tk.LEFT, padx=5)

        self.grav_results_text = scrolledtext.ScrolledText(grav_results_frame, height=25, font=('Consolas', 9))
        self.grav_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Dados Carregados
        data_frame = ttk.Frame(logs_notebook)
        logs_notebook.add(data_frame, text="📊 Dados Carregados")

        data_toolbar = ttk.Frame(data_frame)
        data_toolbar.pack(fill=tk.X, padx=5, pady=(5, 0))
        ttk.Label(data_toolbar, text="Visualizar dados atuais:").pack(side=tk.LEFT)
        ttk.Button(data_toolbar, text="Mostrar Amostra", command=self.show_data_sample).pack(side=tk.LEFT, padx=5)

        self.data_text = scrolledtext.ScrolledText(data_frame, height=25, font=('Consolas', 9))
        self.data_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Log Calculador
        calc_log_frame = ttk.Frame(logs_notebook)
        logs_notebook.add(calc_log_frame, text="🧮 Log Calculador")

        calc_log_toolbar = ttk.Frame(calc_log_frame)
        calc_log_toolbar.pack(fill=tk.X, padx=5, pady=(5, 0))
        ttk.Button(calc_log_toolbar, text="Copiar",
                   command=lambda: self.copy_to_clipboard(self.calc_log_copy_text)).pack(side=tk.LEFT)

        self.calc_log_copy_text = scrolledtext.ScrolledText(calc_log_frame, height=25, font=('Consolas', 9))
        self.calc_log_copy_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ============================================================
    # FUNÇÕES DE LOG E STATUS
    # ============================================================
    def log_gaia(self, msg):
        """Log para aba Gaia"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.gaia_log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.gaia_log_text.see(tk.END)
        self.root.update_idletasks()

        # Atualizar também no log geral e no log copiável
        self.log_general(f"[Gaia] {msg}")
        self.calc_log_copy_text.insert(tk.END, f"[{timestamp}] [Gaia] {msg}\n")
        self.calc_log_copy_text.see(tk.END)

    def log_general(self, msg):
        """Log para aba geral"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.general_log_text.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.general_log_text.see(tk.END)
        self.root.update_idletasks()

    def update_stats(self, msg):
        """Atualiza estatísticas na aba de análise"""
        self.stats_text.insert(tk.END, f"{msg}\n")
        self.stats_text.see(tk.END)
        self.root.update_idletasks()

    def update_viz_info(self, msg):
        """Atualiza informações na aba de visualização"""
        self.viz_info_text.insert(tk.END, f"{msg}\n")
        self.viz_info_text.see(tk.END)
        self.root.update_idletasks()

    def clear_general_log(self):
        """Limpa o log geral"""
        self.general_log_text.delete(1.0, tk.END)

    # ============================================================
    # FUNÇÕES GAIA (da aba 1)
    # ============================================================
    # ============================================================
    # FUNÇÕES GAIA CORRIGIDAS
    # ============================================================

    def create_session_with_retry(self):
        """Cria uma session com retry automático"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def extract_job_info_from_xml(self, xml_content):
        """Extrai Job ID e status de resposta XML UWS - Versão corrigida"""
        try:
            # Limpar o XML se necessário
            if isinstance(xml_content, bytes):
                xml_content = xml_content.decode('utf-8')

            self.log_gaia(f"Tamanho do XML: {len(xml_content)} caracteres")
            self.log_gaia(f"Primeiros 500 chars do XML: {xml_content[:500]}")

            # Verificar se é realmente XML
            if not xml_content.strip().startswith('<?xml') and not xml_content.strip().startswith('<'):
                self.log_gaia("Resposta não parece ser XML válido")
                # Tentar extrair informações como texto simples
                return self.extract_job_info_from_text(xml_content)

            # CORREÇÃO: Remover atributos duplicados do XML
            # Problema comum: xmlns duplicado
            lines = xml_content.split('\n')
            cleaned_lines = []
            for line in lines:
                # Remover atributos xmlns duplicados mantendo apenas o primeiro
                if 'xmlns=' in line and line.count('xmlns=') > 1:
                    # Manter apenas o primeiro xmlns
                    parts = line.split('xmlns=')
                    if len(parts) > 2:
                        # Reconstruir mantendo apenas o primeiro
                        new_line = parts[0] + 'xmlns=' + parts[1].split('"')[0] + '"'
                        # Adicionar o resto da linha após o primeiro xmlns
                        rest = '"'.join(parts[1].split('"')[2:]) if len(parts[1].split('"')) > 2 else ''
                        for part in parts[2:]:
                            rest += 'xmlns=' + part
                        new_line += rest
                        line = new_line
                cleaned_lines.append(line)

            xml_content = '\n'.join(cleaned_lines)

            # Tentar parsear com xml.etree
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as e:
                self.log_gaia(f"Erro ao parsear XML: {e}")
                # Tentar limpar mais agressivamente
                xml_content = self.clean_xml_string(xml_content)
                root = ET.fromstring(xml_content)

            # Namespace que o ASTRON usa
            ns = {'uws': 'http://www.ivoa.net/xml/UWS/v1.0'}

            # Tentar encontrar jobId com namespace
            job_id = None
            phase = None
            results_url = None

            # Procurar jobId
            for elem in root.iter():
                tag = elem.tag
                # Remover namespace para comparação
                if '}' in tag:
                    tag = tag.split('}')[-1]

                if tag == 'jobId':
                    job_id = elem.text
                elif tag == 'phase':
                    phase = elem.text
                elif tag == 'url' and elem.text and 'result' in elem.text:
                    results_url = elem.text

            # Se não encontrou, tentar com busca direta no texto
            if not job_id:
                import re
                # Padrões para buscar no XML/texto
                patterns = [
                    r'<jobId[^>]*>([^<]+)</jobId>',
                    r'jobId>([^<]+)<',
                    r'"jobId"\s*:\s*"([^"]+)"',  # JSON-like
                    r'jobId=([^&\s]+)'
                ]

                for pattern in patterns:
                    match = re.search(pattern, xml_content, re.IGNORECASE)
                    if match:
                        job_id = match.group(1).strip()
                        break

            if not phase:
                import re
                phase_patterns = [
                    r'<phase[^>]*>([^<]+)</phase>',
                    r'phase>([^<]+)<',
                    r'"phase"\s*:\s*"([^"]+)"',
                    r'Phase:\s*([^\s]+)'
                ]
                for pattern in phase_patterns:
                    match = re.search(pattern, xml_content, re.IGNORECASE)
                    if match:
                        phase = match.group(1).strip()
                        break

            self.log_gaia(f"Job ID encontrado: {job_id}")
            self.log_gaia(f"Phase encontrado: {phase}")
            self.log_gaia(f"Results URL: {results_url}")

            return {
                'job_id': job_id,
                'phase': phase or 'UNKNOWN',
                'results_url': results_url
            }

        except Exception as e:
            self.log_gaia(f"❌ Erro crítico ao extrair info do XML: {e}")
            # Fallback para extração de texto simples
            return self.extract_job_info_from_text(xml_content if 'xml_content' in locals() else str(e))

    def clean_xml_string(self, xml_string):
        """Limpa string XML removendo problemas comuns"""
        import re

        # Remover declarações XML duplicadas
        xml_decl_pattern = r'<\?xml[^>]*\?>'
        matches = re.findall(xml_decl_pattern, xml_string)
        if len(matches) > 1:
            # Manter apenas a primeira declaração
            xml_string = re.sub(xml_decl_pattern, '', xml_string, count=len(matches) - 1)

        # Remover caracteres de controle não-XML
        xml_string = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml_string)

        # Garantir que há apenas um elemento raiz
        root_tags = re.findall(r'<([a-zA-Z][^>\s]*)[^>]*>', xml_string)
        unique_root_tags = set(root_tags)

        # Se houver múltiplos possíveis elementos raiz, tentar isolar o uws:job
        if 'uws:job' in root_tags:
            # Extrair apenas a parte do uws:job
            start = xml_string.find('<uws:job')
            if start != -1:
                end = xml_string.find('</uws:job>', start)
                if end != -1:
                    xml_string = xml_string[start:end + 10]  # +10 para incluir </uws:job>

        return xml_string

    def extract_job_info_from_text(self, text):
        """Extrai informações de job de texto não-XML (fallback)"""
        import re

        self.log_gaia("Usando fallback de extração de texto")

        job_id = None
        phase = None
        results_url = None

        # Padrões para ASTRON
        patterns = [
            # Padrão XML-like
            (r'<jobId[^>]*>([^<]+)</jobId>', 1),
            (r'jobId>([^<]+)<', 1),
            # Padrão de URL (pode conter jobId)
            (r'job/([^/]+)', 1),
            (r'jobId=([^&\s]+)', 1),
            # Padrão ASTRON específico
            (r'id[^:]*:\s*([^\s]+)', 1),
        ]

        for pattern, group in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                job_id = match.group(group).strip()
                self.log_gaia(f"Job ID encontrado via padrão '{pattern}': {job_id}")
                break

        # Padrões para phase
        phase_patterns = [
            (r'<phase[^>]*>([^<]+)</phase>', 1),
            (r'phase>([^<]+)<', 1),
            (r'Phase:\s*([^\s]+)', 1),
            (r'"phase"\s*:\s*"([^"]+)"', 1),
        ]

        for pattern, group in phase_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                phase = match.group(group).strip()
                break

        # Se ainda não encontrou job_id, tentar extrair da resposta completa
        if not job_id and 'Location:' in text:
            lines = text.split('\n')
            for line in lines:
                if 'Location:' in line:
                    location = line.split('Location:')[1].strip()
                    # Extrair job_id da URL
                    job_match = re.search(r'/([^/]+)$', location)
                    if job_match:
                        job_id = job_match.group(1)
                        break

        self.log_gaia(f"Fallback - Job ID: {job_id}, Phase: {phase}")

        return {
            'job_id': job_id,
            'phase': phase or 'UNKNOWN',
            'results_url': results_url
        }

    def test_connection(self):
        """Testa a conexão com o servidor selecionado"""
        self.log_gaia("Testando conexão...")
        thread = threading.Thread(target=self._test_connection_thread)
        thread.daemon = True
        thread.start()

    def _test_connection_thread(self):
        """Thread para testar conexão"""
        try:
            if self.server_choice.get() == "astron":
                test_url = "https://gaia.aip.de/tap/availability"
                server_name = "ASTRON"
            else:
                test_url = "https://gea.esac.esa.int/tap-server/tap/availability"
                server_name = "ESA"

            session = self.create_session_with_retry()
            response = session.get(test_url, timeout=15)

            if response.status_code == 200:
                self.log_gaia(f"✅ Conexão OK com {server_name}")
            else:
                self.log_gaia(f"⚠️ {server_name} respondeu com status {response.status_code}")

        except Exception as e:
            self.log_gaia(f"❌ Erro de conexão: {e}")

    def validate_gaia_inputs(self):
        """Valida os inputs do usuário para Gaia"""
        try:
            ra_min = float(self.ra_min.get())
            ra_max = float(self.ra_max.get())
            dec_min = float(self.dec_min.get())
            dec_max = float(self.dec_max.get())
            dist_min = float(self.dist_min.get())
            dist_max = float(self.dist_max.get())
            mag_limit = float(self.mag_limit.get())
            parallax_snr = float(self.parallax_snr.get())
            max_records = int(self.max_records.get())

            if ra_min < 0 or ra_max > 360 or dec_min < -90 or dec_max > 90:
                messagebox.showerror("Erro", "Valores fora do range:\nRA: 0-360\nDEC: -90 a 90")
                return False
            if ra_min >= ra_max:
                messagebox.showerror("Erro", "RA Min deve ser menor que RA Max")
                return False
            if dec_min >= dec_max:
                messagebox.showerror("Erro", "DEC Min deve ser menor que DEC Max")
                return False
            if dist_min >= dist_max:
                messagebox.showerror("Erro", "Distância Min deve ser menor que Distância Max")
                return False
            if max_records <= 0:
                messagebox.showerror("Erro", "Máximo de registros deve ser positivo")
                return False
            if max_records > 100000:
                messagebox.showwarning("Aviso", "Máximo de registros muito alto. Use no máximo 100000.")
                return False

            return True
        except ValueError:
            messagebox.showerror("Erro", "Por favor, insira valores numéricos válidos")
            return False

    def build_gaia_query(self, ra_min, ra_max, dec_min, dec_max, dist_min, dist_max, mag_limit, parallax_snr,
                         max_records):
        """Constrói query ADQL completa"""
        # Converter distância para paralaxe
        parallax_max = 1000.0 / dist_min if dist_min > 0 else 1000.0
        parallax_min = 1000.0 / dist_max if dist_max > 0 else 0.001

        query = f"""
        SELECT TOP {max_records}
            source_id, 
            ra, 
            dec, 
            parallax, 
            parallax_error,
            pmra, 
            pmdec, 
            radial_velocity,        -- ADICIONADO para velocidade 3D
            pmra_error, 
            pmdec_error,
            phot_g_mean_mag, 
            phot_bp_mean_mag, 
            phot_rp_mean_mag,
            bp_rp, 
            bp_g, 
            g_rp,
            phot_bp_rp_excess_factor,
            teff_gspphot, 
            logg_gspphot, 
            mh_gspphot, 
            azero_gspphot,
            ruwe,
            ipd_gof_harmonic_amplitude,
            ipd_frac_multi_peak,
            ipd_frac_odd_win
        FROM gaiadr3.gaia_source
        WHERE 
            ra BETWEEN {ra_min} AND {ra_max}
            AND dec BETWEEN {dec_min} AND {dec_max}
            AND parallax BETWEEN {parallax_min} AND {parallax_max}
            AND parallax IS NOT NULL 
            AND parallax > 0
            AND phot_g_mean_mag < {mag_limit}
            AND parallax_over_error > {parallax_snr}
            AND ruwe < 1.4
        """
        return query.strip()

    def start_gaia_query(self):
        """Inicia a query Gaia em uma thread separada"""
        if not self.validate_gaia_inputs():
            return

        if self.query_thread and self.query_thread.is_alive():
            messagebox.showwarning("Aviso", "Uma query já está em execução")
            return

        # Desabilitar botão de execução apenas
        self.find_and_disable_button("Executar Query")

        self.gaia_progress.start()
        self.gaia_log_text.delete(1.0, tk.END)
        self.log_gaia("Iniciando query Gaia DR3...")

        self.query_thread = threading.Thread(target=self.submit_gaia_query)
        self.query_thread.daemon = True
        self.query_thread.start()

    def find_and_disable_button(self, text):
        """Encontra e desabilita um botão específico"""
        for widget in self.gaia_frame.winfo_children():
            if isinstance(widget, ttk.Button) and text in widget.cget("text"):
                widget.config(state=tk.DISABLED)
                return widget
        return None

    def find_and_enable_button(self, text):
        """Encontra e habilita um botão específico"""
        for widget in self.gaia_frame.winfo_children():
            if isinstance(widget, ttk.Button) and text in widget.cget("text"):
                widget.config(state=tk.NORMAL)
                return widget
        return None

    def submit_gaia_query(self):
        """Submete uma query ao servidor TAP - Versão corrigida"""
        try:
            # Coletar parâmetros
            ra_min = float(self.ra_min.get())
            ra_max = float(self.ra_max.get())
            dec_min = float(self.dec_min.get())
            dec_max = float(self.dec_max.get())
            dist_min = float(self.dist_min.get())
            dist_max = float(self.dist_max.get())
            mag_limit = float(self.mag_limit.get())
            parallax_snr = float(self.parallax_snr.get())
            max_records = int(self.max_records.get())
            timeout = int(self.timeout_val.get())

            server_name = "ESA" if self.server_choice.get() == "esa" else "ASTRON"
            self.log_gaia(f"Usando servidor: {server_name}")
            self.log_gaia("Preparando query ao Gaia DR3...")

            # Construir query SIMPLIFICADA primeiro para testar
            query = self.build_gaia_query_simple(ra_min, ra_max, dec_min, dec_max,
                                                 dist_min, dist_max, mag_limit,
                                                 parallax_snr, max_records)

            self.log_gaia(f"Query ADQL (simplificada):\n{query}")

            # URL do serviço TAP ASYNC
            if self.server_choice.get() == "astron":
                base_url = "https://gaia.aip.de/tap/async"
                self.log_gaia(f"URL do ASTRON: {base_url}")
            else:
                base_url = "https://gea.esac.esa.int/tap-server/tap/async"
                self.log_gaia(f"URL da ESA: {base_url}")

            # Parâmetros da requisição
            params = {
                'REQUEST': 'doQuery',
                'LANG': 'ADQL',
                'FORMAT': 'csv',
                'PHASE': 'RUN',
                'QUERY': query
            }

            # Headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/xml,application/xml'
            }

            # Fazer requisição POST
            self.log_gaia("Submetendo job...")
            session = self.create_session_with_retry()
            response = session.post(base_url, data=params, headers=headers, timeout=timeout)

            self.log_gaia(f"Status da resposta: {response.status_code}")
            self.log_gaia(f"Cabeçalhos: {dict(response.headers)}")

            # Log primeiro kilobyte da resposta para debug
            response_text = response.text[:1000] if response.text else "RESPOSTA VAZIA"
            self.log_gaia(f"Primeiros 1000 chars da resposta:\n{response_text}")

            if response.status_code not in [200, 201, 202, 303]:
                error_msg = f"Servidor retornou status {response.status_code}"
                if response.text:
                    error_msg += f": {response.text[:500]}"
                raise Exception(error_msg)

            # Processar resposta baseado no servidor
            if self.server_choice.get() == "astron":
                # ASTRON - pode retornar XML ou redirecionamento
                self.log_gaia("Processando resposta do ASTRON...")

                # Verificar se há cabeçalho Location
                if 'Location' in response.headers:
                    location = response.headers['Location']
                    self.job_id = location.split('/')[-1]
                    self.job_status = "PENDING"

                    self.log_gaia(f"✅ Redirecionamento recebido!")
                    self.log_gaia(f"Location: {location}")
                    self.log_gaia(f"Job ID (do Location): {self.job_id}")
                else:
                    # Tentar extrair do XML/texto
                    job_info = self.extract_job_info_from_xml(response.text)

                    if job_info and job_info['job_id']:
                        self.job_id = job_info['job_id']
                        self.job_status = job_info['phase']

                        self.log_gaia(f"✅ Job submetido com sucesso ao ASTRON!")
                        self.log_gaia(f"Job ID: {self.job_id}")
                        self.log_gaia(f"Status inicial: {self.job_status}")
                    else:
                        # Talvez a resposta já seja os dados
                        self.log_gaia("Verificando se resposta contém dados CSV...")
                        if 'source_id' in response.text[:200] or 'csv' in response.headers.get('Content-Type', ''):
                            self.process_gaia_csv_data(response.text)
                            return
                        else:
                            raise Exception("Não foi possível extrair Job ID da resposta do ASTRON")

                # Atualizar UI na thread principal
                self.root.after(0, self.update_gaia_job_status_ui)
                self.root.after(0, self.enable_gaia_buttons_after_submit)

                if self.job_status == "COMPLETED":
                    self.log_gaia("Job já está COMPLETED! Baixando resultados...")
                    self.root.after(0, lambda: self.download_gaia_results_astron(
                        job_info.get('results_url') if job_info else None))
                else:
                    self.log_gaia("O job será processado em background. Use 'Verificar Status' para acompanhar.")

            else:
                # ESA
                self.log_gaia("Processando resposta da ESA...")
                if 'Location' in response.headers:
                    location = response.headers['Location']
                    self.job_id = location.split('/')[-1]
                    self.job_status = "PENDING"

                    self.log_gaia(f"✅ Job submetido com sucesso à ESA!")
                    self.log_gaia(f"Job ID: {self.job_id}")
                    self.log_gaia(f"Location: {location}")

                    # Atualizar UI na thread principal
                    self.root.after(0, self.update_gaia_job_status_ui)
                    self.root.after(0, self.enable_gaia_buttons_after_submit)
                else:
                    # Pode ser resposta direta com dados
                    self.log_gaia("Verificando se resposta contém dados diretos...")
                    if 'source_id' in response.text[:200] or 'csv' in response.headers.get('Content-Type', ''):
                        self.process_gaia_csv_data(response.text)
                    else:
                        raise Exception("Job ID não recebido do servidor ESA e resposta não contém dados")

        except Exception as e:
            error_msg = f"❌ Erro ao submeter query: {str(e)}"
            self.log_gaia(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Falha na requisição:\n{str(e)[:500]}"))
            self.root.after(0, self.reset_gaia_ui_after_error)
        finally:
            self.root.after(0, self.gaia_progress.stop)
            self.root.after(0, lambda: self.find_and_enable_button("Executar Query"))

    def build_gaia_query_simple(self, ra_min, ra_max, dec_min, dec_max, dist_min, dist_max, mag_limit, parallax_snr,
                                max_records):
        """Constrói uma query ADQL mais simples e robusta"""
        # Converter distância para paralaxe
        parallax_max = 1000.0 / dist_min if dist_min > 0 else 1000.0
        parallax_min = 1000.0 / dist_max if dist_max > 0 else 0.001

        # Query mais simples para evitar problemas
        query = f"""
        SELECT TOP {max_records}
            source_id, 
            ra, 
            dec, 
            parallax, 
            parallax_error,
            pmra, 
            pmdec, 
            phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 
            ra BETWEEN {ra_min} AND {ra_max}
            AND dec BETWEEN {dec_min} AND {dec_max}
            AND parallax BETWEEN {parallax_min} AND {parallax_max}
            AND parallax > 0
            AND phot_g_mean_mag < {mag_limit}
        """
        return query.strip()

    def enable_gaia_buttons_after_submit(self):
        """Habilita botões após submissão bem-sucedida - USANDO REFERÊNCIAS DIRETAS"""
        try:
            self.log_gaia("Habilitando botões após submissão...")

            # Habilitar botão de verificar status
            if hasattr(self, 'check_btn'):
                self.check_btn.config(state=tk.NORMAL)
                self.log_gaia("✅ Botão 'Verificar Status' habilitado")

            # Botão de salvar permanece desabilitado até termos dados
            if hasattr(self, 'save_btn'):
                self.save_btn.config(state=tk.DISABLED)
                self.log_gaia("Botão 'Salvar CSV' mantido desabilitado (aguardando dados)")

            # Botão de carregar para calculador permanece desabilitado
            if hasattr(self, 'load_calc_btn'):
                self.load_calc_btn.config(state=tk.DISABLED)
                self.log_gaa("Botão 'Carregar para Calculador' mantido desabilitado (aguardando dados)")

            # Atualizar labels de status
            self.update_gaia_job_status_ui()

            self.log_gaia("✅ Processo de habilitação de botões concluído")

        except Exception as e:
            self.log_gaia(f"❌ Erro ao habilitar botões: {str(e)}")

    def create_fallback_check_button(self):
        """Cria um botão de verificar status se não existir"""
        try:
            # Procurar o frame de botões
            for widget in self.gaia_frame.winfo_children():
                if isinstance(widget, ttk.Frame):
                    # Verificar se já tem botão de verificar status
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Button) and "Verificar" in child.cget("text"):
                            return  # Já existe

                    # Criar novo botão
                    check_btn = ttk.Button(widget, text="Verificar Status",
                                           command=self.check_gaia_job_status)
                    check_btn.pack(fill=tk.X, pady=2)
                    self.log_gaia("✅ Botão 'Verificar Status' criado como fallback")
                    break
        except Exception as e:
            self.log_gaia(f"Erro ao criar botão fallback: {e}")

    def update_gaia_job_status_ui(self):
        """Atualiza a UI com o status do job - VERSÃO CORRIGIDA"""
        try:
            status_text = f"Status: {self.job_status}" if self.job_status else "Status: Desconhecido"
            job_id_text = f"Job ID: {self.job_id}" if self.job_id else "Job ID: Nenhum"

            # Usar after para thread safety
            self.root.after(0, lambda: self.gaia_status_label.config(text=status_text))
            self.root.after(0, lambda: self.gaia_job_id_label.config(text=job_id_text))

            self.log_gaia(f"UI atualizada: {status_text}, {job_id_text}")

        except Exception as e:
            self.log_gaia(f"Erro ao atualizar UI: {e}")

    def check_gaia_job_status(self):
        """Verifica o status do job async - VERSÃO CORRIGIDA"""
        if not self.job_id:
            messagebox.showwarning("Aviso", "Nenhum job ativo. Execute uma query primeiro.")
            return

        self.log_gaia(f"=== VERIFICANDO STATUS DO JOB {self.job_id} ===")

        # Desabilitar botão temporariamente para evitar múltiplos cliques
        self.disable_check_button()

        # Iniciar progresso
        self.gaia_progress.start()

        # Executar em thread separada
        thread = threading.Thread(target=self.get_gaia_job_status_thread)
        thread.daemon = True
        thread.start()

    def disable_check_button(self):
        """Desabilita o botão de verificar status"""
        for widget in self.gaia_frame.winfo_children():
            if isinstance(widget, ttk.Button) and "Verificar" in widget.cget("text"):
                widget.config(state=tk.DISABLED)
                break

    def get_gaia_job_status_thread(self):
        """Thread para verificar status do job - VERSÃO CORRIGIDA"""
        try:
            self.log_gaia(f"Consultando status do job {self.job_id}...")

            if self.server_choice.get() == "astron":
                status_url = f"https://gaia.aip.de/tap/async/{self.job_id}"
            else:
                status_url = f"https://gea.esac.esa.int/tap-server/tap/async/{self.job_id}/phase"

            session = self.create_session_with_retry()
            response = session.get(status_url, timeout=30)

            self.log_gaia(f"Resposta do servidor: {response.status_code}")

            if response.status_code == 200:
                if self.server_choice.get() == "astron":
                    job_info = self.extract_job_info_from_xml(response.text)
                    if job_info:
                        self.job_status = job_info['phase']
                        self.log_gaia(f"✅ Status atualizado: {self.job_status}")

                        # Atualizar UI na thread principal
                        self.root.after(0, self.update_gaia_job_status_ui)

                        if self.job_status == "COMPLETED":
                            self.log_gaia("🎉 Job CONCLUÍDO! Baixando resultados...")
                            results_url = job_info.get(
                                'results_url') or f"https://gaia.aip.de/tap/async/{self.job_id}/results/result"
                            self.root.after(0, lambda: self.download_gaia_results_astron(results_url))
                        elif self.job_status == "ERROR":
                            self.log_gaia("❌ Job FALHOU no servidor")
                            self.root.after(0, lambda: messagebox.showerror("Erro", "Job falhou no servidor"))
                            self.root.after(0, self.enable_check_button)
                        elif self.job_status in ["PENDING", "EXECUTING", "QUEUED"]:
                            self.log_gaia(f"Job ainda em processamento: {self.job_status}")
                            self.root.after(0, self.enable_check_button)
                        else:
                            self.root.after(0, self.enable_check_button)
                    else:
                        self.log_gaia("⚠️ Não foi possível extrair informações do job")
                        self.root.after(0, self.enable_check_button)
                else:
                    # ESA
                    self.job_status = response.text.strip()
                    self.log_gaia(f"✅ Status atualizado (ESA): {self.job_status}")

                    self.root.after(0, self.update_gaia_job_status_ui)

                    if self.job_status == "COMPLETED":
                        self.log_gaia("🎉 Job CONCLUÍDO! Baixando resultados...")
                        self.root.after(0, self.download_gaia_results)
                    elif self.job_status == "ERROR":
                        self.log_gaia("❌ Job FALHOU no servidor")
                        self.root.after(0, lambda: messagebox.showerror("Erro", "Job falhou no servidor"))
                        self.root.after(0, self.enable_check_button)
                    else:
                        self.root.after(0, self.enable_check_button)
            else:
                error_msg = f"❌ Erro ao verificar status: {response.status_code}"
                if response.text:
                    error_msg += f"\n{response.text[:200]}"
                self.log_gaia(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Erro", error_msg))
                self.root.after(0, self.enable_check_button)

        except Exception as e:
            error_msg = f"❌ Erro ao verificar status: {str(e)}"
            self.log_gaia(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Erro", error_msg))
            self.root.after(0, self.enable_check_button)
        finally:
            self.root.after(0, self.gaia_progress.stop)

    def enable_check_button(self):
        """Habilita o botão de verificar status"""
        self.root.after(0, lambda: self._enable_check_button_impl())

    def _enable_check_button_impl(self):
        """Implementação para habilitar botão na thread principal"""
        for widget in self.gaia_frame.winfo_children():
            if isinstance(widget, ttk.Button) and "Verificar" in widget.cget("text"):
                widget.config(state=tk.NORMAL)
                self.log_gaia("✅ Botão 'Verificar Status' re-habilitado")
                break

    def download_gaia_results_astron(self, results_url=None):
        """Baixa resultados do ASTRON"""
        try:
            if not results_url:
                results_url = f"https://gaia.aip.de/tap/async/{self.job_id}/results/result"

            self.log_gaia(f"Baixando resultados de: {results_url}")
            timeout = int(self.timeout_val.get())
            session = self.create_session_with_retry()
            response = session.get(results_url, timeout=timeout)

            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                self.log_gaia(f"Content-Type: {content_type}")

                # Verificar se é CSV ou XML
                if 'csv' in content_type.lower() or response.text.strip().startswith('source_id'):
                    self.process_gaia_csv_data(response.text)
                else:
                    # Pode ser XML com link para resultados
                    self.log_gaia("Resposta não é CSV, verificando se é XML...")
                    job_info = self.extract_job_info_from_xml(response.text)
                    if job_info and job_info['results_url']:
                        self.log_gaia(f"Encontrada URL de resultados: {job_info['results_url']}")
                        self.download_gaia_results_astron(job_info['results_url'])
                    else:
                        # Tentar processar como CSV mesmo assim
                        try:
                            self.log_gaia("Tentando processar como CSV...")
                            self.process_gaia_csv_data(response.text)
                        except Exception as e2:
                            self.log_gaia(f"Não conseguiu processar como CSV: {e2}")
                            raise Exception(f"Formato não reconhecido: {content_type}. Conteúdo: {response.text[:200]}")
            else:
                error_msg = f"❌ Erro ao baixar resultados: {response.status_code}"
                if response.text:
                    error_msg += f"\nResposta: {response.text[:200]}"
                self.log_gaia(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Erro", error_msg))

        except Exception as e:
            error_msg = f"❌ Erro ao baixar resultados: {str(e)}"
            self.log_gaia(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Erro", error_msg))

    def download_gaia_results(self):
        """Baixa resultados da ESA"""
        try:
            results_url = f"https://gea.esac.esa.int/tap-server/tap/async/{self.job_id}/results/result"
            self.log_gaia(f"Baixando resultados da ESA: {results_url}")

            timeout = int(self.timeout_val.get())
            session = self.create_session_with_retry()
            response = session.get(results_url, timeout=timeout)

            if response.status_code == 200:
                self.process_gaia_csv_data(response.text)
            else:
                error_msg = f"❌ Erro ao baixar resultados: {response.status_code}"
                if response.text:
                    error_msg += f"\nResposta: {response.text[:200]}"
                self.log_gaia(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Erro", error_msg))

        except Exception as e:
            error_msg = f"❌ Erro ao baixar resultados: {str(e)}"
            self.log_gaia(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Erro", error_msg))

    def process_gaia_csv_data(self, csv_text):
        """Processa dados CSV do Gaia - VERSÃO CORRIGIDA"""
        try:
            self.log_gaia("Processando dados CSV...")

            # Verificar se o texto começa com dados CSV
            if not csv_text.strip():
                raise Exception("Resposta vazia do servidor")

            # Verificar se é realmente CSV
            if 'source_id' not in csv_text[:1000]:
                self.log_gaia(f"Primeiros 500 chars: {csv_text[:500]}")
                # Pode ser XML de erro
                if 'error' in csv_text.lower() or 'exception' in csv_text.lower():
                    raise Exception(f"Servidor retornou erro: {csv_text[:500]}")
                else:
                    raise Exception("Resposta não contém dados CSV esperados")

            self.gaia_data = pd.read_csv(StringIO(csv_text))

            # Verificar se temos dados
            if self.gaia_data.empty:
                raise Exception("DataFrame vazio retornado pelo servidor")

            self.log_gaia(f"✅ Query concluída com sucesso!")
            self.log_gaia(f"Encontradas {len(self.gaia_data)} estrelas")
            self.log_gaia(f"Colunas: {', '.join(self.gaia_data.columns.tolist()[:10])}...")

            # Calcular distância se tiver paralaxe
            if 'parallax' in self.gaia_data.columns:
                mask = self.gaia_data['parallax'].notna() & (self.gaia_data['parallax'] > 0)
                self.gaia_data.loc[mask, 'distance_pc'] = 1000.0 / self.gaia_data.loc[mask, 'parallax']
                self.log_gaia(f"Distância calculada para {mask.sum()} estrelas")

            # Atualizar status
            self.job_status = "COMPLETED"

            # Atualizar UI na thread principal
            self.root.after(0, self.update_gaia_job_status_ui)
            self.root.after(0, self.enable_save_and_load_buttons)

        except Exception as e:
            error_msg = f"❌ Erro ao processar dados CSV: {str(e)}"
            self.log_gaia(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Erro", error_msg))

    def enable_save_and_load_buttons(self):
        """Habilita botões de salvar e carregar quando dados estão prontos - USANDO REFERÊNCIAS DIRETAS"""
        try:
            self.log_gaia("Habilitando botões de salvar e carregar...")

            # Habilitar botão de salvar
            if hasattr(self, 'save_btn'):
                self.save_btn.config(state=tk.NORMAL)
                self.log_gaia("✅ Botão 'Salvar CSV' HABILITADO")

            # Habilitar botão de carregar para calculador
            if hasattr(self, 'load_calc_btn'):
                self.load_calc_btn.config(state=tk.NORMAL)
                self.log_gaia("✅ Botão 'Carregar para Calculador' HABILITADO")

        except Exception as e:
            self.log_gaia(f"❌ Erro ao habilitar botões: {str(e)}")

    def disable_check_button(self):
        """Desabilita o botão de verificar status - USANDO REFERÊNCIA DIRETA"""
        if hasattr(self, 'check_btn'):
            self.check_btn.config(state=tk.DISABLED)

    def _enable_check_button_impl(self):
        """Implementação para habilitar botão na thread principal - USANDO REFERÊNCIA DIRETA"""
        if hasattr(self, 'check_btn'):
            self.check_btn.config(state=tk.NORMAL)
            self.log_gaia("✅ Botão 'Verificar Status' re-habilitado")

    def enable_gaia_save_button(self):
        """Habilita botão de salvar"""
        save_btn = self.find_and_enable_button("Salvar CSV")
        if save_btn:
            save_btn.config(state=tk.NORMAL)
            self.log_gaia("Botão 'Salvar CSV' habilitado")

    def enable_load_to_calculador_button(self):
        """Habilita botão de carregar para calculador"""
        load_btn = self.find_and_enable_button("Carregar para Calculador")
        if load_btn:
            load_btn.config(state=tk.NORMAL)
            self.log_gaia("Botão 'Carregar para Calculador' habilitado")

    def reset_gaia_ui(self):
        """Reseta a UI da aba Gaia"""
        self.gaia_progress.stop()
        # Re-habilitar todos os botões
        for widget in self.gaia_frame.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.config(state=tk.NORMAL)

        self.gaia_status_label.config(text="Pronto para executar")
        self.gaia_job_id_label.config(text="Job ID: Nenhum")
        self.job_id = None
        self.job_status = None

    def reset_gaia_ui_after_error(self):
        """Reseta a UI da aba Gaia após erro"""
        self.gaia_progress.stop()
        # Re-habilitar botão de execução
        exec_btn = self.find_and_enable_button("Executar Query")
        if exec_btn:
            exec_btn.config(state=tk.NORMAL)

        self.gaia_status_label.config(text="Erro - Tente novamente")
        # Manter job_id se existir para debugging
        if self.job_id:
            self.gaia_job_id_label.config(text=f"Job ID (erro): {self.job_id}")

    def save_gaia_data(self):
        """Salva os dados Gaia em arquivo CSV"""
        if self.gaia_data is None or self.gaia_data.empty:
            messagebox.showwarning("Aviso", "Nenhum dado para salvar")
            return

        # Sugerir nome baseado nos parâmetros
        ra_str = f"{float(self.ra_min.get()):.1f}-{float(self.ra_max.get()):.1f}"
        dec_str = f"{float(self.dec_min.get()):.1f}-{float(self.dec_max.get()):.1f}"
        suggested_name = f"gaia_ra{ra_str}_dec{dec_str}.csv"

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=suggested_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Salvar dados Gaia"
        )

        if filename:
            try:
                self.gaia_data.to_csv(filename, index=False)
                self.log_gaia(f"✅ Dados salvos em: {filename}")
                messagebox.showinfo("Sucesso", f"Dados salvos em:\n{filename}")
            except Exception as e:
                self.log_gaia(f"❌ Erro ao salvar: {e}")
                messagebox.showerror("Erro", f"Falha ao salvar:\n{e}")

    def clear_gaia_fields(self):
        """Limpa campos da aba Gaia"""
        self.ra_min.delete(0, tk.END)
        self.ra_min.insert(0, "166.0")
        self.ra_max.delete(0, tk.END)
        self.ra_max.insert(0, "170.0")
        self.dec_min.delete(0, tk.END)
        self.dec_min.insert(0, "-62.0")
        self.dec_max.delete(0, tk.END)
        self.dec_max.insert(0, "-59.0")
        self.dist_min.delete(0, tk.END)
        self.dist_min.insert(0, "2000.0")
        self.dist_max.delete(0, tk.END)
        self.dist_max.insert(0, "3200.0")
        self.mag_limit.delete(0, tk.END)
        self.mag_limit.insert(0, "18.0")
        self.parallax_snr.delete(0, tk.END)
        self.parallax_snr.insert(0, "5.0")
        self.max_records.delete(0, tk.END)
        self.max_records.insert(0, "10000")
        self.timeout_val.delete(0, tk.END)
        self.timeout_val.insert(0, "120")
        self.gaia_log_text.delete(1.0, tk.END)
        self.gaia_data = None
        self.reset_gaia_ui()
        self.log_gaia("Campos limpos e resetados")

    def load_gaia_to_calculador(self):
        """Carrega dados Gaia para o calculador de colunas"""
        if self.gaia_data is None or self.gaia_data.empty:
            messagebox.showwarning("Aviso", "Nenhum dado Gaia carregado")
            return

        try:
            # Carregar para calculador
            self.calc_df = self.gaia_data.copy()

            # Atualizar entrada de arquivo no calculador
            self.calc_file_entry.delete(0, tk.END)
            self.calc_file_entry.insert(0, "Dados Gaia (memória)")

            # Atualizar treeview
            self._atualizar_calc_treeview()

            # Analisar colunas disponíveis
            self._analisar_colunas_disponiveis()

            self.calc_log(f"✅ Dados Gaia carregados para calculador: {len(self.calc_df)} estrelas")
            self.calc_file_info.config(text=f"{len(self.calc_df)} estrelas, {len(self.calc_df.columns)} colunas (Gaia)")
            self.calc_status_var.set(f"🟢 Dados Gaia carregados: {len(self.calc_df)} estrelas")

            # Mostrar estatísticas
            self._mostrar_estatisticas_calculadas()

            # Ir para aba do calculador
            self.notebook.select(1)  # Índice 1 = Calculador de Colunas

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar dados para calculador: {str(e)}")

    # ============================================================
    # FUNÇÕES ANÁLISE GRAVITACIONAL (da aba 4)
    # ============================================================
    def browse_file(self):
        """Abre diálogo para selecionar arquivo"""
        filename = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)

    def load_data(self):
        """Carrega dados de arquivo CSV"""
        filename = self.file_entry.get()
        if not filename:
            messagebox.showerror("Erro", "Selecione um arquivo")
            return

        try:
            self.log_general(f"Carregando {filename}")
            self.status.set("Carregando dados...")

            self.df_estrelas = pd.read_csv(filename)

            # Verificar colunas mínimas
            required = ['ra', 'dec', 'distance_pc']
            missing = [c for c in required if c not in self.df_estrelas.columns]
            if missing:
                messagebox.showerror("Erro", f"Colunas faltando: {missing}")
                return

            # Começar com todos os dados
            self.df_filtrado = self.df_estrelas.copy()
            total_antes = len(self.df_filtrado)

            # Aplicar filtro de distância
            if self.filtro_ativo.get():
                try:
                    min_d = float(self.min_dist.get())
                    max_d = float(self.max_dist.get())

                    if min_d < 0 or max_d <= 0 or min_d >= max_d:
                        raise ValueError("Distâncias inválidas")

                    self.df_filtrado = self.df_filtrado[
                        (self.df_filtrado['distance_pc'] >= min_d) &
                        (self.df_filtrado['distance_pc'] <= max_d)
                        ].copy()

                    self.update_stats(f"✅ Filtro de distância: {min_d} a {max_d} pc")
                except Exception as e:
                    self.update_stats(f"⚠️ Erro no filtro de distância: {e}")
                    self.filtro_ativo.set(False)

            # Aplicar filtro de magnitude
            if self.filtro_mag_ativo.get():
                try:
                    min_mag = float(self.min_mag.get())
                    max_mag = float(self.max_mag.get())

                    mag_columns = ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'mag']
                    available_mag_cols = [col for col in mag_columns if col in self.df_filtrado.columns]

                    if available_mag_cols:
                        mag_col = available_mag_cols[0]
                        antes_mag = len(self.df_filtrado)
                        self.df_filtrado = self.df_filtrado[
                            (self.df_filtrado[mag_col] >= min_mag) &
                            (self.df_filtrado[mag_col] <= max_mag)
                            ].copy()
                        depois_mag = len(self.df_filtrado)

                        self.update_stats(f"✅ Filtro de magnitude: {min_mag} a {max_mag} ({mag_col})")
                        self.update_stats(f"   Removidas: {antes_mag - depois_mag} estrelas")

                except Exception as e:
                    self.update_stats(f"⚠️ Erro no filtro de magnitude: {e}")
                    self.filtro_mag_ativo.set(False)

            # Estatísticas finais
            total_depois = len(self.df_filtrado)
            self.stats_text.delete(1.0, tk.END)
            self.update_stats(f"✅ DADOS CARREGADOS COM SUCESSO")
            self.update_stats(f"Arquivo: {filename.split('/')[-1]}")
            self.update_stats(f"Estrelas inicial: {total_antes}")
            self.update_stats(f"Estrelas final: {total_depois}")
            self.update_stats(f"Total removidas: {total_antes - total_depois}")
            self.update_stats(f"Porcentagem mantida: {total_depois / total_antes * 100:.1f}%")
            self.update_stats("")
            self.update_stats("📊 ESTATÍSTICAS:")
            self.update_stats(f"   RA: {self.df_filtrado['ra'].min():.1f} a {self.df_filtrado['ra'].max():.1f}°")
            self.update_stats(f"   Dec: {self.df_filtrado['dec'].min():.1f} a {self.df_filtrado['dec'].max():.1f}°")
            self.update_stats(
                f"   Distância: {self.df_filtrado['distance_pc'].min():.1f} a {self.df_filtrado['distance_pc'].max():.1f} pc")

            if total_depois == 0:
                messagebox.showwarning("Aviso", "Nenhuma estrela após filtros!")
                self.df_filtrado = self.df_estrelas.copy()

            self.log_general(f"Dados carregados: {len(self.df_filtrado)} estrelas")
            self.status.set(f"Dados carregados - {len(self.df_filtrado)} estrelas")

            # Limpar resultados anteriores
            self.analise_grav = None
            self.grupos_gravitacionais = None
            self.grav_results_text.delete(1.0, tk.END)

        except Exception as e:
            self.update_stats(f"❌ Erro ao carregar: {str(e)}")
            messagebox.showerror("Erro", f"Falha ao carregar:\n{str(e)}")
            self.status.set("Erro ao carregar")

    def analisar_gravidade(self):
        """Executa análise gravitacional"""
        if self.df_filtrado is None or len(self.df_filtrado) == 0:
            messagebox.showerror("Erro", "Carregue dados primeiro")
            return

        try:
            raio = float(self.raio_grav.get())
            forca_min = float(self.forca_min.get())
            min_estrelas = int(self.min_estrelas_grav.get())

            self.log_general(f"Analisando gravidade: raio={raio} pc, força_min={forca_min:.1e} N")
            self.status.set("Calculando forças gravitacionais...")

            # Criar analisador gravitacional
            self.analise_grav = AnaliseGravitacional(self.df_filtrado)

            # Calcular conexões
            conexoes = self.analise_grav.calcular_conexoes_gravitacionais(raio, forca_min)

            if len(conexoes) == 0:
                self.log_general("⚠️ Nenhuma conexão gravitacional significativa encontrada")
                messagebox.showinfo("Resultado", "Nenhuma conexão gravitacional significativa")
                return

            self.log_general(f"✅ {len(conexoes)} conexões gravitacionais encontradas")

            # Identificar grupos
            self.grupos_gravitacionais = self.analise_grav.identificar_grupos_gravitacionais(min_estrelas)

            # Mostrar resultados
            self.grav_results_text.delete(1.0, tk.END)
            self.grav_results_text.insert(tk.END, f"✅ ANÁLISE GRAVITACIONAL CONCLUÍDA\n")
            self.grav_results_text.insert(tk.END, "=" * 60 + "\n\n")
            self.grav_results_text.insert(tk.END, f"Conexões encontradas: {len(conexoes)}\n")
            self.grav_results_text.insert(tk.END, f"Grupos identificados: {len(self.grupos_gravitacionais)}\n\n")

            # Estatísticas das conexões
            self.grav_results_text.insert(tk.END, "📈 ESTATÍSTICAS DAS CONEXÕES:\n")
            self.grav_results_text.insert(tk.END, f"   Distância média: {conexoes['distancia_pc'].mean():.2f} pc\n")
            self.grav_results_text.insert(tk.END, f"   Força média: {conexoes['forca_N'].mean():.2e} N\n")
            self.grav_results_text.insert(tk.END, f"   Força máxima: {conexoes['forca_N'].max():.2e} N\n")
            self.grav_results_text.insert(tk.END, f"   Força mínima: {conexoes['forca_N'].min():.2e} N\n\n")

            # Grupos gravitacionais
            if self.grupos_gravitacionais:
                self.grav_results_text.insert(tk.END, "🪐 GRUPOS GRAVITACIONAIS IDENTIFICADOS:\n")
                self.grav_results_text.insert(tk.END, "-" * 50 + "\n\n")

                for grupo in self.grupos_gravitacionais[:10]:
                    self.grav_results_text.insert(tk.END, f"🔗 GRUPO {grupo['id']}:\n")
                    self.grav_results_text.insert(tk.END, f"   Estrelas: {grupo['n_estrelas']}\n")
                    self.grav_results_text.insert(tk.END, f"   Massa total: {grupo['massa_total_Msun']:.1f} M☉\n")
                    self.grav_results_text.insert(tk.END, f"   Força total: {grupo['forca_total_N']:.2e} N\n")
                    self.grav_results_text.insert(tk.END, f"   Distância média: {grupo['distancia_media_pc']:.2f} pc\n")
                    self.grav_results_text.insert(tk.END,
                                                  f"   Centro: RA={grupo['centro_ra']:.2f}°, Dec={grupo['centro_dec']:.2f}°, Dist={grupo['centro_dist_pc']:.1f} pc\n\n")

            self.log_general(f"✅ Análise gravitacional concluída: {len(self.grupos_gravitacionais)} grupos")
            self.status.set("Análise gravitacional concluída")

            # Ir para aba de resultados
            self.notebook.select(5)  # Índice 5 = Logs e Resultados

        except Exception as e:
            self.log_general(f"❌ Erro análise gravitacional: {str(e)}")
            messagebox.showerror("Erro", f"Falha na análise:\n{str(e)}")
            self.status.set("Erro na análise")

    # ============================================================
    # FUNÇÕES VISUALIZAÇÃO 3D (da aba 5)
    # ============================================================
    def reset_aspect_ratio(self):
        """Reseta o aspect ratio para 1:1:1"""
        self.aspect_x.set(1.0)
        self.aspect_y.set(1.0)
        self.aspect_z.set(1.0)
        self.update_viz_info("Aspect ratio resetado para 1:1:1")

    def ease_in_out_sine(self, x):
        """Função de easing para movimento mais suave"""
        return -(np.cos(np.pi * x) - 1) / 2

    def visualizar_com_ligacoes(self):
        """Visualização 3D com ligações gravitacionais"""
        if self.df_filtrado is None:
            messagebox.showerror("Erro", "Carregue dados primeiro")
            return

        if self.analise_grav is None or self.analise_grav.resultados is None:
            messagebox.showwarning("Aviso", "Execute análise gravitacional primeiro")
            return

        try:
            self.update_viz_info("Gerando visualização 3D com ligações gravitacionais...")

            # Tema escuro
            template_dark = go.layout.Template()
            template_dark.layout.plot_bgcolor = 'rgba(10, 10, 40, 1)'
            template_dark.layout.paper_bgcolor = 'rgba(10, 10, 40, 1)'
            template_dark.layout.font.color = 'white'

            fig = go.Figure()

            # Calcular tamanhos baseados na magnitude
            if 'phot_g_mean_mag' in self.df_filtrado.columns:
                tamanhos = np.array([calcular_tamanho_magnitude(mag)
                                     for mag in self.df_filtrado['phot_g_mean_mag'].values])
            else:
                tamanhos = np.full(len(self.df_filtrado), 1.0)

            # Plotar todas as estrelas
            fig.add_trace(go.Scatter3d(
                x=self.df_filtrado['ra'],
                y=self.df_filtrado['dec'],
                z=self.df_filtrado['distance_pc'],
                mode='markers',
                marker=dict(
                    size=tamanhos * 1.0,
                    color='rgba(255, 255, 255, 0.4)',
                    line=dict(width=0)
                ),
                name='Estrelas',
                hoverinfo='skip'
            ))

            # Plotar ligações gravitacionais
            conexoes = self.analise_grav.resultados

            # Agrupar conexões por força
            conexoes['forca_categoria'] = pd.cut(conexoes['forca_log10'],
                                                 bins=[16, 17, 18, 19, 20],
                                                 labels=['Fraca', 'Média', 'Forte', 'Muito Forte'])

            cores_ligacao = {
                'Fraca': 'rgba(100, 100, 100, 0.2)',
                'Média': 'rgba(200, 200, 200, 0.3)',
                'Forte': 'rgba(200, 100, 100, 0.4)',
                'Muito Forte': 'rgba(250, 100, 100, 0.5)'
            }

            espessuras = {'Fraca': 1.0, 'Média': 1.3, 'Forte': 1.6, 'Muito Forte': 1.9}

            for categoria in ['Fraca', 'Média', 'Forte', 'Muito Forte']:
                conexoes_cat = conexoes[conexoes['forca_categoria'] == categoria]

                if len(conexoes_cat) == 0:
                    continue

                # Criar arrays para as linhas
                x_lines = []
                y_lines = []
                z_lines = []

                for _, conexao in conexoes_cat.iterrows():
                    i = conexao['estrela1']
                    j = conexao['estrela2']

                    x_lines.extend([self.df_filtrado.iloc[i]['ra'],
                                    self.df_filtrado.iloc[j]['ra'], None])
                    y_lines.extend([self.df_filtrado.iloc[i]['dec'],
                                    self.df_filtrado.iloc[j]['dec'], None])
                    z_lines.extend([self.df_filtrado.iloc[i]['distance_pc'],
                                    self.df_filtrado.iloc[j]['distance_pc'], None])

                fig.add_trace(go.Scatter3d(
                    x=x_lines,
                    y=y_lines,
                    z=z_lines,
                    mode='lines',
                    line=dict(
                        color=cores_ligacao[categoria],
                        width=espessuras[categoria]
                    ),
                    name=f'Ligações {categoria}',
                    hoverinfo='skip'
                ))

            # Plotar grupos gravitacionais
            if self.grupos_gravitacionais:
                colors = ['#FFFFFF', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2']

                for idx, grupo in enumerate(self.grupos_gravitacionais[:5]):
                    estrelas = grupo['estrelas']

                    if 'phot_g_mean_mag' in self.df_filtrado.columns:
                        agl_tamanhos = np.array([calcular_tamanho_magnitude(mag)
                                                 for mag in self.df_filtrado.iloc[estrelas]['phot_g_mean_mag'].values])
                    else:
                        agl_tamanhos = np.full(len(estrelas), 3.0)

                    fig.add_trace(go.Scatter3d(
                        x=self.df_filtrado.iloc[estrelas]['ra'],
                        y=self.df_filtrado.iloc[estrelas]['dec'],
                        z=self.df_filtrado.iloc[estrelas]['distance_pc'],
                        mode='markers',
                        marker=dict(
                            size=agl_tamanhos * 1.0,
                            color=colors[idx % len(colors)],
                            opacity=1.0,
                            line=dict(width=1, color='white')
                        ),
                        name=f'Grupo Grav {grupo["id"]}',
                        hovertemplate='Grupo Gravitacional %{text}',
                        text=[f'Força: {grupo["forca_total_N"]:.1e} N<br>'
                              f'Massa: {grupo["massa_total_Msun"]:.1f} M☉']
                    ))

            # Configuração da cena 3D
            aspect_x = self.aspect_x.get()
            aspect_y = self.aspect_y.get()
            aspect_z = self.aspect_z.get()

            camera = dict(
                eye=dict(x=1.8, y=1.8, z=1.8),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0)
            )

            # ADICIONAR ROTAÇÃO AUTOMÁTICA
            if self.rotacao_ativa.get():
                self.update_viz_info("Configurando rotação automática...")

                n_frames = 720
                duracao_total = self.duracao_rotacao.get() * 1000
                velocidade = self.velocidade_rotacao.get()
                eixo = self.eixo_rotacao.get()

                frame_duration = duracao_total / n_frames
                frames = []

                for i in range(n_frames):
                    progress = i / n_frames
                    eased_progress = self.ease_in_out_sine(progress)
                    angle = 2 * np.pi * eased_progress * velocidade

                    if eixo == "x":
                        eye = dict(x=1.8,
                                   y=1.8 * np.cos(angle),
                                   z=1.8 * np.sin(angle))
                    elif eixo == "y":
                        eye = dict(x=1.8 * np.cos(angle),
                                   y=1.8,
                                   z=1.8 * np.sin(angle))
                    elif eixo == "z":
                        eye = dict(x=1.8 * np.cos(angle),
                                   y=1.8 * np.sin(angle),
                                   z=1.8)
                    else:  # "auto"
                        theta = 2 * np.pi * eased_progress * velocidade
                        phi = np.pi / 4 + 0.2 * np.sin(progress * np.pi * 2)
                        distance = 2.0 + 0.3 * np.sin(progress * np.pi * 4)

                        eye = dict(x=distance * np.sin(phi) * np.cos(theta),
                                   y=distance * np.sin(phi) * np.sin(theta),
                                   z=distance * np.cos(phi))

                    camera_frame = dict(
                        eye=eye,
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0)
                    )

                    frames.append(go.Frame(
                        name=f'frame{i}',
                        layout=dict(
                            scene_camera=camera_frame
                        )
                    ))

                # Configuração do slider
                sliders_dict = dict(
                    active=0,
                    steps=[],
                    x=0.1,
                    y=0,
                    len=0.9,
                    xanchor="left",
                    yanchor="top",
                    transition=dict(duration=300, easing='cubic-in-out'),
                    currentvalue=dict(
                        font=dict(size=12),
                        prefix="Ângulo: ",
                        visible=True,
                        xanchor="right"
                    ),
                    pad=dict(t=50, b=10)
                )

                for i in range(0, n_frames, max(1, n_frames // 72)):
                    angle_degrees = int(360 * i / n_frames)
                    slider_step = dict(
                        method="animate",
                        args=[
                            [f'frame{i}'],
                            dict(
                                mode="immediate",
                                frame=dict(duration=300, redraw=True),
                                transition=dict(duration=300, easing='cubic-in-out')
                            )
                        ],
                        label=f'{angle_degrees}°'
                    )
                    sliders_dict['steps'].append(slider_step)

                # Botões de controle
                updatemenus = [
                    dict(
                        type="buttons",
                        showactive=True,
                        buttons=[
                            dict(
                                label="▶️ Play",
                                method="animate",
                                args=[
                                    None,
                                    dict(
                                        frame=dict(duration=frame_duration, redraw=True),
                                        fromcurrent=True,
                                        transition=dict(duration=frame_duration / 2, easing='cubic-in-out')
                                    )
                                ]
                            ),
                            dict(
                                label="⏸️ Pause",
                                method="animate",
                                args=[
                                    [None],
                                    dict(
                                        frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0)
                                    )
                                ]
                            ),
                            dict(
                                label="⏭️ 5 frames",
                                method="animate",
                                args=[
                                    [None],
                                    dict(
                                        frame=dict(duration=frame_duration * 5, redraw=True),
                                        mode="next",
                                        transition=dict(duration=frame_duration * 2, easing='cubic-in-out')
                                    )
                                ]
                            ),
                            dict(
                                label="⏮️ 5 frames",
                                method="animate",
                                args=[
                                    [None],
                                    dict(
                                        frame=dict(duration=frame_duration * 5, redraw=True),
                                        mode="previous",
                                        transition=dict(duration=frame_duration * 2, easing='cubic-in-out')
                                    )
                                ]
                            ),
                            dict(
                                label="↻ Reiniciar",
                                method="animate",
                                args=[
                                    ['frame0'],
                                    dict(
                                        mode="immediate",
                                        frame=dict(duration=1000, redraw=True),
                                        transition=dict(duration=1000, easing='elastic-out')
                                    )
                                ]
                            )
                        ],
                        x=0.1,
                        xanchor="right",
                        y=0,
                        yanchor="top",
                        pad=dict(t=0, r=10),
                        bgcolor='rgba(0, 0, 40, 0.8)',
                        bordercolor='white',
                        borderwidth=1
                    )
                ]

                fig.frames = frames

                fig.update_layout(
                    title=dict(
                        text='🌌 REDE GRAVITACIONAL 3D (Rotação Automática)',
                        font=dict(size=15, color='white', family='Arial Black')
                    ),
                    scene=dict(
                        xaxis_title='RA (graus)',
                        yaxis_title='Dec (graus)',
                        zaxis_title='Distância (pc)',
                        xaxis=dict(
                            autorange='reversed',
                            gridcolor='rgba(100, 100, 150, 0.3)',
                            backgroundcolor='rgba(0, 0, 20, 0.1)'
                        ),
                        yaxis=dict(
                            autorange='reversed',
                            gridcolor='rgba(100, 100, 150, 0.3)',
                            backgroundcolor='rgba(0, 0, 20, 0.1)'
                        ),
                        zaxis=dict(
                            gridcolor='rgba(100, 100, 150, 0.3)',
                            backgroundcolor='rgba(0, 0, 20, 0.1)'
                        ),
                        aspectmode='manual',
                        aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z),
                        camera=camera
                    ),
                    width=1400,
                    height=800,
                    showlegend=True,
                    template=template_dark,
                    legend=dict(
                        bgcolor='rgba(0, 0, 40, 0.8)',
                        bordercolor='white',
                        borderwidth=1,
                        font=dict(color='white'),
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    updatemenus=updatemenus,
                    transition=dict(
                        duration=frame_duration,
                        easing='cubic-in-out'
                    )
                )

                fig.update_layout(sliders=[sliders_dict])

                self.update_viz_info(f"✅ Rotação automática configurada")
                self.update_viz_info(f"   Velocidade: {velocidade:.1f}x")
                self.update_viz_info(f"   Eixo: {eixo}")
                self.update_viz_info(f"   Duração: {self.duracao_rotacao.get()} segundos")
            else:
                # Layout sem animação
                fig.update_layout(
                    title=dict(
                        text='🌌 REDE GRAVITACIONAL 3D',
                        font=dict(size=15, color='white', family='Arial Black')
                    ),
                    scene=dict(
                        xaxis_title='RA (graus)',
                        yaxis_title='Dec (graus)',
                        zaxis_title='Distância (pc)',
                        xaxis=dict(
                            autorange='reversed',
                            gridcolor='rgba(100, 100, 150, 0.3)',
                            backgroundcolor='rgba(0, 0, 20, 0.1)'
                        ),
                        yaxis=dict(
                            autorange='reversed',
                            gridcolor='rgba(100, 100, 150, 0.3)',
                            backgroundcolor='rgba(0, 0, 20, 0.1)'
                        ),
                        zaxis=dict(
                            gridcolor='rgba(100, 100, 150, 0.3)',
                            backgroundcolor='rgba(0, 0, 20, 0.1)'
                        ),
                        aspectmode='manual',
                        aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z),
                        camera=camera
                    ),
                    width=1400,
                    height=800,
                    showlegend=True,
                    template=template_dark,
                    legend=dict(
                        bgcolor='rgba(0, 0, 40, 0.8)',
                        bordercolor='white',
                        borderwidth=1,
                        font=dict(color='white'),
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                self.update_viz_info("ℹ️ Rotação automática desativada")

            # Adicionar informações
            fig.add_annotation(
                x=0.02, y=0.02,
                xref="paper", yref="paper",
                text=f"Estrelas: {len(self.df_filtrado)}<br>"
                     f"Conexões: {len(conexoes)}<br>"
                     f"Grupos: {len(self.grupos_gravitacionais) if self.grupos_gravitacionais else 0}<br>"
                     f"Aspect Ratio: {aspect_x:.1f}:{aspect_y:.1f}:{aspect_z:.1f}",
                showarrow=False,
                font=dict(size=12, color='white'),
                bgcolor="rgba(0, 0, 30, 0.7)",
                bordercolor="white",
                borderwidth=1
            )

            fig.show()
            self.update_viz_info("✅ Visualização gravitacional 3D gerada")
            self.log_general("Visualização 3D gerada com sucesso")

        except Exception as e:
            self.update_viz_info(f"❌ Erro na visualização: {str(e)}")
            self.log_general(f"Erro na visualização 3D: {e}")

    # ============================================================
    # FUNÇÕES UTILITÁRIAS
    # ============================================================
    def exportar_resultados(self):
        """Exporta resultados gravitacionais"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Exportar resultados gravitacionais
            if self.analise_grav and self.analise_grav.resultados is not None:
                self.analise_grav.resultados.to_csv(f"conexoes_gravitacionais_{timestamp}.csv", index=False)
                self.log_general(f"✅ Conexões gravitacionais exportadas: conexoes_gravitacionais_{timestamp}.csv")

            if self.grupos_gravitacionais:
                dados_grav = []
                for grupo in self.grupos_gravitacionais:
                    dados_grav.append({
                        'grupo_id': grupo['id'],
                        'metodo': 'GRAVIDADE',
                        'n_estrelas': grupo['n_estrelas'],
                        'massa_total_Msun': grupo['massa_total_Msun'],
                        'forca_total_N': grupo['forca_total_N'],
                        'distancia_media_pc': grupo['distancia_media_pc'],
                        'densidade_gravitacional': grupo['densidade_gravitacional'],
                        'centro_ra': grupo['centro_ra'],
                        'centro_dec': grupo['centro_dec'],
                        'centro_dist_pc': grupo['centro_dist_pc']
                    })

                df_grav = pd.DataFrame(dados_grav)
                df_grav.to_csv(f"grupos_gravitacionais_{timestamp}.csv", index=False)
                self.log_general(f"✅ Grupos gravitacionais exportados: grupos_gravitacionais_{timestamp}.csv")

            messagebox.showinfo("Sucesso", "Resultados exportados com sucesso!")

        except Exception as e:
            self.log_general(f"❌ Erro ao exportar: {str(e)}")

    def copy_to_clipboard(self, text_widget):
        """Copia texto para a área de transferência"""
        text = text_widget.get(1.0, tk.END)
        if text.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.log_general("Texto copiado para a área de transferência.")

    def export_grav_results(self):
        """Exporta resultados gravitacionais para arquivo"""
        text = self.grav_results_text.get(1.0, tk.END)
        if not text.strip():
            messagebox.showwarning("Aviso", "Nenhum resultado para exportar")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Exportar Resultados Gravitacionais"
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.log_general(f"✅ Resultados exportados para: {filename}")
                messagebox.showinfo("Sucesso", f"Resultados exportados para:\n{filename}")
            except Exception as e:
                self.log_general(f"❌ Erro ao exportar: {e}")
                messagebox.showerror("Erro", f"Falha ao exportar: {e}")

    def show_data_sample(self):
        """Mostra amostra dos dados carregados"""
        if self.df_filtrado is None:
            messagebox.showwarning("Aviso", "Nenhum dado carregado")
            return

        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(tk.END, f"AMOSTRA DOS DADOS CARREGADOS ({len(self.df_filtrado)} estrelas)\n")
        self.data_text.insert(tk.END, "=" * 60 + "\n\n")

        # Mostrar primeiras 20 linhas
        sample = self.df_filtrado.head(20)
        self.data_text.insert(tk.END, sample.to_string())

        self.data_text.insert(tk.END, "\n\n" + "=" * 60 + "\n")
        self.data_text.insert(tk.END, f"\nColunas disponíveis: {', '.join(self.df_filtrado.columns.tolist())}")
        self.data_text.insert(tk.END, f"\n\nEstatísticas básicas:\n")
        self.data_text.insert(tk.END, f"RA: {self.df_filtrado['ra'].min():.2f} a {self.df_filtrado['ra'].max():.2f}°\n")
        self.data_text.insert(tk.END,
                              f"Dec: {self.df_filtrado['dec'].min():.2f} a {self.df_filtrado['dec'].max():.2f}°\n")
        self.data_text.insert(tk.END,
                              f"Distância: {self.df_filtrado['distance_pc'].min():.1f} a {self.df_filtrado['distance_pc'].max():.1f} pc\n")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    root = tk.Tk()
    app = SistemaIntegradoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()