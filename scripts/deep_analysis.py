import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def analisar_preparacao_chicane():
    # Carregar os dados que exportaste previamente
    df = pd.read_csv("../data/2025_monza_q/2025_monza_q.csv")
    df_sim = pd.read_csv("../data/2025_monza_q/2025_monza_q_sim.csv")

    # Criar uma coluna de distância cumulativa para facilitar o eixo X
    df['Distance_m'] = df['Segment_Length'].cumsum()
    
    # Focar apenas na reta principal e na entrada da primeira chicane de Monza
    # Ajusta estes valores (ex: 500m a 1000m) consoante o ponto exato da travagem no teu dataset
    zona_estudo = df[(df['Distance_m'] > 0) & (df['Distance_m'] <= len(df))].copy()
    zona_estudo_sim = df_sim[(df_sim['Distance_m'] > 0) & (df_sim['Distance_m'] <= len(df_sim))].copy()

    distancia = zona_estudo_sim['Distance_m'].values
    x = zona_estudo['X'].values
    y = zona_estudo['Y'].values
    z = zona_estudo['Z'].values
    radius = zona_estudo['Radius'].values
    velocidade = zona_estudo['Real Speed'].values
    velocidade_sim = zona_estudo_sim['Sim_Speed']

    Gmax = 5 # gmax of 5g's
    # rmin = V² / Gmax * g
    radius_speed = (velocidade*velocidade) / (Gmax * 9.8067)
    radius_sim_speed = (velocidade_sim * velocidade_sim) / (Gmax * 9.8067)
    # 3. Visualização Dupla
    fig, ax1 = plt.subplots(1, 1, figsize=(19.20, 10.80), gridspec_kw={'height_ratios': [1]})
    
    # Gráfico 1: Trajetória XY (O movimento lateral real)
    # ax1.plot(x, y, label="Trajetória (Visão Superior)", color='blue', linewidth=2)
    # ax1.set_title("Micro-movimentos: Preparação de Trajetória para a 1ª Chicane (X,Y)")
    # ax1.set_xlabel("Coordenada X (m)")
    # ax1.set_ylabel("Coordenada Y (m)")
    # ax1.axis('equal') # Crítico: mantém a proporção real para não distorcer a curva
    # ax1.grid(True, linestyle='--', alpha=0.6)
    # ax1.legend()

    # Gráfico 2: dunno (movimento vertical)
    # ax2.plot(distancia, z, label="Z value",color='blue', linewidth=2)
    # ax2.set_xlabel("Distance (m)")
    # ax2.set_ylabel("z (m)")
    # ax2.grid(True, linestyle='--', alpha=0.6)
    # ax2.legend()

    # Gráfico 3: Evolução do Raio
    # ax3.plot(distancia, radius, label="Raio de Curvatura Spline", color='orange')
    # ax3.plot(distancia, radius_sim_speed, label="Raio de Curvatura Velocidade Simulada", color='purple')
    # ax3.plot(distancia, radius_speed, label="Raio de Curvatura Velocidade Real", color='blue')
    # ax3.set_title("Curvatura usando Continuous Spline Fitting vs Speed")
    # ax3.set_xlabel("Distância (m)")
    # ax3.set_ylabel("Raio (m)")
    # # ax3.set_yscale('log') # Escala logarítmica é ideal para visualizar a queda de 10000m para <100m
    # ax3.grid(True, linestyle='--', alpha=0.6)
    # ax3.legend()

    # Gráfico 4: Ponto de Travagem (Velocidade)
    # ax4.plot(distancia, velocidade, label="Velocidade Real (km/h)", color='red')
    # ax4.plot(distancia, velocidade_sim, label="Velocidade Simulada (km/h)", color='blue')
    # ax4.set_title("Referência de Travagem")
    # ax4.set_xlabel("Distância (m)")
    # ax4.set_ylabel("Velocidade")
    # ax4.grid(True, linestyle='--', alpha=0.6)
    # ax4.legend()
    
    # grafico dunno
    ax1.plot(distancia, zona_estudo_sim["Normal_Force_N"])

    plt.tight_layout()
    plt.savefig("analise_micro_movimentos.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    analisar_preparacao_chicane()