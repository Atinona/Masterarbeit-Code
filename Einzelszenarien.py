import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Preis einer europäischen Put-Option nach Black-Scholes:
def P_PO(S0, K, r, T, sigma):
    d1 = (np.log(S0 / K) + T * (sigma ** 2 * 0.5 + r)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# Preis einer Nullkuponanleihe:
def P_NK(alpha, m, sigma_r, r, T):
    h = np.sqrt(alpha ** 2 + 2 * sigma_r ** 2)
    C = 2 * h + (alpha + h) * (np.exp(T * h) - 1)
    A = (2 * h * np.exp((alpha + h) * 0.5 * T) / C) ** (2 * alpha * m / sigma_r ** 2)
    B = 2 * (np.exp(T * h) - 1) / C
    return A * np.exp(-r * B)

# Simulation der stochastischen Prozesse:
def simuliereProzesse(K, alpha, m, sigma_r, pi, sigma_s, rho, T, dt, r0, S0):
    r = np.zeros((K, T + 1))
    S = np.ones((K, T + 1))
    r[:,0] = r0
    S[:,0] = S0

    # korrelierte Brownsche Bewegungen:
    dW_S = np.random.normal(0, 1, (K, T))
    dZ = np.random.normal(0, 1, (K, T))
    dW_r = rho * dW_S + np.sqrt(1 - rho ** 2) * dZ

    # Zins- und Aktienpreisprozess mit Euler-Maruyama:
    for t in range(T):
        r[:, t + 1] = (r[:, t] + alpha * (m - r[:, t]) * dt + sigma_r * np.sqrt(np.maximum(r[:, t], 0))
                       * (np.sqrt(dt) * dW_r[:, t] + rho * pi * dt / sigma_s))
        S[:, t + 1] = S[:, t] + (r[:, t] + pi) * S[:, t] * dt + sigma_s * S[:, t] * np.sqrt(dt) * dW_S[:, t]

    return r, S

# Berechnung des AC für eine Simulation:
def berechneVertragOhne(k, n, T, dt, r, S, alpha, m, sigma_r, B, N0, EK, i_mon, gl, gf, gamma, sigma_BS,
                 delta, psi, N_storno, N_diffStorno):
    V = B  # Vertragsguthaben
    # V_KS: Kapital im konventionellen Sicherungsvermögen eines Vertrags
    # V_GF: Kapital im Garantiefonds eines Vertrags
    # V_FF: Kapital im freien Fonds eines Vertrags
    # V_NK: Kapital in den Nullkuponanleihen
    # V_NK_NW: Nennwert der Nullkuponanleihen
    # V_NK_BW: Buchwert der Nullkuponanleihen
    # V_A: Kapital in den Aktien
    # V_A_BW: Buchwert der Aktien
    # V_BW: Buchwert des tatsächlichen Sicherungskapitals
    V_KS_tat = EK  # tatsächliches Sicherungskapital
    US = 0  # Überschuss
    V_KS_array = np.zeros(T + 1)
    V_GF_array = np.zeros(T + 1)
    V_FF_array = np.zeros(T + 1)
    BG_array = np.zeros(T + 1)

    # Umschichtung:
    BG = gl * B * i_mon ** (1 - T)   # Barwert der Garantieleistung
    if gf * V < BG:
        V_KS = (BG - gf * V) / (i_mon - gf)
        V_GF = V - V_KS
    else:
        V_KS = 0
        V_GF = BG / gf
    V_FF = V - V_KS - V_GF
    BG_array[0] = BG

    # Aufteilung des tatsächlichen Sicherungskapitals:
    V_KS_tat += V_KS * N0
    V_A = 0.1 * V_KS_tat
    V_A_BW = V_A
    V_NK = V_KS_tat - V_A
    V_NK_NW = V_NK / P_NK(alpha, m, sigma_r, r[k][0], n)
    V_NK_BW = V_NK
    V_BW = V_KS_tat

    # Entwicklung zum nächsten Monat:
    V_KS_alt = V_KS
    V_KS *= i_mon
    P_POwert = P_PO((1 - gamma) ** (-dt), gf, r[k][0], dt, sigma_BS)
    V_GF *= max(gf, (1 - gamma) ** dt * S[k][1] / (1 + P_POwert))
    V_FF *= (1 - gamma) ** dt * S[k][1]
    V_A *= S[k][1]
    P_NKwert = P_NK(alpha, m, sigma_r, r[k][1], (T - 1) / 12)
    V_NK = V_NK_NW * P_NKwert
    V = V_KS + V_GF + V_FF
    V_KS_tat = V_A + V_NK

    for t in range(1, T):
        V_KS_array[t] = V_KS
        V_GF_array[t] = V_GF
        V_FF_array[t] = V_FF
        BG_array[t] = BG

        # Überschussverteilung:
        V_A_BW = 0.98 * V_A_BW + 0.02 * V_A
        V_NK_BW = 0.98 * V_NK_BW + 0.02 * V_NK
        V_BW_alt = V_BW
        V_BW = V_A_BW + V_NK_BW
        if (V_BW_alt != 0):
            i_BW = V_BW / V_BW_alt
            US += V_KS_alt * i_BW - V_KS
        if (t % 12 == 0):
            if (US > 0):
                V += delta * US
            US = 0

        # Stornierungen
        V_KS_tat += psi * N_diffStorno[t - 1] * V

        # Umschichtung:
        V_KS_alt = V_KS
        BG = gl * B * i_mon ** (1 - (T - t))
        if gf * V < BG:
            V_KS = (BG - gf * V) / (i_mon - gf)
            V_GF = V - V_KS
        else:
            V_KS = 0
            V_GF = BG / gf
        V_FF = V - V_KS - V_GF

        # Aufteilung des tatsächlichen Sicherungskapitals:
        V_KS_tat += N_storno[t] * (V_KS - V_KS_alt)
        if (V_KS_tat < 0):
            print("Unternehmen ist insolvent.")  # Überprüfung, ob Unternehmen während Laufzeit insolvent geht
            return V_KS_array, V_GF_array, V_FF_array, BG_array  # ggf. Abbruch bei Insolvenz
        V_A_alt = V_A
        V_A = 0.1 * V_KS_tat
        V_A_diff = V_A - V_A_alt
        if (V_A_diff > 0):
            V_A_BW += V_A_diff
        elif (V_A_diff < 0):
            V_A_BW *= V_A / V_A_alt
        V_NK_alt = V_NK
        V_NK = V_KS_tat - V_A
        V_NK_NW = V_NK / P_NKwert
        V_NK_diff = V_NK - V_NK_alt
        if (V_NK_diff > 0):
            V_NK_BW += V_NK_diff
        elif (V_NK_diff < 0):
            V_NK_BW *= V_NK / V_NK_alt
        V_BW = V_A_BW + V_NK_BW

        # Entwicklung zum nächsten Monat:
        V_KS_alt = V_KS
        V_KS *= i_mon
        P_POwert = P_PO(S[k][t] * (1 - gamma) ** (-dt), gf * S[k][t], r[k][t], dt, sigma_BS)
        V_GF *= max(gf, (1 - gamma) ** dt * S[k][t + 1] / (S[k][t] + P_POwert))
        V_FF *= (1 - gamma) ** dt * S[k][t + 1] / S[k][t]
        V_A *= S[k][t + 1] / S[k][t]
        P_NKwert = P_NK(alpha, m, sigma_r, r[k][t + 1], (T - t - 1) / 12)
        V_NK = V_NK_NW * P_NKwert
        V = V_KS + V_GF + V_FF
        V_KS_tat = V_A + V_NK

    V_KS_array[T] = V_KS
    V_GF_array[T] = V_GF
    V_FF_array[T] = V_FF
    BG_array[T] = BG

    # nochmal Überschuss:
    V_BW_alt = V_BW
    V_BW = V_A + V_NK_NW
    if (V_BW_alt != 0):
        i_BW = V_BW / V_BW_alt
        US += V_KS_alt * i_BW - V_KS
        if (US > 0):
            V += delta * US

    return V_KS_array, V_GF_array, V_FF_array, BG_array

def berechneVertragEinbahn(k, n, T, dt, r, S, alpha, m, sigma_r, B, N0, EK, i_mon, gl, gf, gamma, sigma_BS,
                 delta, psi, N_storno, N_diffStorno):
    V = B  # Vertragsguthaben
    V_KS = 0  # Kapital im konventionellen Sicherungsvermögen eines Vertrags
    # V_GF: Kapital im Garantiefonds eines Vertrags
    # V_FF: Kapital im freien Fonds eines Vertrags
    # V_NK: Kapital in den Nullkuponanleihen
    # V_NK_NW: Nennwert der Nullkuponanleihen
    # V_NK_BW: Buchwert der Nullkuponanleihen
    # V_A: Kapital in den Aktien
    # V_A_BW: Buchwert der Aktien
    # V_BW: Buchwert des tatsächlichen Sicherungskapitals
    V_KS_tat = EK  # tatsächliches Sicherungskapital
    US = 0  # Überschuss
    V_KS_array = np.zeros(T + 1)
    V_GF_array = np.zeros(T + 1)
    V_FF_array = np.zeros(T + 1)
    BG_array = np.zeros(T + 1)

    # Umschichtung:
    BG = gl * B * i_mon ** (1 - T)   # Barwert der Garantieleistung
    V_KS = max((BG - gf * V) / (i_mon - gf), V_KS)
    if gf * (V - V_KS) + i_mon * V_KS < BG:
        V_GF = V - V_KS
    else:
        V_GF = (BG - i_mon * V_KS) / gf
    V_FF = V - V_KS - V_GF
    BG_array[0] = BG

    # Aufteilung des tatsächlichen Sicherungskapitals:
    V_KS_tat += V_KS * N0
    V_A = 0.1 * V_KS_tat
    V_A_BW = V_A
    V_NK = V_KS_tat - V_A
    V_NK_NW = V_NK / P_NK(alpha, m, sigma_r, r[k][0], n)
    V_NK_BW = V_NK
    V_BW = V_KS_tat

    # Entwicklung zum nächsten Monat:
    V_KS_alt = V_KS
    V_KS *= i_mon
    P_POwert = P_PO((1 - gamma) ** (-dt), gf, r[k][0], dt, sigma_BS)
    V_GF *= max(gf, (1 - gamma) ** dt * S[k][1] / (1 + P_POwert))
    V_FF *= (1 - gamma) ** dt * S[k][1]
    V_A *= S[k][1]
    P_NKwert = P_NK(alpha, m, sigma_r, r[k][1], (T - 1) / 12)
    V_NK = V_NK_NW * P_NKwert
    V = V_KS + V_GF + V_FF
    V_KS_tat = V_A + V_NK

    for t in range(1, T):
        V_KS_array[t] = V_KS
        V_GF_array[t] = V_GF
        V_FF_array[t] = V_FF
        BG_array[t] = BG

        # Überschussverteilung:
        V_A_BW = 0.98 * V_A_BW + 0.02 * V_A
        V_NK_BW = 0.98 * V_NK_BW + 0.02 * V_NK
        V_BW_alt = V_BW
        V_BW = V_A_BW + V_NK_BW
        if (V_BW_alt != 0):
            i_BW = V_BW / V_BW_alt
            US += V_KS_alt * i_BW - V_KS
        if (t % 12 == 0):
            if (US > 0):
                V += delta * US
            US = 0

        # Stornierungen
        V_KS_tat += psi * N_diffStorno[t - 1] * V

        # Umschichtung:
        V_KS_alt = V_KS
        BG = gl * B * i_mon ** (1 - (T - t))
        V_KS = max((BG - gf * V) / (i_mon - gf), V_KS)
        if gf * (V - V_KS) + i_mon * V_KS < BG:
            V_GF = V - V_KS
        else:
            V_GF = (BG - i_mon * V_KS) / gf
        V_FF = V - V_KS - V_GF

        # Aufteilung des tatsächlichen Sicherungskapitals:
        V_KS_tat += N_storno[t] * (V_KS - V_KS_alt)
        if (V_KS_tat < 0):
            print("Unternehmen ist insolvent.")  # Überprüfung, ob Unternehmen während Laufzeit insolvent geht
            return V_KS_array, V_GF_array, V_FF_array, BG_array  # ggf. Abbruch bei Insolvenz
        V_A_alt = V_A
        V_A = 0.1 * V_KS_tat
        V_A_diff = V_A - V_A_alt
        if (V_A_diff > 0):
            V_A_BW += V_A_diff
        elif (V_A_diff < 0):
            V_A_BW *= V_A / V_A_alt
        V_NK_alt = V_NK
        V_NK = V_KS_tat - V_A
        V_NK_NW = V_NK / P_NKwert
        V_NK_diff = V_NK - V_NK_alt
        if (V_NK_diff > 0):
            V_NK_BW += V_NK_diff
        elif (V_NK_diff < 0):
            V_NK_BW *= V_NK / V_NK_alt
        V_BW = V_A_BW + V_NK_BW

        # Entwicklung zum nächsten Monat:
        V_KS_alt = V_KS
        V_KS *= i_mon
        P_POwert = P_PO(S[k][t] * (1 - gamma) ** (-dt), gf * S[k][t], r[k][t], dt, sigma_BS)
        V_GF *= max(gf, (1 - gamma) ** dt * S[k][t + 1] / (S[k][t] + P_POwert))
        V_FF *= (1 - gamma) ** dt * S[k][t + 1] / S[k][t]
        V_A *= S[k][t + 1] / S[k][t]
        P_NKwert = P_NK(alpha, m, sigma_r, r[k][t + 1], (T - t - 1) / 12)
        V_NK = V_NK_NW * P_NKwert
        V = V_KS + V_GF + V_FF
        V_KS_tat = V_A + V_NK

    V_KS_array[T] = V_KS
    V_GF_array[T] = V_GF
    V_FF_array[T] = V_FF
    BG_array[T] = BG

    # nochmal Überschuss:
    V_BW_alt = V_BW
    V_BW = V_A + V_NK_NW
    if (V_BW_alt != 0):
        i_BW = V_BW / V_BW_alt
        US += V_KS_alt * i_BW - V_KS
        if (US > 0):
            V += delta * US

    return V_KS_array, V_GF_array, V_FF_array, BG_array

def berechneVertragLockIn(k, n, T, dt, r, S, alpha, m, sigma_r, B, N0, EK, i_mon, gl, gf, xli, gamma, sigma_BS,
                 delta, psi, N_storno, N_diffStorno):
    V = B  # Vertragsguthaben
    # V_KS: Kapital im konventionellen Sicherungsvermögen eines Vertrags
    # V_GF: Kapital im Garantiefonds eines Vertrags
    # V_FF: Kapital im freien Fonds eines Vertrags
    # V_NK: Kapital in den Nullkuponanleihen
    # V_NK_NW: Nennwert der Nullkuponanleihen
    # V_NK_BW: Buchwert der Nullkuponanleihen
    # V_A: Kapital in den Aktien
    # V_A_BW: Buchwert der Aktien
    # V_BW: Buchwert des tatsächlichen Sicherungskapitals
    V_KS_tat = EK  # tatsächliches Sicherungskapital
    US = 0  # Überschuss
    V_KS_array = np.zeros(T + 1)
    V_GF_array = np.zeros(T + 1)
    V_FF_array = np.zeros(T + 1)
    BG_array = np.zeros(T + 1)

    # Umschichtung:
    G = gl * B
    BG = G * i_mon ** (1 - T)   # Barwert der Garantieleistung
    if gf * V < BG:
        V_KS = (BG - gf * V) / (i_mon - gf)
        V_GF = V - V_KS
    else:
        V_KS = 0
        V_GF = BG / gf
    V_FF = V - V_KS - V_GF
    BG_array[0] = BG

    # Aufteilung des tatsächlichen Sicherungskapitals:
    V_KS_tat += V_KS * N0
    V_A = 0.1 * V_KS_tat
    V_A_BW = V_A
    V_NK = V_KS_tat - V_A
    V_NK_NW = V_NK / P_NK(alpha, m, sigma_r, r[k][0], n)
    V_NK_BW = V_NK
    V_BW = V_KS_tat

    # Entwicklung zum nächsten Monat:
    V_KS_alt = V_KS
    V_KS *= i_mon
    P_POwert = P_PO((1 - gamma) ** (-dt), gf, r[k][0], dt, sigma_BS)
    V_GF *= max(gf, (1 - gamma) ** dt * S[k][1] / (1 + P_POwert))
    V_FF *= (1 - gamma) ** dt * S[k][1]
    V_A *= S[k][1]
    P_NKwert = P_NK(alpha, m, sigma_r, r[k][1], (T - 1) / 12)
    V_NK = V_NK_NW * P_NKwert
    V = V_KS + V_GF + V_FF
    V_KS_tat = V_A + V_NK

    for t in range(1, T):
        V_KS_array[t] = V_KS
        V_GF_array[t] = V_GF
        V_FF_array[t] = V_FF
        BG_array[t] = BG

        # Überschussverteilung:
        V_A_BW = 0.98 * V_A_BW + 0.02 * V_A
        V_NK_BW = 0.98 * V_NK_BW + 0.02 * V_NK
        V_BW_alt = V_BW
        V_BW = V_A_BW + V_NK_BW
        if (V_BW_alt != 0):
            i_BW = V_BW / V_BW_alt
            US += V_KS_alt * i_BW - V_KS
        if (t % 12 == 0):
            if (US > 0):
                V += delta * US
            US = 0

        # Stornierungen
        V_KS_tat += psi * N_diffStorno[t - 1] * V

        # Umschichtung:
        V_KS_alt = V_KS
        if (t % 12 == 0):               #Lock-In
            if (V - G >= xli * G):
                G += 0.5 * (V - G)
        BG = G * i_mon ** (1 - (T - t))
        if gf * V < BG:
            V_KS = (BG - gf * V) / (i_mon - gf)
            V_GF = V - V_KS
        else:
            V_KS = 0
            V_GF = BG / gf
        V_FF = V - V_KS - V_GF

        # Aufteilung des tatsächlichen Sicherungskapitals:
        V_KS_tat += N_storno[t] * (V_KS - V_KS_alt)
        if (V_KS_tat < 0):
            print("Unternehmen ist insolvent.")  # Überprüfung, ob Unternehmen während Laufzeit insolvent geht
            return V_KS_array, V_GF_array, V_FF_array, BG_array  # ggf. Abbruch bei Insolvenz
        V_A_alt = V_A
        V_A = 0.1 * V_KS_tat
        V_A_diff = V_A - V_A_alt
        if (V_A_diff > 0):
            V_A_BW += V_A_diff
        elif (V_A_diff < 0):
            V_A_BW *= V_A / V_A_alt
        V_NK_alt = V_NK
        V_NK = V_KS_tat - V_A
        V_NK_NW = V_NK / P_NKwert
        V_NK_diff = V_NK - V_NK_alt
        if (V_NK_diff > 0):
            V_NK_BW += V_NK_diff
        elif (V_NK_diff < 0):
            V_NK_BW *= V_NK / V_NK_alt
        V_BW = V_A_BW + V_NK_BW

        # Entwicklung zum nächsten Monat:
        V_KS_alt = V_KS
        V_KS *= i_mon
        P_POwert = P_PO(S[k][t] * (1 - gamma) ** (-dt), gf * S[k][t], r[k][t], dt, sigma_BS)
        V_GF *= max(gf, (1 - gamma) ** dt * S[k][t + 1] / (S[k][t] + P_POwert))
        V_FF *= (1 - gamma) ** dt * S[k][t + 1] / S[k][t]
        V_A *= S[k][t + 1] / S[k][t]
        P_NKwert = P_NK(alpha, m, sigma_r, r[k][t + 1], (T - t - 1) / 12)
        V_NK = V_NK_NW * P_NKwert
        V = V_KS + V_GF + V_FF
        V_KS_tat = V_A + V_NK

    V_KS_array[T] = V_KS
    V_GF_array[T] = V_GF
    V_FF_array[T] = V_FF
    BG_array[T] = BG

    # nochmal Überschuss:
    V_BW_alt = V_BW
    V_BW = V_A + V_NK_NW
    if (V_BW_alt != 0):
        i_BW = V_BW / V_BW_alt
        US += V_KS_alt * i_BW - V_KS
        if (US > 0):
            V += delta * US

    return V_KS_array, V_GF_array, V_FF_array, BG_array

### Modellparameter:

# allgemeine Versicherungsparameter:
n = 30  # Versicherungsdauer in Jahren
T = n * 12  # Versicherungsdauer in Monaten
B = 30000  # Einmalbeitrag
N0 = 10000  # Anzahl der Verträge zu Beginn
EK = 30000000  # Eigenkapital des Versicherers zu Beginn
i = 0.01  # Garantiezins
i_mon = (1 + i) ** (1 / 12)
gf = 0.8  # Garantie des Garantiefonds
gl = 0.7  # Leistungsgarantie
xli = 0.2 # Lock-In Prozentsatz
gamma = 0.01  # Fondskosten
psi = 0.02  # Stornokosten
delta = 0.95  # Überschussanteil

# Parameter für Tod und Storno:
q = [0.001005, 0.001083, 0.001181, 0.001301, 0.001447, 0.001623, 0.001833, 0.002082, 0.002364, 0.002669,
     0.002983, 0.003302, 0.003630, 0.003981, 0.004371, 0.004812, 0.005308, 0.005857, 0.006460, 0.007117,
     0.007831, 0.008604, 0.009454, 0.010404, 0.011504, 0.012818, 0.014429, 0.016415, 0.018832, 0.021704]
     #Todeswahrscheinlichkeit ab Alter 37
st = 0.03  # Stornowahrscheinlichkeit

# Anzahl der Verträge im Verlauf der Versicherungsdauer:
N_tod = np.zeros(T)
N_storno = np.zeros(T)
N_tod[0] = N0
N_storno[0] = N0
for t in range(1, T):
    N_tod[t] = round(N_storno[t-1] * (1 - q[int(np.floor((t) / 12))] / 12))
    N_storno[t] = round(N_tod[t] * (1 - st / 12))
N_diffTod = N_storno[:T-1] - N_tod[1:]
N_diffStorno = N_tod[1:] - N_storno[1:]

i1 = 0.02102  # risikofreier Einjahreszins

# Parameter des Kapitalmarkts:
r_0 = 0.025
alpha = 0.3
m = 0.04
sigma_r = 0.025
sigma_s = 0.2
rho = 0.5
pi = 0.02
dt = 1 / 12
sigma_BS = 0.4

# Simulationsparameter:
K = 1  # Anzahl der (äußeren) Simulationen
seed = 1  # Szenario 1 - Seed 1, Szenario 2 - Seed 3, Szenario 3 - Seed 2
np.random.seed(seed)  # für Vergleichbarkeit der Ergebnisse

### Simulation der Kapitalmarktprozesse:

r0 = np.full(K, r_0)
S0 = np.ones(K)
# unter tatsächlichem Maß:
r1, S1 = simuliereProzesse(K, alpha, m, sigma_r, pi, sigma_s, rho, T, dt, r0, S0)
# unter risikoneutralem Maß:
#r1, S1 = simuliereProzesse(K, alpha, m, sigma_r, 0, sigma_s, rho, T, dt, r0, S0)
t = np.arange(0, T + 1)

### Berechnung der Ergebnisse:

V_KS_array, V_GF_array, V_FF_array, BG_array = berechneVertragOhne(
    0, n, T, dt, r1, S1, alpha, m, sigma_r, B, N0, EK, i_mon, gl, gf, gamma,
    sigma_BS, delta, psi, N_storno, N_diffStorno)
V_KS_Einbahn_array, V_GF_Einbahn_array, V_FF_Einbahn_array, BG_Einbahn_array = berechneVertragEinbahn(
    0, n, T, dt, r1, S1, alpha, m, sigma_r, B, N0, EK, i_mon, gl, gf, gamma,
    sigma_BS, delta, psi, N_storno, N_diffStorno)
V_KS_LockIn_array, V_GF_LockIn_array, V_FF_LockIn_array, BG_LockIn_array = berechneVertragLockIn(
    0, n, T, dt, r1, S1, alpha, m, sigma_r, B, N0, EK, i_mon, gl, gf, xli, gamma,
    sigma_BS, delta, psi, N_storno, N_diffStorno)

# Plot:
fig, axs = plt.subplots(5, 1, figsize=(10, 12))  # 3 Zeilen, 1 Spalte
axs[0].plot(t, r1[0], color='green', label='Zins')
axs[0].set_xlabel("Zeit in Monaten")
axs[0].set_ylabel("Zins")
axs[0].legend()
axs[0].grid(True)
axs[1].plot(t, S1[0], label='Aktienkurs')
axs[1].set_xlabel("Zeit in Monaten")
axs[1].set_ylabel("Aktienkurs")
axs[1].legend()
axs[1].grid(True)
daten = np.vstack([V_KS_array, V_GF_array, V_FF_array])
axs[2].stackplot(t, daten,
                 labels=['konventionelles Sicherungsvermögen', 'Garantiefonds', 'freier Fonds'], alpha=0.6)
axs[2].plot(t, BG_array, color='black', linewidth=1, label='Barwert der Garantieleistung')
axs[2].legend(loc='upper left', fontsize=8)
axs[2].set_xlabel('Zeit in Monaten')
axs[2].set_ylabel('Vertragsguthaben in Euro')
axs[2].grid(True)
axs[2].set_title("ohne Zusatzoptionen")
datenEinbahn = np.vstack([V_KS_Einbahn_array, V_GF_Einbahn_array, V_FF_Einbahn_array])
axs[3].stackplot(t, datenEinbahn,
                 labels=['konventionelles Sicherungsvermögen', 'Garantiefonds', 'freier Fonds'], alpha=0.6)
axs[3].plot(t, BG_Einbahn_array, color='black', linewidth=1, label='Barwert der Garantieleistung')
axs[3].legend(loc='upper left', fontsize=8)
axs[3].set_xlabel('Zeit in Monaten')
axs[3].set_ylabel('Vertragsguthaben in Euro')
axs[3].grid(True)
axs[3].set_title("Einbahnstraßen-Option")
datenLockIn = np.vstack([V_KS_LockIn_array, V_GF_LockIn_array, V_FF_LockIn_array])
axs[4].stackplot(t, datenLockIn,
                 labels=['konventionelles Sicherungsvermögen', 'Garantiefonds', 'freier Fonds'], alpha=0.6)
axs[4].plot(t, BG_LockIn_array, color='black', linewidth=1, label='Barwert der Garantieleistung')
axs[4].legend(loc='upper left', fontsize=8)
axs[4].set_xlabel('Zeit in Monaten')
axs[4].set_ylabel('Vertragsguthaben in Euro')
axs[4].grid(True)
axs[4].set_title("Lock-In-Option")
plt.tight_layout()
plt.savefig(f"Vertragsablauf_Seed{seed}.png")