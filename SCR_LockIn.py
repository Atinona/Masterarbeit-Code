import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from joblib import Parallel, delayed
import time

start = time.time()

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
def berechneAC(AC_typ, k, n, T, dt, r, S, alpha, m, sigma_r, B, N0, EK, i_mon, gl, gf, xli, gamma, sigma_BS,
                 delta, psi, N_storno, N_diffTod, N_diffStorno):
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
    dCF = np.zeros(T)  # diskontierter Cash Flow
    r_sum = 0  # Summe des Zinsprozesses
    C = 0  # aktuelles Gesamtkapital
    if (AC_typ == 0):
        C = EK + B * N0
    X1 = np.zeros(10)  # (r, S, V_KS, V_GF, V_FF, V_A, V_A_BW, V_NK_NW, V_NK_BW, G)
    if (AC_typ == 1):
        X1[0] = r[k, 12]
        X1[1] = S[k, 12]

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

        # Todesfälle:
        CF = (N_diffTod[t - 1]) * V

        # Stornierungen
        CF += (1 - psi) * N_diffStorno[t - 1] * V
        V_KS_tat += psi * N_diffStorno[t - 1] * V

        # diskontierter Cash Flow:
        if (t <= 12):
            if (AC_typ == 0):
                r_sum += r[k][t - 1]
        else:
            r_sum += r[k][t - 1]
        if (t <= 12):
            if (AC_typ == 0):
                dCF[t - 1] = CF * np.exp(-dt * r_sum)  # diskontierter Cashflow
        else:
            dCF[t - 1] = CF * np.exp(-dt * r_sum)

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
        if (AC_typ == 1):
            if (t == 12):
                X1[2] = V_KS
                X1[3] = V_GF
                X1[4] = V_FF
                X1[9] = G

        # Aufteilung des tatsächlichen Sicherungskapitals:
        V_KS_tat += N_storno[t] * (V_KS - V_KS_alt)
        if (V_KS_tat < 0):
            print("Unternehmen ist insolvent.")  # Überprüfung, ob Unternehmen während Laufzeit insolvent geht
            return 0, np.zeros(10)
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
        if (AC_typ == 1):
            if (t == 12):
                C = V_KS_tat + N_storno[t] * (V_GF + V_FF)
                X1[5] = V_A
                X1[6] = V_A_BW
                X1[7] = V_NK_NW
                X1[8] = V_NK_BW

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

    # nochmal Überschuss:
    V_BW_alt = V_BW
    V_BW = V_A + V_NK_NW
    if (V_BW_alt != 0):
        i_BW = V_BW / V_BW_alt
        US += V_KS_alt * i_BW - V_KS
        if (US > 0):
            V += delta * US

    # Berechnung des verfügbaren Kapitals:
    r_sum += r[k][T - 1]
    dCF[T - 1] = N_storno[T - 1] * V * np.exp(-dt * r_sum)
    PVFP = np.sum(dCF)
    AC = C - PVFP

    return AC, X1

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
    N_tod[t] = round(N_storno[t-1] * (1 - q[int(np.floor(t / 12))] / 12))
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
K = 100000  # Anzahl der (äußeren) Simulationen
seed = 1
np.random.seed(seed)  # für Vergleichbarkeit der Ergebnisse

### Simulation der Kapitalmarktprozesse:

# für AC0:
r0 = np.full(K, r_0)
S0 = np.ones(K)
r1, S1 = simuliereProzesse(K, alpha, m, sigma_r, 0, sigma_s, rho, T, dt, r0, S0)
# für AC1:
r2 = np.zeros((K, T + 1))
S2 = np.ones((K, T + 1))
r2[:, :13], S2[:, :13] = simuliereProzesse(K, alpha, m, sigma_r, pi, sigma_s, rho, 12, dt, r0, S0)
r2[:, 12:], S2[:, 12:] = simuliereProzesse(
    K, alpha, m, sigma_r,0, sigma_s, rho, T - 12, dt, r2[:, 12], S2[:, 12])

step = time.time()
print("Zeitdifferenz: ", step - start, " Sekunden")

### Berechnung der Ergebnisse:
for gl in np.arange(0.7, 1.01, 0.1):
    gl = round(gl, 1)
    print()
    print(f"Leistungsgarantie: {gl}")

    # Berechnung AC0:

    berechnung = Parallel(n_jobs=-1)(
        delayed(berechneAC)(
            0, k, n, T, dt, r1, S1, alpha, m, sigma_r, B, N0, EK, i_mon, gl, gf, xli, gamma,
            sigma_BS, delta, psi, N_storno, N_diffTod, N_diffStorno
        ) for k in range(K)
    )
    AC0_array, X1_array = zip(*berechnung)
    AC0 = np.mean(AC0_array)
    print(f"AC0: {AC0}")

    # Berechnung AC1:

    berechnung = Parallel(n_jobs=-1)(
        delayed(berechneAC)(
            1, k, n, T, dt, r2, S2, alpha, m, sigma_r, B, N0, EK, i_mon, gl, gf, xli, gamma,
            sigma_BS, delta, psi, N_storno, N_diffTod, N_diffStorno
        ) for k in range(K)
    )
    AC1_array, X1 = zip(*berechnung)
    AC1_array = np.array(AC1_array)
    X1 = np.stack(X1)

    # Least Squares:

    scaler = StandardScaler()
    X1_scaled = scaler.fit_transform(X1)

    poly = PolynomialFeatures(degree=1, include_bias=True)
    E = poly.fit_transform(X1_scaled)  # Matrix der Basisfunktionen
    model = Ridge(alpha=1.0)
    model.fit(E, AC1_array)
    AC1neu = model.predict(E)

    # Quantil-Schätzer:

    Y = -AC1neu
    ind_sort = np.argsort(Y)  # Realisierungen aufsteigend sortieren
    Y_sort = Y[ind_sort]
    z = round(K * 0.995)
    Y_0995 = Y_sort[z - 1]

    print(f"SCR: {AC0 + np.exp(-i1) * Y_0995}")

end = time.time()
print()

print(f"Gesamtzeit: {end - start} Sekunden")
