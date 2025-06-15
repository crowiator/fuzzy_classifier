import numpy as np
import skfuzzy as fuzz
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# -------- rozsah premennej QRSd_ms ----------
lo, hi = 55.6, 500.0
universe = np.linspace(lo, hi, 1000)

# -------- tri Gaussovské množiny ------------
span = (hi - lo) / 6
kratky   = fuzz.gaussmf(universe, lo + span, span/2)
normalny = fuzz.gaussmf(universe, (lo+hi)/2, span/2)
dlhy     = fuzz.gaussmf(universe, hi - span, span/2)

# -------- vykresli pre kontrolu -------------
plt.plot(universe, kratky,   label="kratky")
plt.plot(universe, normalny, label="normalny")
plt.plot(universe, dlhy,     label="dlhy")
plt.title("QRSd_ms – tri členitostné funkcie")
plt.xlabel("QRSd [ms]"); plt.ylabel("stup. členstva")
plt.legend(); plt.show()