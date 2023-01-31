from scipy import integrate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Интегрирование акселерограммы"""

y = pd.read_csv("acc-x.csv")['A'].values
n = y.shape[0]
x = np.linspace(0, (n - 1) * 0.02, n)


def integ(i):
    intgr = integrate.trapz(y[0:i], x[0:i])
    return intgr


vinteg = np.vectorize(integ)

plt.figure("Акселерограмма")
plt.title("Акселерограмма")
plt.grid()
plt.xlabel("t, s")
plt.ylabel("a, m/s^2")
plt.plot(x, y)
plt.show()
y = vinteg(range(0, n))
plt.figure("Велосиграмма")
plt.title("Велосиграмма")
plt.grid()
plt.xlabel("t, s")
plt.ylabel("v, m/s")
plt.plot(x, y)
plt.show()
print("Скорость в конце: ", y[n - 1], " м/с")
y = vinteg(range(0, n))
plt.figure("Сейсмограмма")
plt.title("Сейсмограмма")
plt.grid()
plt.xlabel("t, s")
plt.ylabel("d, m")
plt.plot(x, y)
plt.show()
print("Смещение в конце: ", y[n - 1], " м")
