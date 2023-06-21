import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# input sample height prior to loading stage in mm
samle_hight = 15
# input file
inputfile = np.genfromtxt("DATA.dat")
t = np.array(inputfile[0:, 0])
h = np.array(inputfile[0:, 1])

# interpolating missing values using quadratic function
f_interpolate = interpolate.interp1d(t, h, kind='quadratic')
tn = np.arange(min(t), t[len(t) - 1], 0.01)
hn = f_interpolate(tn)

tv = tn[0]
hv = hn[0]

# evaluating first derivative
fd = [0]
for i in range(1, len(tn)):
    fd.append((hn[i] - hv) / (tn[i] - tv) * (tn[i] + tv) / 2 * np.log(10))
    tv = tn[i]
    hv = hn[i]
# maximum slope
max_index = int(fd.index(max(fd)))
# tangent line equation
tline = max(fd) * np.log10(tn / tn[max_index]) + hn[max_index]


# intersection point
def curvefunc(x):
    return (max(fd) * np.log10(x / tn[max_index]) + hn[max_index] - hn[len(hn) - 1])


t_intersect = fsolve(curvefunc, 1)
h_intersect = f_interpolate(t_intersect)


def get_index_larger_than(list, number):
    for index, value in enumerate(list):
        if value > number:
            return index


time_index = get_index_larger_than(t, t_intersect * 3)

i = time_index
j = 0
tinitial = []
hinitial = []

while i < len(t) and j < 2:
    tinitial.append(t[i])
    hinitial.append(h[i])
    if len(hinitial) >= 2:
        hlast = np.array(hinitial)
        tlast = np.array(np.log10(tinitial)).reshape((-1, 1))
        model = LinearRegression().fit(tlast, hlast)
        h_pred = model.predict(tlast)
        rsq = r2_score(hlast, h_pred)
        if rsq > 0.99:
            B = model.intercept_
            M = model.coef_
        else:
            j = 4
    i = i + 1
tcline = M * np.log10(tn) + B


def func(x):
    return (max(fd) * np.log10(x / tn[max_index]) + hn[max_index]) - (M * np.log10(x) + B)


root = np.round(fsolve(func, 1), 1)

u100 = max(fd) * np.log10(root / tn[max_index]) + hn[max_index]


g = 0
gg = 0

while g < len(tn) and gg == 0:
    incr = f_interpolate(4 * tn[g]) - min(h)
    if incr >= 0.25 * (max(h) - min(h)) and incr <= 0.5 * (max(h) - min(h)):
        gg = 5
        tt = tn[g]
    else:
        g = g + 1

head_diff = f_interpolate(4 * tn[g]) - f_interpolate(tn[g])
u0 = f_interpolate(tn[g]) - head_diff
u50 = (u0 + u100) / 2

i = 0
j = 0
while i < len(hn) - 1 and j < 2:
    if hn[i] < u50 and hn[i + 1] > u50:
        t50 = (tn[i] + tn[i + 1]) / 2
        j = 5
    elif hn[i] == u50:
        t50 = tn[i]
        j = 5
    i = i + 1

u0line = tn / tn * u0
u100line = tn / tn * u100
u50line = tn / tn * u50
print("u0="+str(u0))
print("u100="+str(u100))
print("u50="+str(u50))
print("t50="+str(t50))
print("Cv="+str(0.197 * ((samle_hight - u50) / 2) ** 2 / t50))

plt.semilogx(t, h, "ko", markerfacecolor='none', label="Actual data")
plt.semilogx(tn, hn, "r", label="Fitting curve")
plt.semilogx(tn, tline, linestyle='dashed', label="Steepest tangent")
plt.semilogx(tn, tcline, linestyle='dashed', label="Ending portion", color="blue")
plt.semilogx(tn, u0line, label="U0", color="blue")
plt.semilogx(tn, u100line, label="U100", color="black")
plt.semilogx(tn, u50line, label="U50", color="green")
# Plot the elbow point
plt.ylim([u0 - 0.05, max(h) + 0.05])
plt.xlabel("log time(min)")
plt.ylabel("Delta h(mm)")
plt.legend(loc='upper right')
plt.gca().invert_yaxis()
plt.show()







