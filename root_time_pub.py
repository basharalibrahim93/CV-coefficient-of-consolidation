import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# input sample height prior to loading stage in mm
samle_hight = 15
# input file
inputfile = np.genfromtxt("DATA.dat")
t = np.array(inputfile[0:, 0])
h = np.array(inputfile[0:, 1])
t = np.sqrt(t)
def find_elbow_point(data):
    x1, y1 = 0, data[0]  # First point
    x2, y2 = len(data) - 1, data[-1]  # Last point
    distances = []
    for i in range(len(data)):
        x0, y0 = i, data[i]  # Current point
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        distances.append(numerator / denominator)

    max_index = np.argmax(distances)
    return max_index



f_interpolate = interpolate.interp1d(t, h, kind='quadratic')
tn = np.arange(min(t), max(t), 0.01)
hn = f_interpolate(tn)
we = 0
ce = 0
while we < 2 and ce < len(t):
    if t[ce] >= 0.5:
        we = 5
    else:
        ce = ce + 1

i = ce
j = 0
tinitial = t[i:len(t) - 1]
hinitial = h[i:len(h) - 1]
he = np.array(hinitial)
te = np.array(tinitial).reshape((-1, 1))
elbow_point = he[find_elbow_point(he)]
tlast = [t[i]]
hlast = [h[i]]
i = i + 1

while i < len(t) and j < 2:
    if h[i] <= (elbow_point - h[0]) * 0.9 + h[0]:
        tlast.append(t[i])
        hlast.append(h[i])
        i = i + 1
    else:
        j = 5
hlast = np.array(hlast)
tlast = np.array(tlast).reshape((-1, 1))
model = LinearRegression().fit(tlast, hlast)
B = model.intercept_
M = model.coef_

tline = M * tn + B
tline_15 = 0.85 * M * tn + B

i = 0
j = 0
while i < len(tn) - 1 and j < 2:
    if 0.85 * M * tn[i] + B - f_interpolate(tn[i]) <= 0 and 0.85 * M * tn[i + 1] + B - f_interpolate(
            tn[i + 1]) >= 0:
        root_t90 = (tn[i] + tn[i + 1]) / 2
        u90 = f_interpolate(root_t90)
        u100 = u90 + u90 / 9
        j = 5
    else:
        j = j
    i = i + 1

u50 = (u100 + B) / 2
t90 = root_t90 ** 2

print("u100="+str(u100))
print("u90="+str(u90))
print("t90="+str(t90))
print("Cv="+str(0.848 * ((samle_hight - u50) / 2) ** 2 / t90))

plt.plot(t, h, "ko", markerfacecolor='none', label="Actual data")
plt.plot(tn, tline, "r", label="Initial tangent", linestyle='dashed')
plt.plot(tn, tline_15, label="1.15 Rotated line", linestyle='dashed', color="blue")
plt.plot(tn, hn, "r", label="Fitting curve")
plt.ylim([h[0] - 0.05, max(h) + 0.05])
plt.xlabel("Root time(min^0.5)")
plt.ylabel("Delta h (mm)")
plt.legend(loc='upper right')
plt.gca().invert_yaxis()
plt.show()







