import numpy as np
import math
import cvxopt
import scipy.interpolate as interp
import matplotlib.pyplot as plt

from cvxopt import matrix
import pandas as pd

wp = pd.read_csv('waypoint.csv')
# print(wp)
vel_orig = wp['velocity'].values
wp_x = wp['x'].values
wp_y = wp['y'].values

print(vel_orig[0])


v_max = 10.0 # velocity limit [m/s]
a_max = 1.5  # acceleration limit [m/s2]
s_max = 15.0  # jerk limit [m/s3]
latacc_max = 2.0 # lateral acceleration limit [m/s2]

l = len(vel_orig)

print(' -- original acc -- ')
acc_orig = vel_orig[1:l] * np.array(vel_orig[1:l] - vel_orig[0:l-1])
print(acc_orig)

print(' -- original jerk --')
jerk_orig = (vel_orig[1:l-1] * vel_orig[1:l-1]) * np.array(vel_orig[2:l] - vel_orig[0:l-2])
print(jerk_orig)

vel = vel_orig

def area2(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx**2 + dy**2)**0.5

def calc3PointsCurvature(a, b, c):
    return 2.0 * area2(a, b, c) / (dist(a,b) * dist(b,c) * dist(c,a))


# 曲率計算
curvature = np.zeros((l,1))
for i in range(1,l-1):
    p0 = np.array([wp_x[i-1], wp_y[i-1]])
    p1 = np.array([wp_x[i], wp_y[i]])
    p2 = np.array([wp_x[i+1], wp_y[i+1]])
    curvature[i] = calc3PointsCurvature(p0, p1, p2)
curvature[0] = curvature[1]
curvature[l-1] = curvature[l-2]
# print('curvature : ')
# print(curvature)

latacc_orig = np.abs(curvature) * vel_orig.reshape((l,1)) * vel_orig.reshape((l,1))

# 横Gを考慮した最大速度計算
v_latg_max = np.zeros((l,1))
for i in range(l):
    k = max(np.abs(curvature[i]), 0.00001) # to avoid 0 devide
    v_latg_max[i] = min((latacc_max / k) ** 0.5, v_max)
print('v_latg_max : ')
print(v_latg_max)

cvxopt.solvers.options['abstol'] = 1e-15
cvxopt.solvers.options['reltol'] = 1e-15
cvxopt.solvers.options['feastol'] = 1e-15

# 初期条件 & 終端条件
A = np.zeros((2,l))
A[0,0] = 1
A[1,l-1] = 1
b = np.array([vel[0], vel[l-1]])

for j in range(20):

    # # 速度制約
    G_vel = np.eye(l)
    G_vel = np.vstack((G_vel, -G_vel))
    h_vel = v_latg_max # 横G考慮
    h_vel = np.vstack((h_vel, h_vel))
    G = G_vel
    h = h_vel

    # # 加速度制約
    G_acc = np.zeros((l-1, l))
    for i in range(l-1):
        G_acc[i,i] = -vel[i]
        G_acc[i,i+1] = vel[i]
    G_acc = np.vstack((G_acc, -G_acc))
    h_acc = np.ones((l-1,1)) * a_max
    h_acc = np.vstack((h_acc, h_acc))
    G = np.vstack((G, G_acc))
    h = np.vstack((h, h_acc))

    # # 加加速度制約
    G_jerk = np.zeros((l-2, l))
    for i in range(l-2):
        G_jerk[i,i] = -(vel[i+1]**2)
        G_jerk[i,i+2] = vel[i+1]**2
    G_jerk = np.vstack((G_jerk, -G_jerk))
    h_jerk = np.ones((l-2,1)) * s_max
    h_jerk = np.vstack((h_jerk, h_jerk))
    G = np.vstack((G, G_jerk))
    h = np.vstack((h, h_jerk))

    # 横G制約

    P = np.eye(l)
    q = np.array(-vel_orig)

    cvxopt.solvers.options['show_progress'] = False
    # sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), G=cvxopt.matrix(G), h=cvxopt.matrix(h), A=cvxopt.matrix(A), b=cvxopt.matrix(b))
    sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), G=cvxopt.matrix(G), h=cvxopt.matrix(h))
    # sol = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q))
    vel = np.array(sol['x'])
    cost = sol["primal objective"] + np.dot(vel_orig, vel_orig)
    # print('cost = ', cost)

    # print(sol)
    # print(sol['x'])
    # print(sol["primal objective"])


print('vel res = ', vel)
vel_res = vel

vv = vel_res.reshape((l,1))
acc_res = vv[0:l-1] * (vel[1:l] - vel[0:l-1])
jerk_res = vv[1:l-1] * vv[1:l-1] * (vel[2:l] - vel[0:l-2])
latacc_res = curvature * vel * vel
# print('acc res = ',acc_res)

print('-- result --')
print('acc lim = ', a_max)
print('acc max = ', np.max(acc_res))
print('acc min = ', np.min(acc_res))

print('jerk lim = ', s_max)
print('jerk max = ', np.max(jerk_res))
print('jerk min = ', np.min(jerk_res))

print('latacc lim = ', latacc_max)
print('latacc max = ', np.max(latacc_res))
print('latacc min = ', np.min(latacc_res))

plt.subplot(2, 2, 1)
plt.plot(vel_orig, '-o', label="original")
plt.plot(vel, '-o', label="optimized")
plt.title('velocity')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(acc_orig, '-o', label="original")
plt.plot(acc_res, '-o', label="optimized")
plt.plot(np.ones(l) * a_max, '--', label="limit")
plt.title('acceleration')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(jerk_orig, '-o', label="original")
plt.plot(jerk_res, '-o', label="optimized")
plt.plot(np.ones(l) * s_max, '--', label="limit")
plt.title('jerk')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(latacc_orig, '-o', label="original")
plt.plot(latacc_res, '-o', label="optimized")
plt.plot(np.ones(l) * latacc_max, '--', label="limit")
plt.title('lateral acceleration')
plt.legend()

plt.show()