import numpy as np
import matplotlib.pyplot as plt 

from lib import RANSAC, PolynomialLeastSquare


def main():
    # データ作成
    N = 300
    x = np.arange(-5, 5, 10/N)
    y = x + 0.5*x**2 + 0.2*x**3 + 5 + np.random.randn(len(x))*3
    y[np.random.choice(N, N//10, replace=False)] = 100

    # RANSAC の fitting
    rs = RANSAC(max_trials=100, residual_threshold=10, min_inliers_rate=0.8)
    rs.fit(x, y, 3)
    print(rs.mse)

    print(rs.ls_best.coefficients)

    # 単純な最小二乗法の fitting
    ls = PolynomialLeastSquare()
    ls.fit(x, y, 3)
    print(ls.coefficients)

    # 回帰曲線を描画
    xp = np.arange(-6, 6, 0.1)
    yp_rs = rs.predict(xp)
    yp_ls = ls.predict(xp)
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(12, 5))
    axL.set_title('RANSAC with Least Square')
    axL.set_xlabel('$X$')
    axL.set_ylabel('$Y$')
    axL.scatter(x, y, s=10, label='outliers')
    axL.scatter(rs.x_inliers_best, rs.y_inliers_best, s=15, label='Inliers')
    axL.plot(xp, yp_rs, c='r', label='regression curve')
    axL.grid()
    axL.legend()
    axR.set_title('Simple Least Square')
    axR.set_xlabel('$X$')
    axR.set_ylabel('$Y$')
    axR.scatter(x, y, s=10)
    axR.plot(xp, yp_ls, c='r', label='regression curve')
    axR.grid()
    axR.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()