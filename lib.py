import numpy as np

class PolynomialLeastSquare:
    """
    多項式最小二乗法
    
    Attributes
    ----------
    coefficients : int
        使用する弱分類器の数
    d : int
        fitting を行う多項式の次数
    """
    
    def fit(self, x, y, d):
        """
        回帰曲線の係数を計算
        
        Parameters
        ----------
        x : numpy array [float]
        y : numpy array [float]
        d : int
            fitting する多項式の次数
        """
        n = len(x)
        # x のべき乗を計算
        x_pow = [np.full(n, 1.)]
        for i in range(2*d):
            x_pow.append(x_pow[-1] * x)
        x_pow = np.array(x_pow)
        # 行列 S を計算
        s = []
        for i in range(2*d+1):
            s.append(np.sum(x_pow[i]))
        S = np.zeros([d+1, d+1])
        for i in range(d+1):
            for j in range(d+1):
                S[i][j] = s[i+j]
        # ベクトル t を計算
        t = []
        for i in range(d+1):
            t.append(np.sum(x_pow[i]*y))
        t = np.matrix(t).T
        # 係数を求める
        S_inv = np.linalg.inv(S)
        self.coefficients = np.array(np.dot(S_inv, t)).flatten()
        self.d = d
    
    def predict(self, x):
        """
        学習済み回帰曲線を使って未知の x を変換
        
        Parameters
        ----------
        x : numpy array [float]
        """
        x_pow = np.full(len(x), 1.)
        y = np.zeros(len(x))
        for i in range(len(self.coefficients)):
            y += x_pow * self.coefficients[i]
            x_pow *= x
        return y

class RANSAC:
    """
    Attributes
    ----------
    mse : float
        平均二乗誤差 (Mean Square Error)
    ls_best : object
        最適な最小二乗法モデル
    x_inliers_best : numpy array [float]
        最適モデルを学習した際の正常値 x
    y_inliers_best : numpy array [float]
        最適モデルを学習した際の正常値 y
    """
    
    def __init__(self, max_trials, residual_threshold, min_inliers_rate):
        """
        Parameters
        ----------
        max_trials : int
            ランダムサンプリングによる fitting を試行する最大回数
        residual_threshold : float
            ランダムサンプルから学習したモデルとの残差がこれ以内であれば「正常値」とみなす
        min_inliers_rate : float
            [0, 1] の小数。データサンプル全体に占める正常値の割合がこれ以下のものは最適モデルの候補に含めない
        """
        self.max_trials = max_trials
        self.residual_threshold = residual_threshold
        self.min_inliers_rate = min_inliers_rate
    
    def predict(self, x):
        return self.ls_best.predict(x)
    
    def fit(self, x, y, d):
        """
        Parameters
        ----------
        x : numpy array [float]
        y : numpy array [float]
        d : int
            fitting する多項式の次数
        """
        n = len(x)
        # ランダムサンプリングの件数は最小限（モデルの自由度と同じ）にする
        n_part = d+1
        mse_min = np.inf
        ls_best = None
        x_inliers_best = None
        y_inliers_best = None
        for t in range(self.max_trials):
            ids_part = np.random.choice(n, n_part, replace=False)
            x_part, y_part = x[ids_part], y[ids_part]
            ls_part = PolynomialLeastSquare()
            ls_part.fit(x_part, y_part, d)
            ids_inliers = self.__detect_inliers_indices(x, y, ls_part)
            if len(ids_inliers) / n < self.min_inliers_rate:
                continue
            x_inliers, y_inliers = x[ids_inliers], y[ids_inliers]
            ls_inliers = PolynomialLeastSquare()
            ls_inliers.fit(x_inliers, y_inliers, d)
            mse = self.__calc_mse(x_inliers, y_inliers, ls_inliers)
            if mse < mse_min:
                mse_min = mse
                ls_best = ls_inliers
                x_inliers_best, y_inliers_best = x_inliers, y_inliers
        self.mse = mse_min
        self.ls_best = ls_best
        self.x_inliers_best, self.y_inliers_best = x_inliers_best, y_inliers_best
    
    def __detect_inliers_indices(self, x, y, model):
        """
        回帰モデルに対する正常値（のインデックス）を見つける
        """
        y_pred = model.predict(x)
        return np.where(np.abs(y-y_pred) < self.residual_threshold)[0]
    
    def __calc_mse(self, x, y, model):
        """
        回帰モデルに対するデータの平均二乗誤差（MSE）を計算
        """
        y_pred = model.predict(x)
        return np.average((y_pred-y)**2)