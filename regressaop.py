import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import scipy.stats as stats
import seaborn as sns

base = pd.read_csv('mt_cars.csv')

base  = base.drop(['Unnamed: 0'],axis=1)
print(base.head())

corr = base.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')

column_pairs = [('mpg','cyl'),('mpg','disp'),('mpg','hp'),('mpg','wt'),('mpg','drat'),('mpg','vs')]
n_plots = len(column_pairs)
fig, axes = plt.subplots(nrows=n_plots, ncols=1,figsize=(6,4*n_plots))
for i, pair in enumerate(column_pairs):
    x_col, y_col = pair
    sns.scatterplot(x=x_col,y=y_col,data=base,ax=axes[i])
    axes[i].set_title(f'{x_col} vs {y_col}')
#aic 156.6  bic 162.5
#modelo = sm.ols(formula='mpg ~ wt + disp + hp',data=base)

#aic 165.1  bic 169.5
#modelo = sm.ols(formula='mpg ~  disp + cyl',data=base)

modelo = sm.ols(formula='mpg ~ drat + vs',data=base)
modelo = modelo.fit()

print(modelo.summary())

residuos = modelo.resid
plt.hist(residuos, bins=20)
plt.xlabel("Residuos")
plt.ylabel("Frequencia")
plt.title("Histograma de Residous")

stats.probplot(residuos,dist='norm',plot=plt)
plt.title("Q-Q plot de Residous")

#h0 - dados estão normalmente distribuidos
# p <= 0.05 rejeito a hipostese nula, (não estão normalmente distribuidos)
# p > 0.05 não é possivel rejeitar o h0
stat, pval = stats.shapiro(residuos)
print(f'Shapiro-Wilk statistics: {stat:.3f}, p-value: {pval:.3f}')

plt.show()

plt.tight_layout()