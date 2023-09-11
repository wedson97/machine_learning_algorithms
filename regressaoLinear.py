from numpy import *

class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # metodos privados
        self.__correlation_coefficient = self.__correlacao()
        self.__inclination = self.__inclinacao()
        self.__intercept = self.__interceptacao()


    def __correlacao(self):
        # cov faz o calculo da covarianca e retorna uma matriz
        covariacao = cov(self.x, self.y, bias=True)[0][1]
        # var faz o calculo de variancia
        variancia_x = var(self.x)
        variancia_y = var(self.y)
        return covariacao/sqrt(variancia_x * variancia_y)
    
    def __inclinacao(self):
        # std faz o calculo de desvio padrio
        stdx = std(self.x)
        stdy = std(self.y)
        return self.__correlation_coefficient * (stdy/stdx)
        
    def __interceptacao(self):
        # mean calcula a media
        mediax = mean(self.x)
        mediay = mean(self.y)
        return mediay - mediax * self.__inclination
    
    def previsao(self, valor):
        return self.__intercept + (self.__inclination * valor)

x = array([1,2,3,4,5,7])
y = array([2,4,6,8,10,11])

lr = LinearRegression(x,y)
print(lr.previsao(8))
#Resultado = 12