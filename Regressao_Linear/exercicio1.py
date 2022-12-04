import matplotlib.pyplot as plt
import math


def regressao_linear(x,y,x2,y2,xy,n):
    beta = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x2) - math.pow(sum(x),2))
    alpha = (sum(y) - beta*sum(x))/n
    return (alpha,beta)

def f(x, ab : tuple[float,float])->float:
    return ab[0] + ab[1]*x

def exercicio1():
    x = [22,24,26,31,32,35,36,37,41,43,44,50,55,60,70]
    y = [550,565,565,600,600,610,610,600,700,710,750,750,800,950,950]
    n = len(x)

    x2 = [math.pow(elemento,2) for elemento in x]
    y2 = [math.pow(elemento,2) for elemento in y]
    xy = [x[i]*y[i] for i in range(n)]
    f1 = regressao_linear(x,y,x2,y2,xy,n)
    print(f1)

    x0 = 0
    xn = 100
    y0 = f(x0,f1)
    yn = f(xn,f1)

    ex1c = {19:f(19,f1), 29:f(29,f1), 80:f(80,f1)}
    print(ex1c)

    plt.plot(x,y,'o')
    plt.plot([x0,xn],[y0,yn])
    plt.xlabel('Idade')
    plt.ylabel('Valor do Aluguel')
    plt.show()


def exercicio2():
    x = [28, 30, 31, 33, 35, 36, 39, 41, 45, 49, 52, 53, 60, 66, 73]
    y = [400, 400, 410, 410, 400, 420, 410, 420, 450, 620, 680, 650, 850, 800, 1100]
    #logy = [math.log(elemento) for elemento in y]
    n = len(x)
    x2 = [math.pow(elemento,2) for elemento in x]
    y2 = [math.pow(elemento,2) for elemento in y]
    xy = [x[i]*y[i] for i in range(n)]
    f1 = regressao_linear(x,y,x2,y2,xy,n)
    print(f1)

    x0 = 18
    xn = 90
    y0 = f(x0,f1)
    yn = f(xn,f1)
    ex2c = {19:f(19,f1), 55:f(55,f1), 85:f(85,f1)}
    print(ex2c)

    plt.plot(x,y,'o')
    plt.plot([x0,xn],[y0,yn])
    plt.xlabel('Idade')
    plt.ylabel('Valor do Aluguel')
    plt.show()

def reverte_log(n):
    return math.pow(math.e, n)

def exercicio2log():
    x = [28, 30, 31, 33, 35, 36, 39, 41, 45, 49, 52, 53, 60, 66, 73]
    y = [400, 400, 410, 410, 400, 420, 410, 420, 450, 620, 680, 650, 850, 800, 1100]
    logy = [math.log(elemento) for elemento in y]
    print(logy)
    n = len(x)
    x2 = [math.pow(elemento,2) for elemento in x]
    y2 = [math.pow(elemento,2) for elemento in logy]
    xy = [x[i]*logy[i] for i in range(n)]
    f1 = regressao_linear(x,logy,x2,y2,xy,n)
    print(f1)

    x0 = 18
    xn = 90
    logy0 = f(x0,f1)
    logyn = f(xn,f1)
    y0 = reverte_log(logy0)
    yn = reverte_log(logyn)
    ex2c = {19:reverte_log(f(19,f1)), 55:reverte_log(f(55,f1)), 85:reverte_log(f(85,f1))}
    print(ex2c)

    plt.plot(x,logy,'o')
    plt.plot([x0,xn],[logy0,logyn])
    plt.xlabel('Idade')
    plt.ylabel('log(Valor do Aluguel)')
    plt.show()


exercicio2()