import matplotlib.pyplot as plt

def dxdt(x,y,a,d):
     """dxdt is the change in prey population size (x is the number of prey)"""
     return x*(1-x)-(a*x*y)/(d+x)

#define class and set a and d in class dx dy can then be methods
def dydt(x,y,a,d,b):
     """dydt is the change in predator population size (y is the num of predators)"""
     return b*y*(1-y/x)

a=1
d=0.1
b=float(input("Enter the rate of predation b ∈ [0.1, 0.5] : "))

x=60
y=60
t=1/1000
periods=20

#use these to plot the population changes over time
y_pop=[]
x_pop=[]


for i in range(1, periods+1):
    x=int(x+t*dxdt(x,y,a,d))
    y=int(y+t*dydt(x,y,a,d,b))
    x_pop.append(x)
    y_pop.append(y)

    #if ((t*i)%1 == 0):
    print("Population X:",int(x),"Population y:",int(y), "at period", i)

def plot_curve(x_pop, y_pop):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(x_pop, color='b')
    ax.plot(y_pop, color='r')

    plt.show()

plot_curve(x_pop, y_pop)

