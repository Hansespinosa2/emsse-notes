print(1+2)
print("Hello world")

using Plots
using ControlSystems
bodeplot(tf(1,[1,2,1]))
