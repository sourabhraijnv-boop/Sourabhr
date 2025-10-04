from sympy import symbols, diff
J, w = symbols('J, w')
J = 1/w
dj_dw=diff(J,w)
print(dj_dw.subs([(w,2)]))
print(dj_dw)