import poly
import rbf

def go(kernel):
    if kernel == 'poly':
        poly.run()
    elif kernel == 'rbf':
        rbf.run()
    else:
        pass
        