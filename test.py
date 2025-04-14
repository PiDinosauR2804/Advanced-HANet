a = 1

def func():
    def _nest_func():
        global a
        a = 2
    
    _nest_func()
    global a
    a = 3

    
func()
print(a)
