class Num:
    def __init__(self, num):
        self.num = num
        
        
a = Num(1)
def func(b):
    b = Num(2)
    
print(a.num)
func(a)
print(a.num)