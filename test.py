try:
    a = 1/1
except ZeroDivisionError as e:
    print(f"Error: {e}")
finally:
    print("This will always execute.")