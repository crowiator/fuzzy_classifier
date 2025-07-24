import pickle

# Otvorenie .pkl súboru v režime 'rb' (read binary)
with open('ds2_traditionalb.pkl', 'rb') as file:
    data = pickle.load(file)

# Vypíše obsah načítaného súboru
print(data)