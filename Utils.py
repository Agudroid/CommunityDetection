import matplotlib.pyplot as plt
import networkx as nx

# Crear un grafo de NetworkX
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])

# Visualizar el grafo utilizando Matplotlib
plt.figure(figsize=(6, 4))
nx.draw_networkx(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title('Grafo de NetworkX')
plt.axis('off')
plt.show()

# Datos para los gráficos de líneas
x = [1, 2, 3, 4, 5]
y1 = [3, 5, 2, 7, 4]
y2 = [1, 6, 4, 3, 8]

# Crear una figura con dos subplots para los gráficos de líneas
plt.figure(figsize=(8, 4))

# Primer gráfico de líneas
plt.subplot(1, 2, 1)
plt.plot(x, y1, marker='o', color='blue')
plt.xlabel('X')
plt.ylabel('Y1')
plt.title('Gráfico 1')

# Segundo gráfico de líneas
plt.subplot(1, 2, 2)
plt.plot(x, y2, marker='s', color='red')
plt.xlabel('X')
plt.ylabel('Y2')
plt.title('Gráfico 2')

# Ajustar los espacios entre subplots
plt.tight_layout()

# Mostrar la figura con los gráficos
plt.show()

def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def find_closest_prime(n):
    while n >= 2:
        if is_prime(n):
            return n
        n -= 1
    return None

n = find_closest_prime(20)
print(n)