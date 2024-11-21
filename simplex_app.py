import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# Clase Simplex
class Simplex:
    def __init__(self, c, A, b):
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.n = len(self.c)
        self.m = len(self.b)
        self.create_tableau()

    def create_tableau(self):
        self.table = np.zeros((self.m + 1, self.n + self.m + 1))
        self.table[:self.m, :self.n] = self.A
        self.table[:self.m, self.n:self.n + self.m] = np.eye(self.m)
        self.table[:self.m, -1] = self.b
        self.table[-1, :self.n] = -self.c

        self.df = pd.DataFrame(self.table)
        self.df.columns = [f'x{i + 1}' for i in range(self.n)] + [f's{i + 1}' for i in range(self.m)] + ['b']
        self.df.index = [f's{i + 1}' for i in range(self.m)] + ['z']

    def optimize(self):
        iteration = 1
        steps = []
        operations = []  # Guardar las operaciones (variable entra/sale)
        while True:
            min_value = np.min(self.table[-1, :-1])
            if min_value >= -1e-8:
                break
            row_pivot, col_pivot = self.pivot()
            if row_pivot is None:
                break
            entering_var = self.df.columns[col_pivot]
            leaving_var = self.df.index[row_pivot]
            self.pivot_operation(row_pivot, col_pivot)
            steps.append(self.df.copy())
            operations.append((entering_var, leaving_var))
            iteration += 1
        return steps, operations

    def pivot(self):
        epsilon = 1e-8
        min_value = np.min(self.table[-1, :-1])
        if min_value >= -epsilon:
            return None, None
        col_pivot = np.argmin(self.table[-1, :-1])
        if np.all(self.table[:-1, col_pivot] <= epsilon):
            raise ValueError("El problema es no acotado.")
        valid_rows = self.table[:-1, col_pivot] > epsilon
        numerator = self.table[:-1, -1]
        denominator = self.table[:-1, col_pivot]
        ratios = np.full_like(numerator, np.inf)
        ratios[valid_rows] = numerator[valid_rows] / denominator[valid_rows]
        row_pivot = np.argmin(ratios)
        return row_pivot, col_pivot

    def pivot_operation(self, row_pivot, col_pivot):
        pivot_element = self.table[row_pivot, col_pivot]
        self.table[row_pivot, :] /= pivot_element
        for i in range(len(self.table)):
            if i != row_pivot:
                factor = self.table[i, col_pivot]
                self.table[i, :] -= factor * self.table[row_pivot, :]

        self.df.iloc[:, :] = self.table
        entering_var = self.df.columns[col_pivot]
        leaving_var = self.df.index[row_pivot]
        self.df.index = [entering_var if idx == leaving_var else idx for idx in self.df.index]

    def mostrar_solucion_optima(self):
        solucion = {var: 0 for var in self.df.columns[:-1]}
        for idx, row_name in enumerate(self.df.index[:-1]):
            basic_var = row_name
            var_value = self.df.iloc[idx, -1]
            solucion[basic_var] = var_value
        return solucion, self.df.iloc[-1, -1]

    def sensibilidad_costo(self, variable_index, rango):
        original_c = self.c.copy()
        valores_objetivo = []
        coeficientes = []

        for nuevo_coef in rango:
            self.c[variable_index] = nuevo_coef
            self.create_tableau()
            self.optimize()
            _, valor_z = self.mostrar_solucion_optima()
            valores_objetivo.append(valor_z)
            coeficientes.append(nuevo_coef)

        self.c = original_c.copy()
        self.create_tableau()

        return coeficientes, valores_objetivo


# Aplicación en Streamlit
st.title("Método Simplex con Análisis de Sensibilidad")

# Entradas
st.sidebar.header("Definir el problema")
n = st.sidebar.number_input("Número de variables de decisión", min_value=1, value=2)
m = st.sidebar.number_input("Número de restricciones", min_value=1, value=2)

st.sidebar.subheader("Matriz de restricciones (A)")
A = []
for i in range(m):
    fila = st.sidebar.text_input(f"Fila {i + 1} (separada por comas)", value="1,1")
    A.append(list(map(float, fila.split(","))))

b = st.sidebar.text_input("Vector b (separado por comas)", value="4,6")
b = list(map(float, b.split(",")))

c = st.sidebar.text_input("Vector de costos c (separado por comas)", value="3,5")
c = list(map(float, c.split(",")))

variable_index = st.sidebar.number_input("Índice de variable para análisis de sensibilidad (x1 = 0, x2 = 1, ...)", min_value=0, max_value=n-1, value=0)
rango_min = st.sidebar.number_input("Valor mínimo del rango de sensibilidad", value=1.0)
rango_max = st.sidebar.number_input("Valor máximo del rango de sensibilidad", value=5.0)
num_puntos = st.sidebar.number_input("Número de puntos en el rango", min_value=2, value=10)

# Crear y optimizar el modelo
try:
    simplex = Simplex(c, A, b)
    steps, operations = simplex.optimize()
    solucion, valor_objetivo = simplex.mostrar_solucion_optima()

    # Mostrar pasos intermedios
    st.subheader("Pasos Intermedios")
    for i, (step, operation) in enumerate(zip(steps, operations)):
        st.write(f"**Iteración {i + 1}:** Variable **{operation[0]}** entra, Variable **{operation[1]}** sale.")
        st.dataframe(step)

    # Mostrar solución
    st.subheader("Solución Óptima")
    st.write("Variables:")
    st.write(solucion)
    st.write(f"Valor óptimo de la función objetivo (z): {valor_objetivo}")

    # Análisis de sensibilidad
    st.subheader("Análisis de Sensibilidad")
    rango = np.linspace(rango_min, rango_max, num_puntos)
    coeficientes, valores_objetivo = simplex.sensibilidad_costo(variable_index, rango)

    # Mostrar gráfica
    plt.figure(figsize=(8, 5))
    plt.plot(coeficientes, valores_objetivo, marker="o", linestyle="-", color="blue")
    plt.title(f"Análisis de sensibilidad para x{variable_index + 1}")
    plt.xlabel(f"Coeficiente de x{variable_index + 1}")
    plt.ylabel("Valor óptimo de la función objetivo (z)")
    plt.grid(True)
    st.pyplot(plt)

except Exception as e:
    st.error(f"Error: {e}")
