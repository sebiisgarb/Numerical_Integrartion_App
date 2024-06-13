import tkinter.messagebox
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad
from sympy import symbols, sympify, lambdify, Function
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os.path
import subprocess
import sys

# def install_requirements():
#     try:
#         subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to install dependencies: {e}")
#         sys.exit(1)

# install_requirements()


desktop = os.path.expanduser("~/Desktop")

window = Tk()
window.title("Integrare numerica")

def resolve():
    global y_pts, x_pts, rezultat
    x_func = symbols('x')
    sin_exp = np.sin
    sin_exp = Function('sin')
    cos_exp = np.cos
    cos_exp = Function('cos')
    tan_exp = np.tan
    tan_exp = Function('tan')
    exp_exp = np.exp
    exp_exp = Function('exp')
    log_exp = np.log
    log_exp = Function('log')
    pi_exp = np.pi
    pi_exp = Function('pi')
    e_exp = np.e
    e_exp = Function('e')
    sin_exp, cos_exp, tan_exp, exp_exp, log_exp, pi_exp, e_exp = symbols('sin, cos, tan, exp, log, pi, e', cls=Function)
    predefined_functions = {'sin', 'cos', 'tan', 'exp', 'log', 'pi', 'e'}

    f_sympy = sympify(function_entry.get())
    f_numeric = lambdify(x_func, f_sympy, modules=['numpy'])

    '''try:
        f_sympy = sympify(function_entry.get(), locals=Function)
        f_numeric = lambdify(x_func, f_sympy, modules=['numpy'])
    except Exception:
        tkinter.messagebox.showerror("Eroare", "Introduceti o functie VALIDA!")
        return'''

    try:
        a = float(sympify(a_entry.get()).evalf())
    except ValueError:
        tkinter.messagebox.showerror("Eroare", "Capatul intervalului din stanga trebuie sa fie NUMAR!")
        return

    try:
        b = float(sympify(b_entry.get()).evalf())
    except ValueError:
        tkinter.messagebox.showerror("Eroare", "Capatul intervalului din dreapta trebuie sa fie NUMAR!")
        return

    if a > b:
        tkinter.messagebox.showerror("Eroare", "Capatul intervalului din STANGA trebuie sa aibe o valoare mai mica decat capatul intervalului din DREAPTA!")
        return
    try:
        n = int(n_entry.get())
    except ValueError:
        tkinter.messagebox.showerror("Eroare", "Numarul de subintervale trebuie sa fie NUMAR NATURAL!")
        return
    if n <= 0:
        tkinter.messagebox.showerror("Eroare", "Numarul de subintervale trebuie sa fie MAI MARE decat 0!")
        return

    method = method_var.get()

    if method == "Dreptunghiuri":
        rezultat, x_pts, y_pts = rectangular_rule(f_numeric, a, b, n)
    elif method == "Trapez":
        rezultat, x_pts, y_pts = trapezoidal_rule(f_numeric, a, b)
    elif method == "Simpson":
        if n % 2 == 1:
            tkinter.messagebox.showerror("Eroare", "Introduceti un numar PAR de subintervale!")
            return
        rezultat, x_pts, y_pts = simpson_rule(f_numeric, a, b)
    elif method == "Trapez Compozit":
        rezultat, x_pts, y_pts = composite_trapezoidal_rule(f_numeric, a, b, n)
    elif method == "Gauss-Legendre":
        rezultat, x_pts, y_pts = gauss_legendre_rule(f_numeric, a, b, n)
    elif method == "Gauss-Chebyshev":
        rezultat, x_pts, y_pts = gauss_chebyshev_rule(f_numeric, a, b, n)
    elif method == "Simpson Compozit":
        if n % 2 == 1:
            tkinter.messagebox.showerror("Eroare", "Introduceti un numar PAR de subintervale!")
            return
        rezultat, x_pts, y_pts = composite_simpson_rule(f_numeric, a, b, n)

    result_label.config(text=f"Integrala este:\n {rezultat:6f}")

    try:
        exact_value, _ = quad(f_numeric, a, b)
    except Exception as e:
        tkinter.messagebox.showerror("Eroare", f"Nu s-a putut calcula valoarea exactă a integralei: {e}")
        return

        # Calculate the error
    error = abs(exact_value - rezultat)
    error_label.config(text=f"Eroarea este:\n {error:6f}")

    plot_graph(a, b, x_pts, y_pts, method, f_numeric)

# window config
Label(text="Funcția de integrat f(x):").grid(row=0, column=0, padx=10, pady=10)
function_entry = Entry(width=30)
function_entry.grid(row=0, column=1, padx=10, pady=10)
function_entry.insert(0, "sin(x)")

Label(text="Limita stanga a:").grid(row=1, column=0, padx=10, pady=10)
a_entry = Entry(width=10)
a_entry.grid(row=1, column=1, padx=10, pady=10)
a_entry.insert(0, "0")

Label(text="Limita dreapta b:").grid(row=2, column=0, padx=10, pady=10)
b_entry = Entry(width=10)
b_entry.grid(row=2, column=1, padx=10, pady=10)
b_entry.insert(0, "1")

Label(text="Numărul de subintervale n:").grid(row=3, column=0, padx=10, pady=10)
n_entry = Entry(width=10)
n_entry.grid(row=3, column=1, padx=10, pady=10)
n_entry.insert(0, "20")

method_var = StringVar(value="Dreptunghiuri")
Radiobutton(text="Metoda Dreptunghiurilor", variable=method_var, value="Dreptunghiuri").grid(row=4, column=0, padx=10, pady=10)
Radiobutton(text="Metoda Trapezului", variable=method_var, value="Trapez").grid(row=4, column=1, padx=10, pady=10)
Radiobutton(text="Metoda Simpson", variable=method_var, value="Simpson").grid(row=4, column=2, padx=10, pady=10)
Radiobutton(text="Metoda Trapezelor Compozită", variable=method_var, value="Trapez Compozit").grid(row=5, column=0, padx=10, pady=10)
Radiobutton(text="Metoda Gauss-Legendre", variable=method_var, value="Gauss-Legendre").grid(row=5, column=1, padx=10, pady=10)
Radiobutton(text="Metoda Gauss-Chebyshev", variable=method_var, value="Gauss-Chebyshev").grid(row=5, column=2, padx=10, pady=10)
Radiobutton(text="Metoda Simpson Compozită", variable=method_var, value="Simpson Compozit").grid(row=6, column=1, padx=10, pady=10)


calculate_button = Button(text="Calculează Integrala", command=resolve)
calculate_button.grid(row=7, column=0, columnspan=3, padx=10, pady=10)

result_label = Label(text="Integrala este: ")
result_label.grid(row=8, column=0, columnspan=3, padx=10, pady=10)

error_label = Label(text="Eroarea este: ")
error_label.grid(row=9, column=0, columnspan=3, padx=10, pady=10)


# getting the formula from the user and reformatting it for usage

def rectangular_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
    y = f(x)
    result = np.sum(y) * h
    return result, x, y

def trapezoidal_rule(f, a, b):
    h = (b - a)
    x0 = a
    x1 = b
    y0 = f(x0)
    y1 = f(x1)
    result = (h / 2) * (y0 + y1)
    return result, np.array([x0, x1]), np.array([y0, y1])
def simpson_rule(f, a, b):
    h = (b - a) / 2
    x0 = a
    x1 = (a + b) / 2
    x2 = b
    y0 = f(x0)
    y1 = f(x1)
    y2 = f(x2)
    result = (h / 3) * (y0 + 4 * y1 + y2)
    return result, np.array([x0, x1, x2]), np.array([y0, y1, y2])
def composite_trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    result = (y[0] + 2 * sum(y[1:n]) + y[n]) * h / 2
    return result, x, y
def gauss_legendre_rule(f, a, b, n):
    # Calculăm valorile beta pentru matricea tridiagonală
    beta = np.array([i / np.sqrt(4 * i * i - 1) for i in range(1, n)])
    # Construim matricea tridiagonală
    T = np.diag(beta, -1) + np.diag(beta, 1)
    # Calculăm valorile proprii (nodurile) și vectorii proprii
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    # Nodurile sunt valorile proprii
    x = eigenvalues
    # Greutățile sunt calculate folosind vectorii proprii
    w = 2 * eigenvectors[0, :] ** 2
    # Transformăm nodurile la intervalul [a, b]
    x_transformed = 0.5 * (x + 1) * (b - a) + a
    # Calculăm integralul folosind nodurile și greutățile transformate
    result = 0.5 * (b - a) * np.sum(w * f(x_transformed))

    return result, x_transformed, f(x_transformed)
def gauss_chebyshev_rule(f, a, b, n):
    # Calculăm nodurile (rădăcinile polinomului Chebyshev)
    k = np.arange(1, n + 1)
    x = np.cos((2 * k - 1) * np.pi / (2 * n))
    # Transformăm nodurile la intervalul [a, b]
    x_transformed = 0.5 * (x + 1) * (b - a) + a
    # Greutățile pentru Gauss-Chebyshev sunt constante
    w = np.pi / n
    # Calculăm integralul folosind formula cuadraturii Gauss-Chebyshev
    result = w * np.sum(f(x_transformed) / np.sqrt(1 - x ** 2))
    return result, x_transformed, f(x_transformed)
def composite_simpson_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    result = y[0] + y[-1] + 4 * np.sum(y[1:n:2]) + 2 * np.sum(y[2:n - 1:2])
    result *= h / 3
    return result, x, y

def plot_graph(a, b, x, y, method, f):
    # Crearea ferestrei pentru animație
    animation_window = Toplevel(window)
    animation_window.title("Animatie grafic")

    # Crearea figurii matplotlib și a axelor
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=animation_window)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    # Setarea datelor pentru plot
    x_fill = np.linspace(a, b, 1000)
    y_fill = f(x_fill)
    # ax.fill_between(x_fill, 0, y_fill, color='skyblue', alpha=0.4)

    ax.set_ylim(min(y_fill) * 1.1, max(y_fill) * 1.1)
    line, = ax.plot([], [], 'r-', label=f'f(x)={function_entry.get()}')
    ax.legend(loc='upper left')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        ax.clear()
        ax.fill_between(x_fill, 0, y_fill, color='skyblue', alpha=0.4)
        line, = ax.plot(x, y, 'r-', label=f'f(x)={function_entry.get()}')
        ax.legend(loc="upper left")
        if method == "Dreptunghiuri":
            for i in range(1, frame + 1):
                midpoint = [(x[i - 1] + x[i]) / 2, (y[i - 1] + y[i]) / 2]
                ax.plot([x[i - 1], x[i - 1]], [0, midpoint[1]], 'b')
                ax.plot([x[i], x[i]], [0, midpoint[1]], 'b')
                if i < len(x):
                    ax.plot([x[i - 1], x[i]], [midpoint[1], midpoint[1]], 'b')
        elif method == "Trapez":
            for i in range(frame):
                ax.plot([x[i], x[i], x[i + 1], x[i + 1]], [0, y[i], y[i + 1], 0], 'b')
        elif method == "Simpson":
            for i in range(0, frame - 1, 2):
                if i < len(x) - 2:
                    xi = np.linspace(x[i], x[i + 2], 100)
                    fi = f(x[i]) + (f(x[i + 2]) - f(x[i])) / (x[i + 2] - x[i]) * (xi - x[i])
                    fi += (f(x[i + 1]) - (f(x[i]) + (f(x[i + 2]) - f(x[i])) / (x[i + 2] - x[i]) * (
                            x[i + 1] - x[i]))) * (xi - x[i]) * (xi - x[i + 2]) / (
                                      (x[i + 1] - x[i]) * (x[i + 1] - x[i + 2]))
                    ax.plot(xi, fi, 'b')
        elif method == "Trapez Compozit":
            for i in range(frame):
                ax.plot([x[i], x[i], x[i + 1], x[i + 1]], [0, y[i], y[i + 1], 0], 'b')

        elif method == "Gauss-Legendre":
            # Punctele sunt precalculate și stocate în x, iar ponderile în y pentru simplitate aici
            for i in range(frame):
                ax.plot([x[i], x[i]], [0, y[i]], 'ro')  # Punctele Gauss-Legendre

        elif method == "Gauss-Chebyshev":
            # Similar cu Gauss-Legendre
            for i in range(frame):
                ax.plot([x[i], x[i]], [0, y[i]], 'ro')  # Punctele Gauss-Chebyshev

        elif method == "Simpson Compozit":
            for i in range(0, frame - 1, 2):
                if i < len(x) - 2:
                    xi = np.linspace(x[i], x[i + 2], 100)
                    # Parabola prin trei puncte
                    fi = f(x[i]) + (f(x[i + 2]) - f(x[i])) / (x[i + 2] - x[i]) * (xi - x[i])
                    fi += (f(x[i + 1]) - (
                                f(x[i]) + (f(x[i + 2]) - f(x[i])) / (x[i + 2] - x[i]) * (x[i + 1] - x[i]))) * (
                                      xi - x[i]) * (xi - x[i + 2]) / ((x[i + 1] - x[i]) * (x[i + 1] - x[i + 2]))
                    ax.plot(xi, fi, 'b')

        return line,

    # Crearea și rularea animației
    ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=False, repeat=False)

    # Funcție pentru salvarea animației
    def save_animation():
        ani.save(f'{desktop}/animation.gif', writer='ffmpeg', fps=30)
        print("Animation saved successfully.")

    # Buton pentru salvarea animației

    save_button = Button(animation_window, text="Salvează Animația", command=save_animation)
    save_button.pack()

    # Afișarea ferestrei
    animation_window.mainloop()


window.mainloop()
