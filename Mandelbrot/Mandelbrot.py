import time
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from PIL import Image

def mandelbrot(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    return (r1, r2, np.array([[mandelbrot(complex(r, i), max_iter) for r in r1] for i in r2]))

def parallel_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, cores):
    pool = mp.Pool(cores)
    rows = pool.map(calc_row, [(y, xmin, xmax, width, max_iter) for y in np.linspace(ymin, ymax, height)])
    pool.close()
    pool.join()
    return (np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height), np.array(rows))

def calc_row(args):
    y, xmin, xmax, width, max_iter = args
    r1 = np.linspace(xmin, xmax, width)
    row = np.zeros(width, dtype=int)
    for i, r in enumerate(r1):
        row[i] = mandelbrot(complex(r, y), max_iter)
    return row

def plot_mandelbrot(x, y, z, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(z.T, extent=(x.min(), x.max(), y.min(), y.max()), cmap='hot', origin='lower')
    plt.title(title)
    plt.colorbar(label='Iterations')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.show()

def main():
    xmin, xmax, ymin, ymax = -2.0, 2.0, -2.0, 2.0
    sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    max_iter = 256
    num_cores = [1, 2, 3, 4]

    for width, height in sizes:
        print(f"Size: {width}x{height}")
        
        # Seri hesaplama
        start_serial = time.time()
        serial_result = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
        end_serial = time.time()
        serial_time = end_serial - start_serial
        
        # Paralel hesaplama
        parallel_times = []
        for cores in num_cores:
            start_parallel = time.time()
            parallel_result = parallel_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, cores)
            end_parallel = time.time()
            parallel_times.append(end_parallel - start_parallel)

        speedup = [serial_time / t for t in parallel_times]
        efficiency = [s / c for s, c in zip(speedup, num_cores)]

        # Grafikler
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Time grafikleri
        plt.subplot(1, 3, 1)
        axes[0].plot(num_cores, parallel_times, marker='o')
        #axes[0].axhline(y=serial_time, color='r', linestyle='--')
        axes[0].set_title('Time vs Number of Cores')
        plt.title(f'Time (Width x Height: {width} x {height})')
        axes[0].set_xlabel('Number of Cores')
        axes[0].set_ylabel('Time (s)')

        # Speedup grafikleri
        plt.subplot(1, 3, 2)
        axes[1].plot(num_cores, speedup, marker='o')
        axes[1].set_title('Speedup vs Number of Cores')
        plt.title(f'Speedup (Width x Height: {width} x {height})')
        axes[1].set_xlabel('Number of Cores')
        axes[1].set_ylabel('Speedup')

        # Efficiency grafikleri
        plt.subplot(1, 3, 3)
        axes[2].plot(num_cores, efficiency, marker='o')
        axes[2].set_title('Efficiency vs Number of Cores')
        plt.title(f'Efficiency (Width x Height: {width} x {height})')
        axes[2].set_xlabel('Number of Cores')
        axes[2].set_ylabel('Efficiency')

        plt.tight_layout()
        plt.show()
        #plot_mandelbrot(*parallel_result, title='Parallel Mandelbrot Set')
        
    plot_mandelbrot(*parallel_result, title='Parallel Mandelbrot Set') # 2048 * 2048

if __name__ == "__main__":
    main()
