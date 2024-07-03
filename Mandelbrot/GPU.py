import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import time

@cuda.jit(device=True)
def mandelbrot_kernel(c, max_iter):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iter:
        z = z*z + c
        n += 1
    return n

@cuda.jit
def mandelbrot_gpu(xmin, xmax, ymin, ymax, width, height, max_iter, output):
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y
    
    dx = (xmax - xmin) / width
    dy = (ymax - ymin) / height
    
    for x in range(startX, width, gridX):
        real = xmin + x * dx
        for y in range(startY, height, gridY):
            imag = ymin + y * dy
            c = complex(real, imag)
            output[y, x] = mandelbrot_kernel(c, max_iter)

def plot_mandelbrot(x, y, z, title):
    plt.figure(figsize=(10, 10))
    plt.imshow(z.T, extent=(x.min(), x.max(), y.min(), y.max()), cmap='hot', origin='lower')
    plt.title(title)
    plt.colorbar(label='Iterations')
    plt.xlabel('Re(c)')
    plt.ylabel('Im(c)')
    plt.show()

def main():
    sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    thread_sayilari = [1, 256, 512, 1024]
    
    for width, height in sizes:
        print(f"Size: {width}x{height}")
        
        output_times = []
        efficiencies = []
        speedups = []
        for threadsperblock in thread_sayilari:
            xmin, xmax, ymin, ymax = -2.0, 2.0, -2.0, 2.0
            max_iter = 256
            
            blockspergrid_x = (width + (threadsperblock - 1)) // threadsperblock
            blockspergrid_y = (height + (threadsperblock - 1)) // threadsperblock
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            
            output = np.zeros((height, width), dtype=np.int32)
            
            start_gpu = time.time()
            mandelbrot_gpu[blockspergrid, threadsperblock](xmin, xmax, ymin, ymax, width, height, max_iter, output)
            end_gpu = time.time()
            gpu_time = end_gpu - start_gpu
            
            output_times.append(gpu_time)
            print(output_times)
            
            # Efficiency hesaplama
            single_thread_time = output_times[0]
            efficiency = single_thread_time / gpu_time / threadsperblock
            efficiencies.append(efficiency)
            
            # Speedup hesaplama
            speedup = single_thread_time / gpu_time
            speedups.append(speedup)
            
            #plot_mandelbrot(np.linspace(xmin, xmax, width), np.linspace(ymin, ymax, height), output, title=f'GPU Mandelbrot Set ({width}x{height}, Threads per Block: {threadsperblock})')
        

        # Grafikleri çiz
        plt.figure(figsize=(12, 10))
        
        # Zaman grafiði
        plt.subplot(3, 1, 1)
        plt.plot(thread_sayilari, output_times, marker='o')
        plt.title(f'Elapsed Time vs Threads per Block (Size: {width}x{height})')
        plt.xlabel('Threads per Block')
        plt.ylabel('Elapsed Time (s)')
        plt.grid(True)
        
        # Efficiency grafiði
        plt.subplot(3, 1, 2)
        plt.plot(thread_sayilari, efficiencies, marker='o')
        plt.title(f'Efficiency vs Threads per Block (Size: {width}x{height})')
        plt.xlabel('Threads per Block')
        plt.ylabel('Efficiency')
        plt.grid(True)
        
        # Speedup grafiði
        plt.subplot(3, 1, 3)
        plt.plot(thread_sayilari, speedups, marker='o')
        plt.title(f'Speedup vs Threads per Block (Size: {width}x{height})')
        plt.xlabel('Threads per Block')
        plt.ylabel('Speedup')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

