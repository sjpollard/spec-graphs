import numpy as np
from matplotlib import pyplot as plt

def compare_lbm_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    plt.title("LBM Tiny - GNU Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[0], c="red", linestyle="solid", marker="x", label="TX2 OMP 16x4")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[0], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[0], c="green", linestyle="solid", marker="x", label="G2 OMP 16x4")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[0], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_lbm_t_gnu_speedup.pdf")
    plt.clf()

def compare_lbm_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    plt.title("LBM Tiny - GNU Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1400])
    plt.plot([1,2,4,8], tx2_omp_times.T[0], c="red", linestyle="solid", marker="x", label="TX2 OMP 16x4")
    plt.plot([1,2,4,8], tx2_mpi_times.T[0], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[0], c="green", linestyle="solid", marker="x", label="G2 OMP 16x4")
    plt.plot([1,2,4,8], g2_mpi_times.T[0], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_lbm_t_gnu_time.pdf")
    plt.clf()

def compare_soma_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    plt.title("SOMA Tiny - GNU Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[1], c="red", linestyle="solid", marker="x", label="TX2 OMP 2x32")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[1], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[1], c="green", linestyle="solid", marker="x", label="G2 OMP 2x32")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[1], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_soma_t_gnu_speedup.pdf")
    plt.clf()

def compare_soma_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    plt.title("SOMA Tiny - GNU Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 2000])
    plt.plot([1,2,4,8], tx2_omp_times.T[1], c="red", linestyle="solid", marker="x", label="TX2 OMP 2x32")
    plt.plot([1,2,4,8], tx2_mpi_times.T[1], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[1], c="green", linestyle="solid", marker="x", label="G2 OMP 2x32")
    plt.plot([1,2,4,8], g2_mpi_times.T[1], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_soma_t_gnu_time.pdf")
    plt.clf()

def compare_tealeaf_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    plt.title("TeaLeaf Tiny - GNU Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 9])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[2], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[2], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[2], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[2], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_tealeaf_t_gnu_speedup.pdf")
    plt.clf()

def compare_tealeaf_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    plt.title("TeaLeaf Tiny - GNU Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1200])
    plt.plot([1,2,4,8], tx2_omp_times.T[2], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_times.T[2], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[2], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_mpi_times.T[2], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_tealeaf_t_gnu_time.pdf")
    plt.clf()

def compare_clvleaf_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    plt.title("CloverLeaf Tiny - GNU Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[3], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[3], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[3], c="green", linestyle="solid", marker="x", label="G2 OMP 4x16")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[3], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_clvleaf_t_gnu_speedup.pdf")
    plt.clf()

def compare_clvleaf_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    plt.title("CloverLeaf Tiny - GNU Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1200])
    plt.plot([1,2,4,8], tx2_omp_times.T[3], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_times.T[3], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[3], c="green", linestyle="solid", marker="x", label="G2 OMP 4x16")
    plt.plot([1,2,4,8], g2_mpi_times.T[3], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_clvleaf_t_gnu_time.pdf")
    plt.clf()

def compare_miniswp_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    plt.title("Minisweep Tiny - GNU Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 6])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[4], c="red", linestyle="solid", marker="x", label="TX2 OMP 2x32")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[4], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[4], c="green", linestyle="solid", marker="x", label="G2 OMP 2x32")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[4], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_miniswp_t_gnu_speedup.pdf")
    plt.clf()

def compare_miniswp_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    plt.title("Minisweep Tiny - GNU Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1200])
    plt.plot([1,2,4,8], tx2_omp_times.T[4], c="red", linestyle="solid", marker="x", label="TX2 OMP 2x32")
    plt.plot([1,2,4,8], tx2_mpi_times.T[4], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[4], c="green", linestyle="solid", marker="x", label="G2 OMP 2x32")
    plt.plot([1,2,4,8], g2_mpi_times.T[4], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_miniswp_t_gnu_time.pdf")
    plt.clf()

def compare_pot3d_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    plt.title("POT3D Tiny - GNU Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 9])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[5], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[5], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[5], c="green", linestyle="solid", marker="x", label="G2 OMP 4x16")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[5], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_pot3d_t_gnu_speedup.pdf")
    plt.clf()

def compare_pot3d_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    plt.title("Minisweep Tiny - GNU Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1800])
    plt.plot([1,2,4,8], tx2_omp_times.T[5], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_times.T[5], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[5], c="green", linestyle="solid", marker="x", label="G2 OMP 4x16")
    plt.plot([1,2,4,8], g2_mpi_times.T[5], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_pot3d_t_gnu_time.pdf")
    plt.clf()

def compare_sph_exa_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    plt.title("SPH-EXA Tiny - GNU Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[6], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[6], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[6], c="green", linestyle="solid", marker="x", label="G2 OMP 32x2")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[6], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_sph_exa_t_gnu_speedup.pdf")
    plt.clf()

def compare_sph_exa_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    plt.title("SPH-EXA Tiny - GNU Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1800])
    plt.plot([1,2,4,8], tx2_omp_times.T[6], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_times.T[6], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[6], c="green", linestyle="solid", marker="x", label="G2 OMP 32x2")
    plt.plot([1,2,4,8], g2_mpi_times.T[6], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_sph_exa_t_gnu_time.pdf")
    plt.clf()

def compare_hpgmgfv_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    plt.title("HPGMG-FV Tiny - GNU Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[7], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[7], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[7], c="green", linestyle="solid", marker="x", label="G2 OMP 16x4")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[7], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_hpgmgfv_t_gnu_speedup.pdf")
    plt.clf()

def compare_hpgmgfv_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    plt.title("HPGMG-FV Tiny - GNU Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1200])
    plt.plot([1,2,4,8], tx2_omp_times.T[7], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_times.T[7], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[7], c="green", linestyle="solid", marker="x", label="G2 OMP 16x4")
    plt.plot([1,2,4,8], g2_mpi_times.T[7], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_hpgmgfv_t_gnu_time.pdf")
    plt.clf()

def compare_weather_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    plt.title("miniWeather Tiny - GNU Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 12])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[8], c="red", linestyle="solid", marker="x", label="TX2 OMP 8x8")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[8], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[8], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[8], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_weather_t_gnu_speedup.pdf")
    plt.clf()

def compare_weather_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times):
    plt.title("miniWeather Tiny - GNU Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1800])
    plt.plot([1,2,4,8], tx2_omp_times.T[8], c="red", linestyle="solid", marker="x", label="TX2 OMP 8x8")
    plt.plot([1,2,4,8], tx2_mpi_times.T[8], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[8], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_mpi_times.T[8], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_weather_t_gnu_time.pdf")
    plt.clf()

def compare_tiny_gnu_total_time(tx2_times, g2_times):
    plt.title("Tiny suite - GNU Compiler - Total Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 12000])
    plt.plot([1,2,4,8], tx2_times[3], c="red", linestyle="solid", marker="x", label="TX2 OMP 8x8")
    plt.plot([1,2,4,8], tx2_times[6], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_times[3], c="orange", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_times[6], c="orange", linestyle="dashed", marker=".", label="G2 MPI")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_tiny_gnu_total_time.pdf")
    plt.clf()

def compare_tiny_gnu_spec_score(tx2_scores, g2_scores):
    plt.title("Tiny suite - GNU Compiler - SPEC Score")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 18])
    plt.plot([1,2,4,8], tx2_scores[3], c="red", linestyle="solid", marker="x", label="TX2 OMP 8x8")
    plt.plot([1,2,4,8], tx2_scores[6], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_scores[3], c="orange", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_scores[6], c="orange", linestyle="dotted", marker=".", label="G2 MPI")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("SPEC Score")
    plt.savefig("compare_tiny_gnu_spec_score.pdf")   
    plt.clf()

def main():
    tx2_tiny_gnu_mpi_times = np.array([[1253,1872,1043,942,1022,1370,1603,1051,1475],
                                   [633,990,558,482,687,702,792,532,764],
                                   [327,549,365,256,412,389,396,284,381],
                                   [166,332,197,139,308,214,209,167,212]])
    tx2_tiny_gnu_omp_1ppn_times = np.array([[1601,1847,1013,1067,786,2592,3705,1098,2574],
                                        [819,945,596,586,431,1362,1098,843,1300],
                                        [422,493,381,332,240,717,541,586,652],
                                        [213,266,272,192,159,395,271,431,348]])
    tx2_tiny_gnu_omp_2ppn_times = np.array([[1276,1865,1254,1582,768,2076,1840,1590,2259],
                                        [635,907,563,571,398,884,928,676,1037],
                                        [321,475,333,313,257,508,453,414,455],
                                        [170,260,237,173,159,270,239,264,236]])
    tx2_tiny_gnu_omp_4ppn_times = np.array([[1269,1785,1056,1073,728,1682,1602,1209,2124],
                                        [629,917,557,563,437,874,843,644,925],
                                        [325,480,341,280,261,478,442,332,402],
                                        [169,266,212,157,174,256,235,207,208]])
    tx2_tiny_gnu_omp_8ppn_times = np.array([[1263,1802,1030,1005,823,1544,1580,1009,1726],
                                        [645,914,534,499,455,849,812,545,837],
                                        [328,484,345,274,295,422,427,309,387],
                                        [165,268,204,153,183,245,232,172,196]])
    tx2_tiny_gnu_omp_16ppn_times = np.array([[1275,1822,1027,982,846,1624,1546,1040,1659],
                                         [641,920,527,484,537,749,795,485,770],
                                         [318,500,350,273,321,440,433,304,380],
                                         [164,281,202,152,238,234,224,172,196]])
    tx2_tiny_gnu_omp_32ppn_times = np.array([[1269,1762,902,880,992,1159,1480,858,1423],
                                         [626,964,587,516,579,791,816,555,716],
                                         [320,518,371,268,416,411,409,293,370],
                                         [167,302,200,147,261,232,211,168,207]])
    tx2_tiny_gnu_total_times = np.array([tx2_tiny_gnu_omp_1ppn_times.sum(axis=1),
                                     tx2_tiny_gnu_omp_2ppn_times.sum(axis=1),
                                     tx2_tiny_gnu_omp_4ppn_times.sum(axis=1),
                                     tx2_tiny_gnu_omp_8ppn_times.sum(axis=1),
                                     tx2_tiny_gnu_omp_16ppn_times.sum(axis=1),
                                     tx2_tiny_gnu_omp_32ppn_times.sum(axis=1),
                                     tx2_tiny_gnu_mpi_times.sum(axis=1)])
    tx2_tiny_gnu_spec_scores = np.array([[1.27,2.43,4.39,7.50],
                                      [1.31,2.88,5.28,9.24],
                                      [1.52,2.94,5.57,9.81],
                                      [1.60,3.08,5.66,10.2],
                                      [1.59,3.16,5.59,9.96],
                                      [1.75,3.02,5.49,9.84],
                                      [1.61,3.03,5.54,9.71]])
    g2_tiny_gnu7_mpi_times = np.array([[812,1160,1145,982,655,1583,785,777,1070],
                                       [410,639,583,494,440,802,398,398,542],
                                       [221,388,293,257,260,408,205,204,250],
                                       [117,258,145,131,190,211,111,108,119]])
    g2_tiny_gnu7_omp_1ppn_times = np.array([[884,1009,1163,1072,414,1646,1460,738,1153],
                                            [449,515,593,527,241,841,553,432,577],
                                            [230,267,286,260,146,417,278,213,363],
                                            [228,271,308,260,146,417,278,213,287]])
    g2_tiny_gnu7_omp_2ppn_times = np.array([[868,1028,1167,1015,426,1629,909,752,1123],
                                            [434,526,572,507,236,819,474,387,562],
                                            [219,274,286,257,155,407,242,203,272],
                                            [118,147,141,129,103,205,128,113,111]])
    g2_tiny_gnu7_omp_4ppn_times = np.array([[868,1047,1136,1008,419,1635,851,751,1109],
                                            [431,543,568,507,256,809,440,382,561],
                                            [223,280,281,251,158,407,228,196,258],
                                            [117,154,139,127,106,200,119,107,106]])
    g2_tiny_gnu7_omp_8ppn_times = np.array([[862,1054,1135,1010,468,1613,815,746,1116],
                                            [430,545,565,497,268,810,418,379,551],
                                            [218,286,281,250,179,399,212,193,257],
                                            [118,156,139,129,113,200,116,104,103]])
    g2_tiny_gnu7_omp_16ppn_times = np.array([[863,1073,1128,991,493,1618,795,749,1122],
                                             [430,556,566,497,316,791,398,381,551],
                                             [217,300,283,252,187,400,211,194,259],
                                             [117,170,142,128,136,202,111,103,106]])
    g2_tiny_gnu7_omp_32ppn_times = np.array([[865,1107,1134,993,601,1565,765,757,1130],
                                             [431,585,574,500,343,802,399,386,549],
                                             [217,326,288,253,238,403,207,197,271],
                                             [120,198,143,135,147,203,109,105,107]])
    g2_tiny_gnu7_total_times = np.array([g2_tiny_gnu7_omp_1ppn_times.sum(axis=1),
                                     g2_tiny_gnu7_omp_2ppn_times.sum(axis=1),
                                     g2_tiny_gnu7_omp_4ppn_times.sum(axis=1),
                                     g2_tiny_gnu7_omp_8ppn_times.sum(axis=1),
                                     g2_tiny_gnu7_omp_16ppn_times.sum(axis=1),
                                     g2_tiny_gnu7_omp_32ppn_times.sum(axis=1),
                                     g2_tiny_gnu7_mpi_times.sum(axis=1)])
    g2_tiny_gnu7_spec_scores = np.array([[2.04,4.03,7.70,7.83],
                                         [2.16,4.23,8.12,15.5],
                                         [2.18,4.23,8.24,15.8],
                                         [2.17,4.25,8.23,15.8],
                                         [2.16,4.20,8.13,15.4],
                                         [2.12,4.12,7.78,14.8],
                                         [2.10,3.98,7.55,13.7]])
    compare_tiny_gnu_spec_score(tx2_tiny_gnu_spec_scores, g2_tiny_gnu7_spec_scores)
    compare_tiny_gnu_total_time(tx2_tiny_gnu_total_times, g2_tiny_gnu7_total_times)
    compare_lbm_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_16ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_16ppn_times)
    compare_lbm_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_16ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_16ppn_times)
    compare_soma_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_2ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_2ppn_times)
    compare_soma_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_2ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_2ppn_times)
    compare_tealeaf_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_8ppn_times)
    compare_tealeaf_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_8ppn_times)
    compare_clvleaf_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_4ppn_times)
    compare_clvleaf_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_4ppn_times)
    compare_miniswp_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_2ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_2ppn_times)
    compare_miniswp_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_2ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_2ppn_times)
    compare_pot3d_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_4ppn_times)
    compare_pot3d_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_4ppn_times)
    compare_sph_exa_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_32ppn_times)
    compare_sph_exa_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_32ppn_times)
    compare_hpgmgfv_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_16ppn_times)
    compare_hpgmgfv_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_16ppn_times)
    compare_weather_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_8ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_8ppn_times)
    compare_weather_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_8ppn_times, g2_tiny_gnu7_mpi_times, g2_tiny_gnu7_omp_8ppn_times)


if __name__ == "__main__":
    main()
