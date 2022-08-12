import numpy as np
from matplotlib import pyplot as plt

def compare_lbm_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times, g2_omp_times, g3_mpi_times, g3_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    g3_mpi_speedup = g3_mpi_times[0]/g3_mpi_times
    g3_omp_speedup = g3_omp_times[0]/g3_omp_times
    plt.title("LBM Tiny - GNU 9 Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[0], c="red", linestyle="solid", marker="x", label="TX2 OMP 16x4")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[0], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[0], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[0], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.plot([1,2,4,8], g3_omp_speedup.T[0], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_speedup.T[0], c="blue", linestyle="dotted", marker=".", label="G3 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_lbm_t_gnu_speedup.pdf")
    plt.clf()

def compare_lbm_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    plt.title("LBM Tiny - GNU 9 Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1400])
    plt.plot([1,2,4,8], tx2_omp_times.T[0], c="red", linestyle="solid", marker="x", label="TX2 OMP 16x4")
    plt.plot([1,2,4,8], tx2_mpi_times.T[0], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[0], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_mpi_times.T[0], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_times.T[0], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_times.T[0], c="blue", linestyle="dotted", marker=".", label="G3 MPI") 
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_lbm_t_gnu_time.pdf")
    plt.clf()

def compare_soma_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    g3_mpi_speedup = g3_mpi_times[0]/g3_mpi_times
    g3_omp_speedup = g3_omp_times[0]/g3_omp_times
    plt.title("SOMA Tiny - GNU 9 Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[1], c="red", linestyle="solid", marker="x", label="TX2 OMP 2x32")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[1], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[1], c="green", linestyle="solid", marker="x", label="G2 OMP 2x32")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[1], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_speedup.T[1], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_speedup.T[1], c="blue", linestyle="dotted", marker=".", label="G3 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_soma_t_gnu_speedup.pdf")
    plt.clf()

def compare_soma_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    plt.title("SOMA Tiny - GNU 9 Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 2000])
    plt.plot([1,2,4,8], tx2_omp_times.T[1], c="red", linestyle="solid", marker="x", label="TX2 OMP 2x32")
    plt.plot([1,2,4,8], tx2_mpi_times.T[1], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[1], c="green", linestyle="solid", marker="x", label="G2 OMP 2x32")
    plt.plot([1,2,4,8], g2_mpi_times.T[1], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_times.T[1], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_times.T[1], c="blue", linestyle="dotted", marker=".", label="G3 MPI") 
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_soma_t_gnu_time.pdf")
    plt.clf()

def compare_tealeaf_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    g3_mpi_speedup = g3_mpi_times[0]/g3_mpi_times
    g3_omp_speedup = g3_omp_times[0]/g3_omp_times
    plt.title("TeaLeaf Tiny - GNU 9 Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 9])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[2], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[2], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[2], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[2], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_speedup.T[2], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_speedup.T[2], c="blue", linestyle="dotted", marker=".", label="G3 MPI") 
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_tealeaf_t_gnu_speedup.pdf")
    plt.clf()

def compare_tealeaf_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    plt.title("TeaLeaf Tiny - GNU 9 Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1200])
    plt.plot([1,2,4,8], tx2_omp_times.T[2], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_times.T[2], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[2], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_mpi_times.T[2], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_times.T[2], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_times.T[2], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_tealeaf_t_gnu_time.pdf")
    plt.clf()

def compare_clvleaf_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    g3_mpi_speedup = g3_mpi_times[0]/g3_mpi_times
    g3_omp_speedup = g3_omp_times[0]/g3_omp_times
    plt.title("CloverLeaf Tiny - GNU 9 Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[3], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[3], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[3], c="green", linestyle="solid", marker="x", label="G2 OMP 4x16")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[3], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_speedup.T[3], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_speedup.T[3], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_clvleaf_t_gnu_speedup.pdf")
    plt.clf()

def compare_clvleaf_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    plt.title("CloverLeaf Tiny - GNU 9 Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1200])
    plt.plot([1,2,4,8], tx2_omp_times.T[3], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_times.T[3], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[3], c="green", linestyle="solid", marker="x", label="G2 OMP 4x16")
    plt.plot([1,2,4,8], g2_mpi_times.T[3], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_times.T[3], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_times.T[3], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_clvleaf_t_gnu_time.pdf")
    plt.clf()

def compare_miniswp_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    g3_mpi_speedup = g3_mpi_times[0]/g3_mpi_times
    g3_omp_speedup = g3_omp_times[0]/g3_omp_times
    plt.title("Minisweep Tiny - GNU 9 Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 6])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[4], c="red", linestyle="solid", marker="x", label="TX2 OMP 2x32")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[4], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[4], c="green", linestyle="solid", marker="x", label="G2 OMP 4x16")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[4], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_speedup.T[4], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_speedup.T[4], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_miniswp_t_gnu_speedup.pdf")
    plt.clf()

def compare_miniswp_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    plt.title("Minisweep Tiny - GNU 9 Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1200])
    plt.plot([1,2,4,8], tx2_omp_times.T[4], c="red", linestyle="solid", marker="x", label="TX2 OMP 2x32")
    plt.plot([1,2,4,8], tx2_mpi_times.T[4], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[4], c="green", linestyle="solid", marker="x", label="G2 OMP 4x16")
    plt.plot([1,2,4,8], g2_mpi_times.T[4], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_times.T[4], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_times.T[4], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_miniswp_t_gnu_time.pdf")
    plt.clf()

def compare_pot3d_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    g3_mpi_speedup = g3_mpi_times[0]/g3_mpi_times
    g3_omp_speedup = g3_omp_times[0]/g3_omp_times
    plt.title("POT3D Tiny - GNU 9 Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[5], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[5], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[5], c="green", linestyle="solid", marker="x", label="G2 OMP 2x32")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[5], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_speedup.T[5], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_speedup.T[5], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_pot3d_t_gnu_speedup.pdf")
    plt.clf()

def compare_pot3d_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    plt.title("Minisweep Tiny - GNU 9 Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1800])
    plt.plot([1,2,4,8], tx2_omp_times.T[5], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_times.T[5], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[5], c="green", linestyle="solid", marker="x", label="G2 OMP 2x32")
    plt.plot([1,2,4,8], g2_mpi_times.T[5], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_times.T[5], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_times.T[5], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_pot3d_t_gnu_time.pdf")
    plt.clf()

def compare_sph_exa_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    g3_mpi_speedup = g3_mpi_times[0]/g3_mpi_times
    g3_omp_speedup = g3_omp_times[0]/g3_omp_times
    plt.title("SPH-EXA Tiny - GNU 9 Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[6], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[6], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[6], c="green", linestyle="solid", marker="x", label="G2 OMP 2x32")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[6], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_speedup.T[6], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_speedup.T[6], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_sph_exa_t_gnu_speedup.pdf")
    plt.clf()

def compare_sph_exa_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    plt.title("SPH-EXA Tiny - GNU 9 Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1800])
    plt.plot([1,2,4,8], tx2_omp_times.T[6], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_times.T[6], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[6], c="green", linestyle="solid", marker="x", label="G2 OMP 2x32")
    plt.plot([1,2,4,8], g2_mpi_times.T[6], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_times.T[6], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_times.T[6], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_sph_exa_t_gnu_time.pdf")
    plt.clf()

def compare_hpgmgfv_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    g3_mpi_speedup = g3_mpi_times[0]/g3_mpi_times
    g3_omp_speedup = g3_omp_times[0]/g3_omp_times
    plt.title("HPGMG-FV Tiny - GNU 9 Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[7], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[7], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[7], c="green", linestyle="solid", marker="x", label="G2 OMP 16x4")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[7], c="green", linestyle="dotted", marker=".", label="G2 MPI") 
    plt.plot([1,2,4,8], g3_omp_speedup.T[7], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_speedup.T[7], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_hpgmgfv_t_gnu_speedup.pdf")
    plt.clf()

def compare_hpgmgfv_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    plt.title("HPGMG-FV Tiny - GNU 9 Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1200])
    plt.plot([1,2,4,8], tx2_omp_times.T[7], c="red", linestyle="solid", marker="x", label="TX2 OMP 32x2")
    plt.plot([1,2,4,8], tx2_mpi_times.T[7], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[7], c="green", linestyle="solid", marker="x", label="G2 OMP 16x4")
    plt.plot([1,2,4,8], g2_mpi_times.T[7], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_times.T[7], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_times.T[7], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_hpgmgfv_t_gnu_time.pdf")
    plt.clf()

def compare_weather_t_gnu_speedup(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    tx2_mpi_speedup = tx2_mpi_times[0]/tx2_mpi_times
    tx2_omp_speedup = tx2_omp_times[0]/tx2_omp_times
    g2_mpi_speedup = g2_mpi_times[0]/g2_mpi_times
    g2_omp_speedup = g2_omp_times[0]/g2_omp_times
    g3_mpi_speedup = g3_mpi_times[0]/g3_mpi_times
    g3_omp_speedup = g3_omp_times[0]/g3_omp_times
    plt.title("miniWeather Tiny - GNU 9 Compiler - Speedup")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 12])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], tx2_omp_speedup.T[8], c="red", linestyle="solid", marker="x", label="TX2 OMP 8x8")
    plt.plot([1,2,4,8], tx2_mpi_speedup.T[8], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_speedup.T[8], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_mpi_speedup.T[8], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_speedup.T[8], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_speedup.T[8], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("compare_weather_t_gnu_speedup.pdf")
    plt.clf()

def compare_weather_t_gnu_time(tx2_mpi_times, tx2_omp_times, g2_mpi_times,g2_omp_times, g3_mpi_times, g3_omp_times):
    plt.title("miniWeather Tiny - GNU 9 Compiler - Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 1800])
    plt.plot([1,2,4,8], tx2_omp_times.T[8], c="red", linestyle="solid", marker="x", label="TX2 OMP 8x8")
    plt.plot([1,2,4,8], tx2_mpi_times.T[8], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_omp_times.T[8], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_mpi_times.T[8], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_omp_times.T[8], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_mpi_times.T[8], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_weather_t_gnu_time.pdf")
    plt.clf()

def compare_tiny_gnu_total_time(tx2_times, g2_times, g3_times):
    plt.title("Tiny suite - GNU 9 Compiler - Total Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 12000])
    plt.plot([1,2,4,8], tx2_times[3], c="red", linestyle="solid", marker="x", label="TX2 OMP 8x8")
    plt.plot([1,2,4,8], tx2_times[6], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_times[3], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_times[6], c="green", linestyle="dashed", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_times[3], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_times[6], c="blue", linestyle="dashed", marker=".", label="G3 MPI")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Time(s)")
    plt.savefig("compare_tiny_gnu_total_time.pdf")
    plt.clf()

def compare_tiny_gnu_spec_score(tx2_scores, g2_scores, g3_scores):
    plt.title("Tiny suite - GNU 9 Compiler - SPEC Score")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 18])
    plt.plot([1,2,4,8], tx2_scores[3], c="red", linestyle="solid", marker="x", label="TX2 OMP 8x8")
    plt.plot([1,2,4,8], tx2_scores[6], c="red", linestyle="dotted", marker=".", label="TX2 MPI")
    plt.plot([1,2,4,8], g2_scores[3], c="green", linestyle="solid", marker="x", label="G2 OMP 8x8")
    plt.plot([1,2,4,8], g2_scores[6], c="green", linestyle="dotted", marker=".", label="G2 MPI")
    plt.plot([1,2,4,8], g3_scores[3], c="blue", linestyle="solid", marker="x", label="G3 OMP 8x8")
    plt.plot([1,2,4,8], g3_scores[6], c="blue", linestyle="dotted", marker=".", label="G3 MPI")
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
    g2_tiny_gnu_mpi_times = np.array([[764,1147,1170,979,604,1581,776,772,1068],
                                       [387,636,591,493,408,803,396,401,539],
                                       [203,383,297,256,242,403,206,204,243],
                                       [108,257,147,131,179,208,113,108,113]])
    g2_tiny_gnu_omp_1ppn_times = np.array([[0,0,0,0,0,0,0,0,0],
                                            [0,0,0,0,0,0,0,0,0],
                                            [0,0,0,0,0,0,0,0,0],
                                            [0,0,0,0,0,0,0,0,0]])
    g2_tiny_gnu_omp_2ppn_times = np.array([[770,1048,1164,1014,427,1625,922,750,1085],
                                            [387,534,578,508,237,817,478,387,545],
                                            [198,278,287,257,155,408,246,202,263],
                                            [99.2,156,140,128,104,205,130,112,108]])
    g2_tiny_gnu_omp_4ppn_times = np.array([[764,1054,1133,1007,421,1627,856,748,1095],
                                            [385,549,571,507,257,811,445,381,543],
                                            [194,293,282,250,159,405,230,196,250],
                                            [98.6,160,140,127,107,201,121,107,101]])
    g2_tiny_gnu_omp_8ppn_times = np.array([[762,1063,1135,1007,469,1608,827,744,1110],
                                            [383,544,565,495,269,809,424,378,538],
                                            [194,294,281,249,180,398,214,193,250],
                                            [97.9,157,138,127,113,198,115,103,101]])
    g2_tiny_gnu_omp_16ppn_times = np.array([[762,1074,1133,985,495,1607,802,747,1098],
                                             [385,563,566,497,317,795,403,379,538],
                                             [194,306,295,252,188,395,208,193,251],
                                             [98.0,170,142,127,135,200,111,103,102]])
    g2_tiny_gnu_omp_32ppn_times = np.array([[0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0],
                                             [0,0,0,0,0,0,0,0,0]])
    g2_tiny_gnu_total_times = np.array([g2_tiny_gnu_omp_1ppn_times.sum(axis=1),
                                     g2_tiny_gnu_omp_2ppn_times.sum(axis=1),
                                     g2_tiny_gnu_omp_4ppn_times.sum(axis=1),
                                     g2_tiny_gnu_omp_8ppn_times.sum(axis=1),
                                     g2_tiny_gnu_omp_16ppn_times.sum(axis=1),
                                     g2_tiny_gnu_omp_32ppn_times.sum(axis=1),
                                     g2_tiny_gnu_mpi_times.sum(axis=1)])
    g2_tiny_gnu_spec_scores = np.array([[0,0,0,0],
                                         [0,0,0,0],
                                         [0,0,0,0],
                                         [2.20,4.32,8.33,16.2],
                                         [0,0,0,0],
                                         [0,0,0,0],
                                         [2.14,4.04,7.71,14.0]])
    g3_tiny_gnu_mpi_times = np.array([[444,827,700,565,356,914,496,486,725],
                                     [221,452,1733,306,275,557,304,269,369],
                                     [111,271,305,156,175,265,154,157,170],
                                     [100,100,100,100,100,100,100,100,100]])#placeholder
    g3_tiny_gnu_omp_1ppn_times = np.array([[434,720,706,623,225,947,693,470,735],
                                            [228,376,371,291,193,487,380,321,376],
                                            [100,100,100,100,100,100,100,100,100],
                                            [100,100,100,100,100,100,100,100,100]])
    g3_tiny_gnu_omp_2ppn_times = np.array([[434,741,705,567,233,941,605,475,707],
                                            [224,381,365,287,181,481,303,276,350],
                                            [100,100,100,100,100,100,100,100,100],
                                            [100,100,100,100,100,100,100,100,100]])
    g3_tiny_gnu_omp_4ppn_times = np.array([[428,745,695,566,230,943,517,473,698],
                                            [218,387,359,285,173,468,270,283,326],
                                            [100,100,100,100,100,100,100,100,100],
                                            [100,100,100,100,100,100,100,100,100]])
    g3_tiny_gnu_omp_8ppn_times = np.array([[427,752,695,565,258,928,494,472,687],
                                            [216,388,363,283,178,474,268,290,318],
                                            [100,100,100,100,100,100,100,100,100],
                                            [100,100,100,100,100,100,100,100,100]])
    g3_tiny_gnu_omp_16ppn_times = np.array([[432,766,700,559,272,928,491,474,696],
                                            [219,397,366,284,196,486,291,281,326],
                                            [100,100,100,100,100,100,100,100,100],
                                            [100,100,100,100,100,100,100,100,100]])
    g3_tiny_gnu_omp_32ppn_times = np.array([[431,791,699,560,332,910,496,477,705],
                                            [219,419,385,290,212,464,269,276,339],
                                            [100,100,100,100,100,100,100,100,100],
                                            [100,100,100,100,100,100,100,100,100]])
    g3_tiny_gnu_total_times = np.array([g3_tiny_gnu_omp_1ppn_times.sum(axis=1),
                                     g3_tiny_gnu_omp_2ppn_times.sum(axis=1),
                                     g3_tiny_gnu_omp_4ppn_times.sum(axis=1),
                                     g3_tiny_gnu_omp_8ppn_times.sum(axis=1),
                                     g3_tiny_gnu_omp_16ppn_times.sum(axis=1),
                                     g3_tiny_gnu_omp_32ppn_times.sum(axis=1),
                                     g3_tiny_gnu_mpi_times.sum(axis=1)])
    g3_tiny_gnu_spec_scores = np.array([[3.51,6.24,12,24],#end 12/24s are placeholders
                                        [3.59,6.64,12,24],
                                        [3.67,6.84,12,24],
                                        [3.66,6.81,12,24],
                                        [3.62,6.63,12,24],
                                        [3.53,6.56,12,24],
                                        [3.45,5.10,10.8,24]])
    compare_tiny_gnu_spec_score(tx2_tiny_gnu_spec_scores, g2_tiny_gnu_spec_scores, g3_tiny_gnu_spec_scores)
    compare_tiny_gnu_total_time(tx2_tiny_gnu_total_times, g2_tiny_gnu_total_times, g3_tiny_gnu_total_times)
    compare_lbm_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_16ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_8ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_lbm_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_16ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_8ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_soma_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_2ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_2ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_soma_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_2ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_2ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_tealeaf_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_8ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_tealeaf_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_8ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_clvleaf_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_4ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_clvleaf_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_4ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_miniswp_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_2ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_4ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_miniswp_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_2ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_4ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_pot3d_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_2ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_pot3d_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_2ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_sph_exa_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_2ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_sph_exa_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_2ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_hpgmgfv_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_16ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_hpgmgfv_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_32ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_16ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_weather_t_gnu_speedup(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_8ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_8ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)
    compare_weather_t_gnu_time(tx2_tiny_gnu_mpi_times, tx2_tiny_gnu_omp_8ppn_times, g2_tiny_gnu_mpi_times, g2_tiny_gnu_omp_8ppn_times, g3_tiny_gnu_mpi_times, g3_tiny_gnu_omp_8ppn_times)


if __name__ == "__main__":
    main()
