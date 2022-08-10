import numpy as np
from matplotlib import pyplot as plt

def g2_tiny_gnu7_mpi_time(times):
    times = np.log10(times)
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - MPI Runtime (64 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([2, 3.4])
    plt.plot([1,2,4,8], times.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], times.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], times.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], times.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], times.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], times.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], times.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], times.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], times.T[8], marker="x", label="weather")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("$log_{10}$ Time(s)")
    plt.savefig("g2_tiny_gnu7_mpi_time.pdf")
    plt.clf()

def g2_tiny_gnu7_mpi_speedup(times):
    speedup = times[0]/times
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - MPI Speedup (64 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 10])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], speedup.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], speedup.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], speedup.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], speedup.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], speedup.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], speedup.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], speedup.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], speedup.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], speedup.T[8], marker="x", label="weather")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("g2_tiny_gnu7_mpi_speedup.pdf")
    plt.clf()

def g2_tiny_gnu7_mpi_efficiency(times):
    speedup = times[0]/times
    efficiency = speedup/np.array([[1],[2],[4],[8]]) * 100
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - MPI Efficiency (64 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([40, 120])
    plt.axhline(y=100, c="grey", linestyle="--")
    plt.plot([1,2,4,8], efficiency.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], efficiency.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], efficiency.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], efficiency.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], efficiency.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], efficiency.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], efficiency.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], efficiency.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], efficiency.T[8], marker="x", label="weather")
    plt.legend(loc="lower left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Parallel efficiency(%)")
    plt.savefig("g2_tiny_gnu7_mpi_efficiency.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_1ppn_time(times):
    times = np.log10(times)
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Runtime (1 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([2, 3.4])
    plt.plot([1,2,4,8], times.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], times.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], times.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], times.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], times.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], times.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], times.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], times.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], times.T[8], marker="x", label="weather")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("$log_{10}$ Time(s)")
    plt.savefig("g2_tiny_gnu7_omp_1ppn_time.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_1ppn_speedup(times):
    speedup = times[0]/times
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Speedup (1 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 12])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], speedup.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], speedup.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], speedup.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], speedup.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], speedup.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], speedup.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], speedup.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], speedup.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], speedup.T[8], marker="x", label="weather")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("g2_tiny_gnu7_omp_1ppn_speedup.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_1ppn_efficiency(times):
    speedup = times[0]/times
    efficiency = speedup/np.array([[1],[2],[4],[8]]) * 100
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Efficiency (1 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([30, 140])
    plt.axhline(y=100, c="grey", linestyle="--")
    plt.plot([1,2,4,8], efficiency.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], efficiency.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], efficiency.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], efficiency.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], efficiency.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], efficiency.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], efficiency.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], efficiency.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], efficiency.T[8], marker="x", label="weather")
    plt.legend(loc="lower left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Parallel efficiency(%)")
    plt.savefig("g2_tiny_gnu7_omp_1ppn_efficiency.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_2ppn_time(times):
    times = np.log10(times)
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Runtime (2 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([2, 3.4])
    plt.plot([1,2,4,8], times.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], times.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], times.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], times.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], times.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], times.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], times.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], times.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], times.T[8], marker="x", label="weather")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("$log_{10}$ Time(s)")
    plt.savefig("g2_tiny_gnu7_omp_2ppn_time.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_2ppn_speedup(times):
    speedup = times[0]/times
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Speedup (2 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 12])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], speedup.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], speedup.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], speedup.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], speedup.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], speedup.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], speedup.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], speedup.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], speedup.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], speedup.T[8], marker="x", label="weather")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("g2_tiny_gnu7_omp_2ppn_speedup.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_2ppn_efficiency(times):
    speedup = times[0]/times
    efficiency = speedup/np.array([[1],[2],[4],[8]]) * 100
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Efficiency (2 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([40, 140])
    plt.axhline(y=100, c="grey", linestyle="--")
    plt.plot([1,2,4,8], efficiency.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], efficiency.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], efficiency.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], efficiency.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], efficiency.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], efficiency.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], efficiency.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], efficiency.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], efficiency.T[8], marker="x", label="weather")
    plt.legend(loc="lower left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Parallel efficiency(%)")
    plt.savefig("g2_tiny_gnu7_omp_2ppn_efficiency.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_4ppn_time(times):
    times = np.log10(times)
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Runtime (4 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([2, 3.4])
    plt.plot([1,2,4,8], times.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], times.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], times.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], times.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], times.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], times.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], times.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], times.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], times.T[8], marker="x", label="weather")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("$log_{10}$ Time(s)")
    plt.savefig("g2_tiny_gnu7_omp_4ppn_time.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_4ppn_speedup(times):
    speedup = times[0]/times
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Speedup (4 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 12])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], speedup.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], speedup.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], speedup.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], speedup.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], speedup.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], speedup.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], speedup.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], speedup.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], speedup.T[8], marker="x", label="weather")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("g2_tiny_gnu7_omp_4ppn_speedup.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_4ppn_efficiency(times):
    speedup = times[0]/times
    efficiency = speedup/np.array([[1],[2],[4],[8]]) * 100
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Efficiency (4 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([40, 140])
    plt.axhline(y=100, c="grey", linestyle="--")
    plt.plot([1,2,4,8], efficiency.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], efficiency.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], efficiency.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], efficiency.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], efficiency.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], efficiency.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], efficiency.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], efficiency.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], efficiency.T[8], marker="x", label="weather")
    plt.legend(loc="lower left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Parallel efficiency(%)")
    plt.savefig("g2_tiny_gnu7_omp_4ppn_efficiency.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_8ppn_time(times):
    times = np.log10(times)
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Runtime (8 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([2, 3.4])
    plt.plot([1,2,4,8], times.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], times.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], times.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], times.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], times.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], times.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], times.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], times.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], times.T[8], marker="x", label="weather")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("$log_{10}$ Time(s)")
    plt.savefig("g2_tiny_gnu7_omp_8ppn_time.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_8ppn_speedup(times):
    speedup = times[0]/times
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Speedup (8 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 12])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], speedup.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], speedup.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], speedup.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], speedup.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], speedup.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], speedup.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], speedup.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], speedup.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], speedup.T[8], marker="x", label="weather")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("g2_tiny_gnu7_omp_8ppn_speedup.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_8ppn_efficiency(times):
    speedup = times[0]/times
    efficiency = speedup/np.array([[1],[2],[4],[8]]) * 100
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Efficiency (8 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([40, 140])
    plt.axhline(y=100, c="grey", linestyle="--")
    plt.plot([1,2,4,8], efficiency.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], efficiency.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], efficiency.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], efficiency.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], efficiency.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], efficiency.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], efficiency.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], efficiency.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], efficiency.T[8], marker="x", label="weather")
    plt.legend(loc="lower left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Parallel efficiency(%)")
    plt.savefig("g2_tiny_gnu7_omp_8ppn_efficiency.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_16ppn_time(times):
    times = np.log10(times)
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Runtime (16 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([2, 3.4])
    plt.plot([1,2,4,8], times.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], times.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], times.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], times.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], times.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], times.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], times.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], times.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], times.T[8], marker="x", label="weather")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("$log_{10}$ Time(s)")
    plt.savefig("g2_tiny_gnu7_omp_16ppn_time.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_16ppn_speedup(times):
    speedup = times[0]/times
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Speedup (16 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 12])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], speedup.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], speedup.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], speedup.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], speedup.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], speedup.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], speedup.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], speedup.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], speedup.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], speedup.T[8], marker="x", label="weather")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("g2_tiny_gnu7_omp_16ppn_speedup.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_16ppn_efficiency(times):
    speedup = times[0]/times
    efficiency = speedup/np.array([[1],[2],[4],[8]]) * 100
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Efficiency (16 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([40, 140])
    plt.axhline(y=100, c="grey", linestyle="--")
    plt.plot([1,2,4,8], efficiency.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], efficiency.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], efficiency.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], efficiency.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], efficiency.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], efficiency.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], efficiency.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], efficiency.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], efficiency.T[8], marker="x", label="weather")
    plt.legend(loc="lower left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Parallel efficiency(%)")
    plt.savefig("g2_tiny_gnu7_omp_16ppn_efficiency.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_32ppn_time(times):
    times = np.log10(times)
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Runtime (32 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([2, 3.4])
    plt.plot([1,2,4,8], times.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], times.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], times.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], times.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], times.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], times.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], times.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], times.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], times.T[8], marker="x", label="weather")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("$log_{10}$ Time(s)")
    plt.savefig("g2_tiny_gnu7_omp_32ppn_time.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_32ppn_speedup(times):
    speedup = times[0]/times
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Speedup (32 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 12])
    plt.plot(np.arange(0, 9), c="grey", linestyle="--")
    plt.plot([1,2,4,8], speedup.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], speedup.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], speedup.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], speedup.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], speedup.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], speedup.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], speedup.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], speedup.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], speedup.T[8], marker="x", label="weather")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Speedup")
    plt.savefig("g2_tiny_gnu7_omp_32ppn_speedup.pdf")
    plt.clf()

def g2_tiny_gnu7_omp_32ppn_efficiency(times):
    speedup = times[0]/times
    efficiency = speedup/np.array([[1],[2],[4],[8]]) * 100
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - OMP Efficiency (32 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([40, 140])
    plt.axhline(y=100, c="grey", linestyle="--")
    plt.plot([1,2,4,8], efficiency.T[0], marker="x", label="lbm")
    plt.plot([1,2,4,8], efficiency.T[1], marker="x", label="soma")
    plt.plot([1,2,4,8], efficiency.T[2], marker="x", label="tealeaf")
    plt.plot([1,2,4,8], efficiency.T[3], marker="x", label="clvleaf")
    plt.plot([1,2,4,8], efficiency.T[4], marker="x", label="miniswp")
    plt.plot([1,2,4,8], efficiency.T[5], marker="x", label="pot3d")
    plt.plot([1,2,4,8], efficiency.T[6], marker="x", label="sph_exa")
    plt.plot([1,2,4,8], efficiency.T[7], marker="x", label="hpgmgfv")
    plt.plot([1,2,4,8], efficiency.T[8], marker="x", label="weather")
    plt.legend(loc="lower left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("Parallel efficiency(%)")
    plt.savefig("g2_tiny_gnu7_omp_32ppn_efficiency.pdf")
    plt.clf()

def g2_tiny_gnu7_total_time(times):
    times = np.log10(times)
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - Total Runtime")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([3.0, 4.2])
    plt.plot([1,2,4,8], times[0], marker="x", label="1x64")
    plt.plot([1,2,4,8], times[1], marker="x", label="2x32")
    plt.plot([1,2,4,8], times[2], marker="x", label="4x16")
    plt.plot([1,2,4,8], times[3], marker="x", label="8x8")
    plt.plot([1,2,4,8], times[4], marker="x", label="16x4")
    plt.plot([1,2,4,8], times[5], marker="x", label="32x2")
    plt.plot([1,2,4,8], times[6], marker="x", label="64x1")
    plt.legend(loc="upper right")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("$log_{10}$ Time(s)")
    plt.savefig("g2_tiny_gnu7_total_time.pdf")
    plt.clf()

def g2_tiny_gnu7_spec_score(scores):
    plt.title("Tiny suite - Graviton2 - GNU 7 Compiler - SPEC Score")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([0, 18])
    plt.plot([1,2,4,8], scores[0], marker="x", label="1x64")
    plt.plot([1,2,4,8], scores[1], marker="x", label="2x32")
    plt.plot([1,2,4,8], scores[2], marker="x", label="4x16")
    plt.plot([1,2,4,8], scores[3], marker="x", label="8x8")
    plt.plot([1,2,4,8], scores[4], marker="x", label="16x4")
    plt.plot([1,2,4,8], scores[5], marker="x", label="32x2")
    plt.plot([1,2,4,8], scores[6], marker="x", label="64x1")
    plt.legend(loc="upper left")
    plt.xlabel("Number of nodes (64 cores/node)")
    plt.ylabel("SPEC Score")
    plt.savefig("g2_tiny_gnu7_spec_score.pdf")   
    plt.clf() 

def main():

    g2_tiny_gnu7_mpi_times = np.array([[812,1160,1145,982,655,1583,785,777,1070],
                                       [410,639,583,494,440,802,398,398,542],
                                       [221,388,293,257,260,408,205,204,250],
                                       [117,258,145,131,190,211,111,108,119]])
    g2_tiny_gnu7_omp_1ppn_times = np.array([[884,1009,1163,1072,414,1646,1460,738,1153],
                                            [449,515,593,527,241,841,553,432,577],
                                            [230,267,286,260,146,417,278,213,363],
                                            [112,143,177,135,108,211,144,127,183]])
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
    g2_tiny_gnu7_spec_scores = np.array([[2.04,4.03,7.70,13.9],
                                         [2.16,4.23,8.12,15.5],
                                         [2.18,4.23,8.24,15.8],
                                         [2.17,4.25,8.23,15.8],
                                         [2.16,4.20,8.13,15.4],
                                         [2.12,4.12,7.78,14.8],
                                         [2.10,3.98,7.55,13.7]])
    g2_tiny_gnu7_mpi_time(g2_tiny_gnu7_mpi_times)
    g2_tiny_gnu7_mpi_speedup(g2_tiny_gnu7_mpi_times)
    g2_tiny_gnu7_mpi_efficiency(g2_tiny_gnu7_mpi_times)
    g2_tiny_gnu7_omp_1ppn_time(g2_tiny_gnu7_omp_1ppn_times)
    g2_tiny_gnu7_omp_1ppn_speedup(g2_tiny_gnu7_omp_1ppn_times)
    g2_tiny_gnu7_omp_1ppn_efficiency(g2_tiny_gnu7_omp_1ppn_times)
    g2_tiny_gnu7_omp_2ppn_time(g2_tiny_gnu7_omp_2ppn_times)
    g2_tiny_gnu7_omp_2ppn_speedup(g2_tiny_gnu7_omp_2ppn_times)
    g2_tiny_gnu7_omp_2ppn_efficiency(g2_tiny_gnu7_omp_2ppn_times)
    g2_tiny_gnu7_omp_4ppn_time(g2_tiny_gnu7_omp_4ppn_times)
    g2_tiny_gnu7_omp_4ppn_speedup(g2_tiny_gnu7_omp_4ppn_times)
    g2_tiny_gnu7_omp_4ppn_efficiency(g2_tiny_gnu7_omp_4ppn_times)
    g2_tiny_gnu7_omp_8ppn_time(g2_tiny_gnu7_omp_8ppn_times)
    g2_tiny_gnu7_omp_8ppn_speedup(g2_tiny_gnu7_omp_8ppn_times)
    g2_tiny_gnu7_omp_8ppn_efficiency(g2_tiny_gnu7_omp_8ppn_times)
    g2_tiny_gnu7_omp_16ppn_time(g2_tiny_gnu7_omp_16ppn_times)
    g2_tiny_gnu7_omp_16ppn_speedup(g2_tiny_gnu7_omp_16ppn_times)
    g2_tiny_gnu7_omp_16ppn_efficiency(g2_tiny_gnu7_omp_16ppn_times)
    g2_tiny_gnu7_omp_32ppn_time(g2_tiny_gnu7_omp_32ppn_times)
    g2_tiny_gnu7_omp_32ppn_speedup(g2_tiny_gnu7_omp_32ppn_times)
    g2_tiny_gnu7_omp_32ppn_efficiency(g2_tiny_gnu7_omp_32ppn_times)
    g2_tiny_gnu7_total_time(g2_tiny_gnu7_total_times)
    g2_tiny_gnu7_spec_score(g2_tiny_gnu7_spec_scores)


if __name__ == "__main__":
    main()
