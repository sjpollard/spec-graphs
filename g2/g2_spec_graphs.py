import numpy as np
from matplotlib import pyplot as plt

def g2_tiny_gnu7_mpi_time(times):
    times = np.log10(times)
    plt.title("Tiny suite - Graviton2 - GNU Compiler - MPI Runtime (64 ranks/node)")
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
    plt.title("Tiny suite - Graviton2 - GNU Compiler - MPI Speedup (64 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([1, 8])
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
    plt.title("Tiny suite - Graviton2 - GNU Compiler - MPI Efficiency (64 ranks/node)")
    plt.xticks([1,2,4,8])
    plt.xlim([1, 8])
    plt.ylim([30, 110])
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

def main():

    g2_tiny_gnu7_mpi_times = np.array([[812,1160,1145,982,655,1583,785,777,1070],
                                       [410,639,583,494,440,802,398,398,542],
                                       [221,388,293,257,260,408,205,204,250],
                                       [117,258,145,131,190,211,111,108,119]])
    g2_tiny_gnu7_mpi_time(g2_tiny_gnu7_mpi_times)
    g2_tiny_gnu7_mpi_speedup(g2_tiny_gnu7_mpi_times)
    g2_tiny_gnu7_mpi_efficiency(g2_tiny_gnu7_mpi_times)


if __name__ == "__main__":
    main()
