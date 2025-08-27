plt.figure(8,6)
sorted_drifts = mean_slopes.sort_values()
plt.errorbar(len(sorted_drifts), sorted_drifts, yerr = std_slopes, )
