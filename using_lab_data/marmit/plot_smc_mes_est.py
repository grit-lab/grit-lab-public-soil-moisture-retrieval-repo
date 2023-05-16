from anc_functions import get_L_eps_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Load Algodones soil moisture data and convert it to float
algodones = get_L_eps_data('./outputs/smc_output/alg_smc_output.csv').astype(float)
# Extract measured, estimated, and standard deviation of soil moisture content (SMC) for Algodones
algodones_smc_meas = algodones[:,0]
algodones_smc_est = algodones[:,1]
algodones_smc_std = algodones[:,2]

# Similar process repeated for Nevada data
nevada = get_L_eps_data('./outputs/smc_output/nev_smc_output.csv').astype(float)
nevada_smc_meas = nevada[:,0]
nevada_smc_est = nevada[:,1]
nevada_smc_std = nevada[:,2]

# Similar process repeated for Hogp data
hogp = get_L_eps_data('./outputs/smc_output/hogp_smc_output.csv').astype(float)
hogp_smc_meas = hogp[:,0]
hogp_smc_est = hogp[:,1]
hogp_smc_std = hogp[:,2]

# Similar process repeated for Hogb data
hogb = get_L_eps_data('./outputs/smc_output/hogb_smc_output.csv').astype(float)
hogb_smc_meas = hogb[:,0]
hogb_smc_est = hogb[:,1]
hogb_smc_std = hogb[:,2]

# Compute R2 score for combined measured and estimated SMC for all four samples
r2_val = r2_score(np.concatenate([algodones_smc_meas,nevada_smc_meas,hogp_smc_meas,hogb_smc_meas]),
         np.concatenate([algodones_smc_est, nevada_smc_est, hogp_smc_est, hogb_smc_est]))

print('R2-val for the lab measurements (all four samples) = ',r2_val)

# Compute the RMSE and NRMSE for combined measured and estimated SMC for all four samples
rmse = mean_squared_error(np.concatenate([algodones_smc_meas,nevada_smc_meas,hogp_smc_meas,hogb_smc_meas]),
                           np.concatenate([algodones_smc_est, nevada_smc_est, hogp_smc_est, hogb_smc_est]), squared=False)
nrmse = rmse/np.mean(np.concatenate([algodones_smc_meas,nevada_smc_meas,hogp_smc_meas,hogb_smc_meas]))

print('NRMSE for the lab measurements (all four samples) = ',nrmse)

# Create a new figure
fig = plt.figure()
# Change legend settings
plt.rcParams['legend.numpoints'] = 1

# Plot error bars and points for Algodones data
plt.errorbar(algodones_smc_meas,algodones_smc_est,algodones_smc_std,linestyle='',color='k')
plt.plot(algodones_smc_meas,algodones_smc_est,'o',color = 'white', markeredgecolor='red',markersize=8,markeredgewidth=2,label='ALG')

# Plot error bars and points for Hogb data
plt.errorbar(hogb_smc_meas,hogb_smc_est,hogb_smc_std,linestyle='',color='k')
plt.plot(hogb_smc_meas,hogb_smc_est,'o',color = 'white', markeredgecolor='magenta',markersize=8,markeredgewidth=2,label='HOGB')

# Plot error bars and points for Hogp data
plt.errorbar(hogp_smc_meas,hogp_smc_est,hogp_smc_std,linestyle='',color='k')
plt.plot(hogp_smc_meas,hogp_smc_est,'o',color = 'white', markeredgecolor='green',markersize=8,markeredgewidth=2,label='HOGP')

# Plot error bars and points for Nevada data
plt.errorbar(nevada_smc_meas,nevada_smc_est,nevada_smc_std,linestyle='',color='k')
plt.plot(nevada_smc_meas,nevada_smc_est,'o',color = 'white', markeredgecolor='blue',markersize=8,markeredgewidth=2,label='NEV')

# Add R^2 and NRMSE values as text on the plot
plt.text(0.5, 34, '$R^{2}$ = ' + str( np.round(r2_val, 3)), fontsize=12)
plt.text(0.5, 32, '$NRMSE$ = ' + str( np.round(nrmse, 3)), fontsize=12)

# Add legend to the plot, location is lower right
plt.legend(loc='lower right')

# Set labels for x and y axes
plt.xlabel('Measured SMC (%)', fontsize=15)
plt.ylabel('Estimated SMC (%)', fontsize=15)

# Set the limits for x-axis
plt.xlim([0,33])

# Adjust layout for better visualization
plt.tight_layout()

# Save the plot as a .png file with a resolution of 150 dpi, tight bounding box, and no padding around the image
fig.savefig('./outputs/image_output/all_smc_true_v_pred.png',dpi=150, bbox_inches='tight',pad_inches=0)
