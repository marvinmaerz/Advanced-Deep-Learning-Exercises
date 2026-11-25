import os
import matplotlib.pyplot as plt
from ex02_diffusion import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule

timesteps = 100
beta_start = 0.0001
beta_end = 0.02

# Compute beta values for all three schedules
lin_betas = linear_beta_schedule(beta_start, beta_end, timesteps).numpy()
cos_betas = cosine_beta_schedule(timesteps).numpy()
sig_betas = sigmoid_beta_schedule(beta_start, beta_end, timesteps).numpy()

# Plot beta schedules side by side and overlaid
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: all three beta schedules together
axes[0].plot(lin_betas, label="Linear")
axes[0].plot(cos_betas, label="Cosine")
axes[0].plot(sig_betas, label="Sigmoid")
axes[0].set_title(f"Beta Schedules (T={timesteps})")
axes[0].set_xlabel("Timestep t")
axes[0].set_ylabel("Beta")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: zoomed in on Linear vs. Sigmoid [beta_start, beta_end]
axes[1].plot(lin_betas, label="Linear")
axes[1].plot(sig_betas, label="Sigmoid")
axes[1].set_title("Linear vs. Sigmoid Zoomed")
axes[1].set_xlabel("Timestep t")
axes[1].set_ylabel("Beta")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Ensure the plot directory exists before saving
os.makedirs("plot", exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join("plot", "beta_schedules.png"), dpi=150)
plt.show()

# Compute per-step alpha values
lin_alphas = 1.0 - lin_betas
cos_alphas = 1.0 - cos_betas
sig_alphas = 1.0 - sig_betas

# Plot corresponding alpha schedules side by side
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: all three alpha schedules together
axes[0].plot(lin_alphas, label="Linear")
axes[0].plot(cos_alphas, label="Cosine")
axes[0].plot(sig_alphas, label="Sigmoid")
axes[0].set_title(f"Alpha Schedules (T={timesteps})")
axes[0].set_xlabel("Timestep t")
axes[0].set_ylabel("Alpha")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: zoomed in on Linear vs. Sigmoid alpha
axes[1].plot(lin_alphas, label="Linear")
axes[1].plot(sig_alphas, label="Sigmoid")
axes[1].set_title("Linear vs. Sigmoid Alpha Zoomed")
axes[1].set_xlabel("Timestep t")
axes[1].set_ylabel("Alpha")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join("plot", "alpha_schedules.png"), dpi=150)
plt.show()