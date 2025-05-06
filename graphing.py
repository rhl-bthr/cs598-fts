import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

colors = ["#f1a200", "#cf4c32", "#995ec3"]

matplotlib.rcParams.update({'font.size': 18, 'font.family': "GillSans"})

# Plot 1 specific code starts
fig = plt.figure(figsize=(8, 6.5))
ax = plt.subplot(1, 1, 1)
i = 0

MODEL = "bert"
default_data = json.load(open("logs/" + MODEL + "-default.json"))
failure_data = json.load(open("logs/" + MODEL + "-fail.json"))
no_failure_data = json.load(open("logs/" + MODEL + "-no-fail.json"))

ax.plot(default_data["timestamps"], [x/1000 for x in default_data["iterations"]], color = colors[0], label = "No failure")
ax.plot(failure_data["timestamps"], [x/1000 for x in failure_data["iterations"]], color = colors[1], label = "Vanilla NCCL")
ax.plot(no_failure_data["timestamps"], [x/1000 for x in no_failure_data["iterations"]], color = colors[2], label = "Failure-Tolerant NCCL")

ax.set_xlabel("Time (s)")
ax.set_ylabel('# iterations (k)')


ax.tick_params(axis="both", direction="in", labelcolor="grey", width=3, length=6)

plt.ylim(bottom=0)
ax.grid(color="gray", alpha=0.3)

# Changing the color of the axis
ax.spines["bottom"].set(color="grey", alpha=0.3)
ax.spines["top"].set(color="grey", alpha=0.3)
ax.spines["left"].set(color="grey", alpha=0.3)
ax.spines["right"].set(color="grey", alpha=0.3)

ax.set_xlim([0, 240])
# ax.set_ylim([0, 200])
ax.set_ylim([0, 600])
ax.set_xticks([0, 60, 120, 180, 240])
# ax.set_yticks([0, 50, 100, 150, 200, 250])
ax.set_yticks([0, 150, 300, 450, 600])
plt.tight_layout()

# get handles
# handles, labels = ax.get_legend_handles_labels()
# handles = [h[0] for h in handles]
# legend = ax.legend(handles, labels, numpoints=1, fontsize="22", markerscale=0.7, handlelength=0.7, handletextpad=0.4, framealpha=0.3)

legend = plt.legend(fontsize="20", markerscale=0.7, handlelength=0.7, handletextpad=0.4, framealpha=0.3)

plt.savefig("plots/" + MODEL + ".png")
