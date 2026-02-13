import os 
import numpy as np
import matplotlib.pyplot as plt

def generate_degradation_plots():
    drift_values = np.array([0.0, 0.1, 0.2, 0.5])
    dqn_rewards = np.array([-45.0, -50.6, -57.6, -95.0])
    double_dqn_rewards = np.array([-45.0, -50.6, -57.6, -8512])

    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    plt.figure()
    plt.plot(drift_values, dqn_rewards)
    plt.xlabel("Target drift per step")
    plt.ylabel("Mean Episode Reward")
    plt.title("DQN Robustness under target drift")
    plt.savefig(os.path.join(results_dir, "dqn_degration_curve.png"))
    plt.close()


    plt.figure()
    plt.plot(drift_values, double_dqn_rewards)
    plt.xlabel("Target drift per step")
    plt.ylabel("Mean episode reward")
    plt.title("Double DQN Robustness under target drift")
    plt.savefig(os.path.join(results_dir, "double_dqn_degradation_curve.png"))
    plt.close()

    print("Plots Done")

if __name__ == "__main__":
    generate_degradation_plots()