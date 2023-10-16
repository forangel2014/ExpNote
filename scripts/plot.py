import matplotlib.pyplot as plt

def plot_training_curve(baselines, performance, steps, filename):
    plt.figure(dpi=1000)
    # Set up colors for baselines
    colors = ['gray', 'orange', 'pink']
    # Plot baselines as horizontal lines
    for i, (name, value) in enumerate(baselines.items()):
        plt.axhline(y=value, color=colors[i], linestyle='--', label=name)
    # Plot performance as a curve
    plt.plot(steps, performance, color='blue', label='ExpNote')
    # Set axis labels and title
    plt.xlabel('Training samples')
    plt.ylabel('Accuracy')
    plt.title('Training Curve')
    # Set x-axis limits to start at zero
    plt.xlim(0, max(steps))
    # Add legend
    plt.legend()
    # Save plot to file
    plt.savefig(filename)
    # Show plot
    plt.show()

if __name__ == "__main__":
    baselines = {"ExpNote(disabled)": 35, "CoT": 40, "Reflexion": 54}
    performance = [35, 41, 44, 45, 53, 54, 58, 59, 61]
    steps =       [0, 25, 50, 75, 100, 125, 150, 175, 200]
    plot_training_curve(baselines, performance, steps, filename="train.pdf")