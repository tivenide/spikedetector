def visualize_assignments(electrode_locations, neuron_locations, assignments):
    import matplotlib.pyplot as plt
    electrode_pos = pd.DataFrame({
        'x': electrode_locations[:, 1],
        'y': electrode_locations[:, 2],
        'z': electrode_locations[:, 0]
    })
    neuron_pos = pd.DataFrame({
        'x': neuron_locations[:, 1],
        'y': neuron_locations[:, 2],
        'z': neuron_locations[:, 0]
    })
    
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the channel positions
    ax.scatter(electrode_pos['x'], electrode_pos['y'], electrode_pos['z'], color='blue', label='Electrode Positions')

    # Plot the points of interest
    ax.scatter(neuron_pos['x'], neuron_pos['y'], neuron_pos['z'], color='red', label='Neuron Positions')

    # Plot the assignments
    for col in assignments.columns:
        assigned_positions = electrode_pos[assignments[col]]
        assigned_points = neuron_pos.loc[[col]] * len(assigned_positions)
        for pos, point in zip(assigned_positions.values, assigned_points.values):
            ax.plot([pos[0], point[0]], [pos[1], point[1]], [pos[2], point[2]], color='green')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def visualize_assignments_of_one_recording(path, distance_threshold):
    from tools import import_recording_h5, assign_neuron_locations_to_electrode_locations
    from visualization import visualize_assignments
    signal_raw, timestamps, ground_truth, electrode_locations, neuron_locations = import_recording_h5(path)
    assignments = assign_neuron_locations_to_electrode_locations(electrode_locations, neuron_locations, distance_threshold)
    visualize_assignments(electrode_locations, neuron_locations, assignments)