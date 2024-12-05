import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def visualize_hub_assignment_solution(coordinates, best_hub_selection, customer_assignments):
    """
    Visualize the solution with hubs and their assigned nodes.
    
    Parameters:
    - coordinates (dict): A dictionary of node coordinates, e.g., {'Node1': (x1, y1), ...}.
    - best_hub_selection (list of bool): Indicates if a node is a hub (True) or not (False).
    - customer_assignments (list): List where each index corresponds to a node.
        - For customers: contains the index of the assigned hub.
        - For hubs: contains None.
    """
    import matplotlib.pyplot as plt

    # Extract node names and coordinates
    nodes = list(coordinates.keys())
    x_coords = [coordinates[node][0] for node in nodes]
    y_coords = [coordinates[node][1] for node in nodes]

    # Identify hubs and customers
    hub_indices = [i for i, is_hub in enumerate(best_hub_selection) if is_hub]
    customer_indices = [i for i, is_hub in enumerate(best_hub_selection) if not is_hub]

    # Plot all nodes
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, y_coords, c='gray', label='Customers', alpha=0.6)

    # Highlight hubs
    for hub_idx in hub_indices:
        plt.scatter(
            x_coords[hub_idx],
            y_coords[hub_idx],
            c='red',
            s=200,
            label=f'Hub {nodes[hub_idx]}',
            edgecolor='black'
        )

    # Draw lines from customers to assigned hubs
    for customer_idx in customer_indices:
        assigned_hub = customer_assignments[customer_idx]
        if assigned_hub is not None:
            plt.plot(
                [x_coords[customer_idx], x_coords[assigned_hub]],
                [y_coords[customer_idx], y_coords[assigned_hub]],
                linestyle='--',
                color='blue',
                alpha=0.5
            )

    # Add labels for all nodes
    for i, node in enumerate(nodes):
        plt.text(x_coords[i] + 1, y_coords[i] + 1, node, fontsize=8)

    plt.title('Hubs and Customer Assignments')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_vrp_solution(coordinates, best_solution_vrp, hubs, index_to_node):
    """
    Visualize the VRP solution with routes starting and ending at hubs.
    Each route from each hub is visualized with a different color, and all routes are included in the legend.
    
    Parameters:
    - coordinates (dict): A dictionary of node coordinates, e.g., {'Node1': (x1, y1), ...}.
    - best_solution_vrp (dict): Best VRP solution; mapping from hub index to list of routes.
    - hubs (list): List of hub indices.
    - index_to_node (dict): Mapping from node indices to node names.
    """

    
    plt.figure(figsize=(24, 16))
    # Plot all nodes
    for node_name, coord in coordinates.items():
        x, y = coord
        plt.scatter(x, y, c='gray', alpha=0.6)
        plt.text(x + 0.5, y + 0.5, node_name, fontsize=8)

    # For hubs, plot them with larger markers
    for hub_idx in hubs:
        hub_name = index_to_node[hub_idx]
        hub_coord = coordinates[hub_name]
        plt.scatter(hub_coord[0], hub_coord[1], c='black', s=200, label=f'Hub {hub_name}', edgecolors='black')

    # Now plot the routes
    # Use a color cycle to ensure each route has a unique color
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_idx = 0
    for hub_idx in hubs:
        hub_name = index_to_node[hub_idx]
        hub_coord = coordinates[hub_name]
        routes = best_solution_vrp[hub_idx]
        for route_idx, route in enumerate(routes):
            route_coords = [hub_coord]  # Start at hub
            for customer_idx in route:
                customer_name = index_to_node[customer_idx]
                customer_coord = coordinates[customer_name]
                route_coords.append(customer_coord)
            route_coords.append(hub_coord)  # Return to hub
            xs, ys = zip(*route_coords)
            color = color_cycle[color_idx % len(color_cycle)]
            plt.plot(xs, ys, marker='o', color=color, alpha=0.7, label=f'Hub {hub_name} Route {route_idx+1}')
            color_idx += 1  # Increment color index for next route

    plt.title('VRP Solution Routes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()
