import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog

# Function to save grid state
def save_grid():
    grid_data = []
    for row in grid:
        grid_data.append([1 if btn["bg"] == "black" else 0 for btn in row])

    # Ask user for the save option
    save_option = simpledialog.askstring("Save Option", "Enter '1' to append to dataset, '2' to save as input data:")

    if save_option == '1':
        # Prompt for label
        label = simpledialog.askstring("Label Input", "Enter label (e.g., 'two' or 'not two'):")

        # Append to the dataset file
        with open("../Training Data/Training_Data.txt", "a") as f:
            f.write(label + ',' + ','.join(map(str, [item for sublist in grid_data for item in sublist])) + '\n')
        messagebox.showinfo("Save", "Grid pattern appended to dataset with label!")

    elif save_option == '2':
        # Save as plain text to be read by a C program
        with open("../InputPattern.txt", "w") as f:
            # Flatten the grid_data into a one-dimensional array
            one_d_array = [cell for row in grid_data for cell in row]
            f.write(','.join(map(str, one_d_array)) + '\n')  # Separate values by spaces
        messagebox.showinfo("Save", "Grid pattern saved")

    else:
        messagebox.showwarning("Invalid Input", "Please enter '1' or '2'.")

# Toggle the state of the grid cell
def toggle(btn):
    btn["bg"] = "black" if btn["bg"] == "white" else "white"

# Function to clear the grid
def clear_grid():
    for row in grid:
        for btn in row:
            btn.config(bg="white")

# Start toggle while clicking and dragging
def start_toggle(event):
    global toggled_buttons
    toggled_buttons = set()  # Reset the set of toggled buttons when click starts
    toggle(event.widget)  # Toggle the clicked button
    toggled_buttons.add(event.widget)  # Add the button to the set
    root.bind("<B1-Motion>", drag_toggle)

# Toggle while dragging
def drag_toggle(event):
    x, y = event.x_root, event.y_root  # Get global mouse position
    widget = root.winfo_containing(x, y)  # Find widget under mouse
    if widget and widget in [btn for row in grid for btn in row]:  # Ensure it's part of the grid
        if widget not in toggled_buttons:  # Only toggle if it hasn't been toggled already
            toggle(widget)
            toggled_buttons.add(widget)  # Add to the set of toggled buttons

# Stop toggling when mouse is released
def stop_toggle(event):
    root.unbind("<B1-Motion>")

# Initialize main window
root = tk.Tk()
root.title("8x8 Grid Pattern")

# Create 8x8 grid of buttons
grid = []
for i in range(8):
    row = []
    for j in range(8):
        btn = tk.Button(root, bg="white", width=4, height=2)
        btn.grid(row=i, column=j)
        btn.bind("<ButtonPress-1>", start_toggle)    # Start toggling on click
        btn.bind("<ButtonRelease-1>", stop_toggle)   # Stop toggling when released
        row.append(btn)
    grid.append(row)

# Save button
save_button = tk.Button(root, text="Save", command=save_grid)
save_button.grid(row=8, column=2, columnspan=2)

# Clear button
clear_button = tk.Button(root, text="Clear", command=clear_grid)
clear_button.grid(row=8, column=4, columnspan=2)

# Global set to keep track of toggled buttons during drag
toggled_buttons = set()

# Start the application
root.mainloop()
